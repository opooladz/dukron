"""Quad optimizer with simplified sharding scheme.

This implements the simple sharding approach where:
- Q matrices are always sharded on first dimension only
- Only shard if Q dimension >= min_size_to_shard
- Results in Q.T @ Q being [None, sharded] @ [sharded, None]
"""

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec, Mesh
from typing import Optional, Union, Callable, Any
import string

# Import base quad functionality
from dukron._quad import (
    scale_by_quad,
    quad as quad_original,
    _init_Q_exprs as _init_Q_exprs_original,
    EINSUM_OPTIM,
    get_precond_lr,
    _norm_lower_bound,
    _update_precond,
)


def _get_mesh_info():
    """Get current mesh information from JAX context."""
    try:
        # Try to get mesh from current context
        from jax.experimental import mesh_utils
        devices = jax.devices()
        if len(devices) > 1:
            # For EasyDeL, the mesh is typically along "dp" axis
            mesh = Mesh(devices, axis_names=("dp",))
            return mesh, "dp", len(devices)
    except Exception:
        pass
    return None, None, 1


def _init_Q_exprs_simple_sharding(
    t_shape,
    scale,
    have_qs_sharding,
    dtype,
    precond_sharding=None,
    param_sharding=None,
    existing_Q=None,
    existing_L=None,
    min_size_to_shard=512,
):
    """Initialize Q matrices with simple first-axis-only sharding."""
    import string
    from jax.lax import with_sharding_constraint
    
    letters = string.ascii_lowercase
    
    # Get mesh information
    mesh, fsdp_axis_name, fsdp_size = _get_mesh_info()
    
    # Determine which dimensions use diagonal vs triangular matrices
    total_params = sum(t_shape) if t_shape else 1
    dim_diag = [total_params < 512]  # All diagonal if too small
    if total_params >= 512 and len(t_shape) > 0:
        # Make smaller dimensions diagonal
        num_diag = max(1, len(t_shape) - 1)
        sorted_dims = sorted(enumerate(t_shape), key=lambda x: x[1])
        dim_diag = [True] * len(t_shape)
        for i, (dim_idx, size) in enumerate(sorted_dims[:-num_diag]):
            dim_diag[dim_idx] = False
    
    # Initialize Q and L
    Q = [] if existing_Q is None else existing_Q
    L = [] if existing_L is None else existing_L
    piece1P, piece2P, piece3P, piece4P = ([], [], "", "")
    exprGs = []
    
    params_specs = param_sharding
    if param_sharding is None:
        params_specs = PartitionSpec(*((None,) * len(t_shape)))
    sharding_out = [None] * len(t_shape)
    if have_qs_sharding:
        sharding_out = [PartitionSpec(None)] * len(t_shape)
    
    for i, (size, dim_d, dim_sh) in enumerate(zip(t_shape, dim_diag, params_specs)):
        if existing_L is None:
            L.append(jnp.zeros([], dtype=jnp.float32))
            
        if dim_d:
            # Use diagonal matrix as preconditioner for this dim
            if existing_Q is None:
                q = scale * jnp.ones(size, dtype=dtype)
                Q.append(q)
            
            sym = letters[i + 13]
            piece1P.append(sym)
            piece2P.append(sym)
            piece3P = piece3P + sym
            piece4P = piece4P + sym
            sub = ''.join(letters[i + 13] if j == i else letters[j] for j in range(len(t_shape)))
            exprGs.append(f"{sub},{sub}->{sym}")
        else:
            # Use triangular matrix as preconditioner for this dim
            q_sharding = None
            
            # Simple sharding logic: only shard first dimension if conditions met
            if have_qs_sharding and mesh is not None and fsdp_axis_name is not None:
                if size % fsdp_size == 0 and size >= min_size_to_shard:
                    q_sharding = PartitionSpec(fsdp_axis_name, None)
                else:
                    q_sharding = PartitionSpec(None, None)
            elif precond_sharding is not None:
                # Use provided preconditioner sharding
                q_sharding = precond_sharding
            
            if have_qs_sharding:
                sharding_out[i] = q_sharding
            
            if existing_Q is None:
                q = scale * jnp.eye(size, dtype=dtype)
                if have_qs_sharding and q_sharding is not None:
                    q = with_sharding_constraint(q, q_sharding)
                Q.append(q)
            
            a = letters[i]
            b = letters[i + 13]
            c = letters[i + 26]
            piece1P.append(a + b)
            piece2P.append(a + c)
            piece3P = piece3P + c
            piece4P = piece4P + b
            sub1 = ''.join(letters[i + 13] if j == i else letters[j] for j in range(len(t_shape)))
            sub2 = ''.join(letters[i + 26] if j == i else letters[j] for j in range(len(t_shape)))
            exprGs.append(f"{sub1},{sub2}->{b}{c}")
    
    exprP = ",".join(piece1P) + "," + ",".join(piece2P) + "," + piece3P + "->" + piece4P
    exprGs = tuple(exprGs)
    if existing_Q is not None:
        return (exprP, exprGs), sharding_out
    return Q, L, (exprP, exprGs), sharding_out


def scale_by_quad_simple_sharding(
    b1: float = 0.95,
    max_size_dense: int = 8192,
    max_skew_dense: float = 1.0,
    preconditioner_lr: float = 0.7,
    preconditioner_init_scale: float = 1.0,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_dtype: Optional[Union[str, jnp.dtype]] = None,
    scanned_layers: Optional[Any] = None,
    lax_map_scanned_layers: bool = False,
    lax_map_batch_size: int = 8,
    do_merge_small_dims: bool = False,
    target_merged_dim_size: int = 4096,
    partition_grads_into_blocks: bool = False,
    block_size: int = 512,
    params_sharding: Optional[Any] = None,
    preconditioner_sharding: Optional[PartitionSpec] = None,
    min_size_to_shard: int = 512,
    **kwargs,
):
    """Scale by quad with simple first-axis sharding.
    
    Additional Args:
        min_size_to_shard: Minimum Q dimension size to apply sharding (default: 512)
    """
    # Override the _init_Q_exprs function in the module temporarily
    import dukron._quad as quad_module
    original_init_Q_exprs = quad_module._init_Q_exprs
    
    def patched_init_Q_exprs(*args, **kwargs):
        # Extract relevant args
        t_shape = args[0]
        scale = args[1]
        dim_diag = args[2]
        dtype = args[3]
        existing_Q = kwargs.get('existing_Q', None)
        existing_L = kwargs.get('existing_L', None)
        precond_sharding = kwargs.get('precond_sharding', None)
        param_sharding = kwargs.get('param_sharding', None)
        
        # Determine if we have sharding
        have_qs_sharding = (params_sharding is not None or 
                          preconditioner_sharding is not None)
        
        return _init_Q_exprs_simple_sharding(
            t_shape, scale, have_qs_sharding, dtype,
            precond_sharding, param_sharding,
            existing_Q, existing_L,
            min_size_to_shard
        )
    
    # Temporarily patch the module
    quad_module._init_Q_exprs = patched_init_Q_exprs
    
    try:
        # Call original scale_by_quad with patched init function
        result = scale_by_quad(
            b1=b1,
            max_size_dense=max_size_dense,
            max_skew_dense=max_skew_dense,
            preconditioner_lr=preconditioner_lr,
            preconditioner_init_scale=preconditioner_init_scale,
            mu_dtype=mu_dtype,
            precond_dtype=precond_dtype,
            scanned_layers=scanned_layers,
            lax_map_scanned_layers=lax_map_scanned_layers,
            lax_map_batch_size=lax_map_batch_size,
            do_merge_small_dims=do_merge_small_dims,
            target_merged_dim_size=target_merged_dim_size,
            partition_grads_into_blocks=partition_grads_into_blocks,
            block_size=block_size,
            params_sharding=params_sharding,
            preconditioner_sharding=preconditioner_sharding,
            **kwargs,
        )
    finally:
        # Restore original function
        quad_module._init_Q_exprs = original_init_Q_exprs
    
    return result


def quad_simple_sharding(
    learning_rate: Union[float, Callable[[int], float]] = 0.0003,
    b1: float = 0.95,
    weight_decay: float = 0.5,
    weight_decay_mask: Optional[Any] = None,
    max_size_dense: int = 8192,
    max_skew_dense: float = 1.0,
    preconditioner_lr: float = 0.7,
    preconditioner_init_scale: float = 1.0,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_dtype: Optional[Union[str, jnp.dtype]] = None,
    scanned_layers: Optional[Any] = None,
    lax_map_scanned_layers: bool = False,
    lax_map_batch_size: int = 8,
    do_merge_small_dims: bool = False,
    target_merged_dim_size: int = 4096,
    partition_grads_into_blocks: bool = False,
    block_size: int = 512,
    params_sharding: Optional[Any] = None,
    preconditioner_sharding: Optional[PartitionSpec] = None,
    min_size_to_shard: int = 512,
):
    """Quad optimizer with simple first-axis sharding scheme.
    
    This implements the simple sharding approach where:
    - Q matrices are always sharded on first dimension only
    - Only shard if Q dimension >= min_size_to_shard
    - Results in Q.T @ Q being [None, sharded] @ [sharded, None]
    
    Additional Args:
        min_size_to_shard: Minimum Q dimension size to apply sharding (default: 512)
    """
    from optax._src import transform
    from optax._src.combine import chain
    
    optimizer = [
        scale_by_quad_simple_sharding(
            b1=b1,
            max_size_dense=max_size_dense,
            max_skew_dense=max_skew_dense,
            preconditioner_lr=preconditioner_lr,
            preconditioner_init_scale=preconditioner_init_scale,
            mu_dtype=mu_dtype,
            precond_dtype=precond_dtype,
            scanned_layers=scanned_layers,
            lax_map_scanned_layers=lax_map_scanned_layers,
            lax_map_batch_size=lax_map_batch_size,
            do_merge_small_dims=do_merge_small_dims,
            target_merged_dim_size=target_merged_dim_size,
            partition_grads_into_blocks=partition_grads_into_blocks,
            block_size=block_size,
            params_sharding=params_sharding,
            preconditioner_sharding=preconditioner_sharding,
            min_size_to_shard=min_size_to_shard,
        )
    ]
    if weight_decay > 0.0:
        optimizer.append(transform.add_decayed_weights(weight_decay, weight_decay_mask))
    optimizer.append(transform.scale_by_learning_rate(learning_rate))
    return chain(*optimizer)