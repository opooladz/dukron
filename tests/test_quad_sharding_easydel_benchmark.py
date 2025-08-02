"""
Benchmark test comparing different sharding approaches for quad optimizer.
"""

import ray
from eformer.executor.ray import TpuAcceleratorConfig, execute

ray.init(runtime_env={
    'py_modules': ['/home/omead/dukron_pkg/dukron']
})

tpu_config = TpuAcceleratorConfig("v5p-16")

@execute(tpu_config)
@ray.remote
def benchmark_sharding():
    """Benchmark different sharding strategies for quad with 1B model."""
    import time
    import jax
    import jax.numpy as jnp
    import easydel as ed
    from dukron import quad
    from jax.sharding import PartitionSpec
    import optax
    
    logger = ed.utils.get_logger("QuadSharding-1B")
    logger.info(f"JAX backend: {jax.default_backend()}")
    logger.info(f"Devices: {jax.device_count()}")
    
    # Get mesh information
    devices = jax.devices()
    logger.info(f"Number of devices: {len(devices)}")
    
    # Create mesh for sharding
    mesh = jax.sharding.Mesh(devices, axis_names=("dp",))
    logger.info(f"Created mesh with shape: {mesh.shape}")
    
    # 1B model config
    config = ed.LlamaConfig(
        vocab_size=32768,      # Divisible by 8
        hidden_size=2048,      # 1B+ size
        num_attention_heads=16,
        num_hidden_layers=20,  # 20 layers for ~1B params
        intermediate_size=5632,
        max_position_embeddings=512,
        attn_mechanism=ed.AttentionMechanisms.VANILLA,
        gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
        sharding_axis_dims=(1, -1, 1, 1, 1),
    )
    
    logger.info("Creating model...")
    with mesh:
        model = ed.LlamaForCausalLM(
            config=config,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            rngs=ed.Rngs(0),
        ).shard_model()
        
        state = model.to_state()
        params = state.graphstate
        param_count = sum(x.size for x in jax.tree.leaves(params))
        logger.info(f"Model parameters: {param_count:,} ({param_count/1e9:.2f}B)")
        
        # Create gradients
        grads = jax.tree.map(lambda x: jnp.ones_like(x) * 0.001, params)
        
        # Results dictionary
        results = {}
        
        # Test 1: Default sharding (no explicit preconditioner sharding)
        logger.info("\n=== TEST 1: DEFAULT SHARDING ===")
        logger.info("Using default sharding inferred from parameters...")
        
        tx_default = quad(
            learning_rate=1e-4,
            b1=0.9,
            weight_decay=0.01,
            preconditioner_lr=0.1,
            preconditioner_init_scale=1.0,
            precond_dtype=jnp.float32,
            partition_grads_into_blocks=True,
            block_size=1024,
        )
        
        logger.info("Initializing QUAD with default sharding...")
        opt_state_default = tx_default.init(params)
        logger.info("✓ Default sharding initialized!")
        
        # Warmup
        logger.info("Warmup (3 steps)...")
        for i in range(3):
            start = time.time()
            updates, opt_state_default = tx_default.update(grads, opt_state_default, params)
            jax.block_until_ready(updates)
            warmup_time = time.time() - start
            logger.info(f"  Warmup {i+1}: {warmup_time:.3f}s")
        
        # Benchmark
        logger.info("Benchmarking (10 steps)...")
        default_times = []
        for i in range(10):
            start = time.time()
            updates, opt_state_default = tx_default.update(grads, opt_state_default, params)
            jax.block_until_ready(updates)
            step_time = time.time() - start
            default_times.append(step_time)
            if i % 5 == 0:
                logger.info(f"  Step {i}: {step_time:.3f}s")
        
        default_avg = sum(default_times) / len(default_times)
        results["default"] = default_avg
        
        # Test 2: Simple first-axis sharding
        logger.info("\n=== TEST 2: SIMPLE FIRST-AXIS SHARDING ===")
        logger.info("Using PartitionSpec('dp', None) - shard Q matrices on first axis only...")
        
        tx_simple = quad(
            learning_rate=1e-4,
            b1=0.9,
            weight_decay=0.01,
            preconditioner_lr=0.1,
            preconditioner_init_scale=1.0,
            precond_dtype=jnp.float32,
            partition_grads_into_blocks=True,
            block_size=1024,
            preconditioner_sharding=PartitionSpec("dp", None),
        )
        
        logger.info("Initializing QUAD with simple sharding...")
        opt_state_simple = tx_simple.init(params)
        logger.info("✓ Simple sharding initialized!")
        
        # Warmup
        logger.info("Warmup (3 steps)...")
        for i in range(3):
            start = time.time()
            updates, opt_state_simple = tx_simple.update(grads, opt_state_simple, params)
            jax.block_until_ready(updates)
            warmup_time = time.time() - start
            logger.info(f"  Warmup {i+1}: {warmup_time:.3f}s")
        
        # Benchmark
        logger.info("Benchmarking (10 steps)...")
        simple_times = []
        for i in range(10):
            start = time.time()
            updates, opt_state_simple = tx_simple.update(grads, opt_state_simple, params)
            jax.block_until_ready(updates)
            step_time = time.time() - start
            simple_times.append(step_time)
            if i % 5 == 0:
                logger.info(f"  Step {i}: {step_time:.3f}s")
        
        simple_avg = sum(simple_times) / len(simple_times)
        results["simple"] = simple_avg
        
        # Test 3: No sharding (replicated)
        logger.info("\n=== TEST 3: NO SHARDING (REPLICATED) ===")
        logger.info("Using PartitionSpec(None, None) - fully replicated Q matrices...")
        
        tx_none = quad(
            learning_rate=1e-4,
            b1=0.9,
            weight_decay=0.01,
            preconditioner_lr=0.1,
            preconditioner_init_scale=1.0,
            precond_dtype=jnp.float32,
            partition_grads_into_blocks=True,
            block_size=1024,
            preconditioner_sharding=PartitionSpec(None, None),
        )
        
        logger.info("Initializing QUAD with no sharding...")
        opt_state_none = tx_none.init(params)
        logger.info("✓ No sharding initialized!")
        
        # Warmup
        logger.info("Warmup (3 steps)...")
        for i in range(3):
            start = time.time()
            updates, opt_state_none = tx_none.update(grads, opt_state_none, params)
            jax.block_until_ready(updates)
            warmup_time = time.time() - start
            logger.info(f"  Warmup {i+1}: {warmup_time:.3f}s")
        
        # Benchmark
        logger.info("Benchmarking (10 steps)...")
        none_times = []
        for i in range(10):
            start = time.time()
            updates, opt_state_none = tx_none.update(grads, opt_state_none, params)
            jax.block_until_ready(updates)
            step_time = time.time() - start
            none_times.append(step_time)
            if i % 5 == 0:
                logger.info(f"  Step {i}: {step_time:.3f}s")
        
        none_avg = sum(none_times) / len(none_times)
        results["none"] = none_avg
        
        # Test Adam for baseline comparison
        logger.info("\n=== ADAM BASELINE ===")
        tx_adam = optax.adam(1e-4)
        opt_state_adam = tx_adam.init(params)
        
        logger.info("Adam benchmark (10 steps)...")
        adam_times = []
        for i in range(10):
            start = time.time()
            updates, opt_state_adam = tx_adam.update(grads, opt_state_adam)
            jax.block_until_ready(updates)
            step_time = time.time() - start
            adam_times.append(step_time)
        
        adam_avg = sum(adam_times) / len(adam_times)
        results["adam"] = adam_avg
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("SHARDING COMPARISON RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Model size: {param_count/1e9:.2f}B parameters")
        logger.info(f"\nAverage time per step:")
        logger.info(f"  Default sharding:      {results['default']:.4f}s")
        logger.info(f"  Simple sharding:       {results['simple']:.4f}s")
        logger.info(f"  No sharding:           {results['none']:.4f}s")
        logger.info(f"  Adam (baseline):       {results['adam']:.4f}s")
        
        logger.info(f"\nSpeedup vs no sharding:")
        logger.info(f"  Default: {results['none']/results['default']:.2f}x")
        logger.info(f"  Simple:  {results['none']/results['simple']:.2f}x")
        
        logger.info(f"\nOverhead vs Adam:")
        logger.info(f"  Default: {results['default']/results['adam']:.2f}x slower")
        logger.info(f"  Simple:  {results['simple']/results['adam']:.2f}x slower")
        logger.info(f"  None:    {results['none']/results['adam']:.2f}x slower")
        
        # Find best approach
        best_sharding = min(results.items(), key=lambda x: x[1] if x[0] != "adam" else float('inf'))
        logger.info(f"\n✅ Best sharding approach: {best_sharding[0]} ({best_sharding[1]:.4f}s per step)")
        logger.info(f"✅ All sharding approaches work with EasyDeL!")
        
        return {
            "model_size": f"{param_count/1e9:.2f}B",
            **results,
            "best_sharding": best_sharding[0],
        }


if __name__ == "__main__":
    print("Running sharding comparison benchmark...")
    result = benchmark_sharding()
    print(f"\nResult: {result}")