"""
Simple speed test for quad with 1B+ model.
"""

import ray
from eformer.executor.ray import TpuAcceleratorConfig, execute

ray.init(runtime_env={
    'py_modules': ['/home/omead/dukron_pkg/dukron']
})

tpu_config = TpuAcceleratorConfig("v5p-16")

@execute(tpu_config)
@ray.remote
def benchmark_1b():
    """Benchmark quad with 1B model."""
    import time
    import jax
    import jax.numpy as jnp
    import easydel as ed
    from dukron import quad
    import optax
    
    logger = ed.utils.get_logger("Quad-1B")
    logger.info(f"JAX backend: {jax.default_backend()}")
    logger.info(f"Devices: {jax.device_count()}")
    
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
    
    # Test Quad
    logger.info("\nTesting QUAD optimizer...")
    tx_quad = quad(
        learning_rate=1e-4,
        b1=0.9,
        weight_decay=0.01,
        preconditioner_lr=0.1,
        preconditioner_init_scale=1.0,
        precond_dtype=jnp.float32,
        partition_grads_into_blocks=True,
        block_size=1024,
    )
    
    logger.info("Initializing QUAD...")
    opt_state = tx_quad.init(params)
    logger.info("✓ QUAD initialized successfully!")
    
    # Simple update test
    logger.info("\nRunning update steps...")
    grads = jax.tree.map(lambda x: jnp.ones_like(x) * 0.001, params)
    
    # Warmup
    logger.info("Warmup (3 steps)...")
    warmup_times = []
    for i in range(3):
        start = time.time()
        updates, opt_state = tx_quad.update(grads, opt_state, params)
        jax.block_until_ready(updates)
        warmup_time = time.time() - start
        warmup_times.append(warmup_time)
        logger.info(f"  Warmup {i+1}: {warmup_time:.3f}s")
    
    # Benchmark
    logger.info("\nBenchmarking (20 steps)...")
    step_times = []
    for i in range(20):
        start = time.time()
        updates, opt_state = tx_quad.update(grads, opt_state, params)
        jax.block_until_ready(updates)
        step_time = time.time() - start
        step_times.append(step_time)
        if i % 5 == 0:
            logger.info(f"  Step {i}: {step_time:.3f}s")
    
    avg_time = sum(step_times) / len(step_times)
    min_time = min(step_times)
    
    logger.info(f"\nQuad Results:")
    logger.info(f"  Average time: {avg_time:.4f}s per step")
    logger.info(f"  Min time: {min_time:.4f}s per step")
    logger.info(f"  Steps/second: {1/avg_time:.2f}")
    
    # Test Adam for comparison
    logger.info("\nTesting ADAM optimizer...")
    tx_adam = optax.adam(1e-4)
    opt_state_adam = tx_adam.init(params)
    
    # Adam benchmark
    logger.info("Adam benchmark (20 steps)...")
    adam_times = []
    for i in range(20):
        start = time.time()
        updates, opt_state_adam = tx_adam.update(grads, opt_state_adam)
        jax.block_until_ready(updates)
        step_time = time.time() - start
        adam_times.append(step_time)
    
    adam_avg = sum(adam_times) / len(adam_times)
    logger.info(f"\nAdam Results:")
    logger.info(f"  Average time: {adam_avg:.4f}s per step")
    logger.info(f"  Steps/second: {1/adam_avg:.2f}")
    
    # Compare
    speedup = adam_avg / avg_time
    logger.info(f"\nComparison:")
    if speedup > 1:
        logger.info(f"  Quad is {speedup:.2f}x faster than Adam")
    else:
        logger.info(f"  Adam is {1/speedup:.2f}x faster than Quad")
    
    logger.info(f"\n✅ Benchmark complete!")
    logger.info(f"✅ Quad works with 1B+ model on TPU with FSDP!")
    
    return {
        "model_size": f"{param_count/1e9:.2f}B",
        "quad_time": avg_time,
        "adam_time": adam_avg,
        "speedup": speedup
    }


if __name__ == "__main__":
    print("Running 1B model benchmark...")
    result = benchmark_1b()
    print(f"\nResult: {result}")