# dukron

PSGD optimizers for JAX, including Kron and Quad variants.

## Available Optimizers

- `kron` - PSGD Kron optimizer
- `quad` - PSGD Quad optimizer (ported from levanter by [@evanatyourservice](https://github.com/evanatyourservice))

## References

For Xi-Lin's original PSGD repo, see [psgd_torch](https://github.com/lixilinx/psgd_torch).

For JAX versions, see:
- [psgd_jax](https://github.com/evanatyourservice/psgd_jax) by [@evanatyourservice](https://github.com/evanatyourservice)
- [distributed_kron](https://github.com/evanatyourservice/distributed_kron) by [@evanatyourservice](https://github.com/evanatyourservice)
- [levanter](https://github.com/stanford-crfm/levanter) - source of the Quad implementation
