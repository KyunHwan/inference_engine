# inference_engine_utils

**Parent:** [env_actor](../README.md)

Shared utilities for inference algorithms.

## Table of contents

- [Modules](#modules)
- [compute_guided_prefix_weights](#compute_guided_prefix_weights)
- [guided_action_chunk_inference](#guided_action_chunk_inference)
- [Related docs](#related-docs)

## Modules

| File | Purpose |
|---|---|
| [`action_inpainting.py`](action_inpainting.py) | Action inpainting helpers — both the post-hoc numpy blender (`compute_guided_prefix_weights`) and the in-ODE PyTorch variant (`guided_action_chunk_inference`). |

Based on [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/pdf/2506.07339).

## compute_guided_prefix_weights

```python
def compute_guided_prefix_weights(
    delay_steps: int,
    executed: int,
    total: int,
    *,
    schedule: str = "exp",
) -> np.ndarray  # shape (total,) values in [0, 1]
```

**Used by**: `OpenPiPolicy.guided_inference` and `DsrlOpenpiPolicy.guided_inference`. Both call it then blend:

```python
weights = compute_guided_prefix_weights(est_delay, min_executed, chunk_size).reshape(-1, 1)
next_actions = prev_action * weights + pred_actions * (1.0 - weights)
```

Weight structure:

```
indices:    0      ...    start      ...    total - span    ...    total - 1
weight:    1.0    1.0    decay       decay       0.0           0.0     0.0
           ╰── keep old ──╯ ╰── blend ──╯           ╰── use new ──╯
                            (exp decay)
```

Where:

- `start = max(min(delay_steps, total), 0)` — how many initial actions to keep verbatim from the previous chunk.
- `span = max(min(executed, max(total - start, 1)), 1)` — width of the "use new" tail.
- The middle is `inter_vals = c_i * np.expm1(c_i) / (e - 1.0)` where `c_i = (total - span - indices) / (total - span - start + 1)`. This is the "exp" schedule; `"ones"` keeps everything from the previous chunk, `"zeros"` uses everything from the new prediction.

Edge case: if `delay_steps >= total`, returns `np.ones(total)` — the previous chunk fully wins, the new prediction is discarded. This is the conservative behavior when the inference latency is bigger than the chunk we have to work with.

## guided_action_chunk_inference

```python
def guided_action_chunk_inference(
    action_decoder: torch.nn.Module,
    cond_memory: torch.Tensor,
    discrete_semantic_input: torch.Tensor | None,
    prev_action_chunk: torch.Tensor,
    delay: int,
    executed_steps: int,
    num_ode_sim_steps: int,
    num_queries: int,
    action_dim: int,
    max_guidance_weight: float = 5.0,
    input_noise: torch.Tensor | None = None,
) -> torch.Tensor  # (batch, num_queries, action_dim)
```

A PyTorch-based, in-ODE variant. Instead of blending after the policy returns a chunk, it modifies the flow-matching ODE *during* denoising via VJP-based guidance, pulling the trajectory toward `prev_action_chunk` weighted by the same prefix schedule.

**Not currently called** by the shipped `OpenPiPolicy` or `DsrlOpenpiPolicy` — both use the post-hoc numpy blender. This function is included for future policies whose action decoder is a transformer-with-cross-attention and where in-ODE guidance is preferable (cleaner gradients, less post-hoc patching).

## Related docs

- [docs/concepts.md § Action inpainting](../../docs/concepts.md#action-inpainting)
- [docs/api.md § Action inpainting](../../docs/api.md#action-inpainting)
- [docs/glossary.md § Action inpainting](../../docs/glossary.md#action-inpainting)
