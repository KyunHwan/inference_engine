# inference_engine_utils

Shared utilities for inference algorithms.

## Modules

### `action_inpainting.py`

Implements action inpainting — the technique for smoothly blending consecutive action chunks during real-time control. Based on [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/pdf/2506.07339).

**Two functions for different use cases:**

#### `compute_guided_prefix_weights(delay_steps, executed, total, schedule="exp")`

NumPy-based weight computation. Used by `OpenPiPolicy.guided_inference()` when the model architecture doesn't support native action inpainting during denoising.

**Parameters:**
- `delay_steps` (int): Estimated inference latency in control steps
- `executed` (int): Minimum number of actions executed before re-planning
- `total` (int): Total action chunk size
- `schedule` (str): Weight schedule — `"exp"` (exponential decay), `"ones"` (keep all old), `"zeros"` (replace all)

**Returns:** `np.ndarray` of shape `(total,)` with values in [0, 1]

**Weight structure:**
```
[1.0, 1.0, ..., decay, decay, ..., 0.0, 0.0, ...]
 ╰── keep old ──╯  ╰── blend ──╯   ╰── use new ──╯
      (delay)        (transition)     (pure prediction)
```

**Usage in guided inference:**
```python
weights = compute_guided_prefix_weights(est_delay, min_executed, chunk_size).reshape(-1, 1)
blended = prev_action * weights + new_prediction * (1.0 - weights)
```

#### `guided_action_chunk_inference(...)`

PyTorch-based guided denoising for flow-matching policies with cross-attention transformer action decoders. Performs ODE simulation with VJP-based guidance toward the previous action chunk.

This is a more advanced function that modifies the denoising process itself (rather than post-hoc blending). It requires the action decoder to accept `time`, `noise`, `memory_input`, and optional `discrete_semantic_input` arguments.
