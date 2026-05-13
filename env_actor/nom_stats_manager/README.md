# nom_stats_manager

**Parent:** [env_actor](../README.md)

Data normalization and denormalization using pre-computed statistics from training. Keeps observations in the range the policy expects and converts actions back into the robot's scale.

## Table of contents

- [Structure](#structure)
- [DataNormalizationInterface](#datanormalizationinterface)
- [Normalization math](#normalization-math)
- [Statistics file format](#statistics-file-format)
- [Per-robot READMEs](#per-robot-readmes)
- [Extension points](#extension-points)
- [Related docs](#related-docs)

## Structure

```
nom_stats_manager/
├── data_normalization_interface.py    # Factory: robot name → normalization bridge
└── robots/
    ├── igris_b/
    │   ├── data_normalization_manager.py
    │   └── README.md
    └── igris_c/                       # Stub
```

## DataNormalizationInterface

Factory class in [`data_normalization_interface.py`](data_normalization_interface.py). Forwards every call to the robot-specific `DataNormalizationBridge`.

```python
normalizer = DataNormalizationInterface(robot="igris_b", data_stats=stats_dict)

# Used by policies:
normalizer.normalize_state(state_dict) -> state_dict     # z-score proprio + scale images
normalizer.normalize_action(action)    -> normalized
normalizer.denormalize_action(action)  -> robot_scale
```

The `data_stats` argument is the dict returned by [`RuntimeParams.read_stats_file()`](../runtime_settings_configs/robots/igris_b/inference_runtime_params.py). Each actor calls `read_stats_file()` once at startup and passes the result here.

## Normalization math

### State

**Proprioception** (z-score):
```
normalized = (state - mean) / (std + eps)        # eps = 1e-8
```

**Images** (scale to `[0, 1]`):
```
normalized = image.astype(float32) / 255.0
```

### Action

**Normalize**:
```
normalized = (action - mean) / (std + eps)
```

**Denormalize** (inverse z-score, used after the policy):
```
action = normalized_action * std + mean
```

## Statistics file format

The IGRIS_B bridge expects a pickle with these keys:

```python
{
    "observation.state":   {"mean": np.ndarray, "std": np.ndarray},
    "observation.current": {"mean": np.ndarray, "std": np.ndarray},
    "action":              {"mean": np.ndarray, "std": np.ndarray},
}
```

The bridge **concatenates** `observation.state.mean` and `observation.current.mean` along the last axis and uses the prefix of length `proprio.shape[-1]`. If your training data only used `observation.state`, you'll need to either: (a) regenerate the pickle with a zero-`std` `observation.current` block, (b) duplicate the same stats under both keys, or (c) edit the bridge to read only `observation.state`.

These statistics are typically computed once during training data preprocessing and saved alongside the model checkpoint.

## Per-robot READMEs

- [robots/igris_b/](robots/igris_b/README.md) — IGRIS_B specifics (state vs current concatenation, image handling).

## Extension points

To add normalization for a new robot:

1. Create `robots/your_robot/data_normalization_manager.py` with a `DataNormalizationBridge` class.
2. Implement `normalize_state(state: dict) -> dict`, `normalize_action(action) -> np.ndarray`, `denormalize_action(action) -> np.ndarray`.
3. Add an `elif robot == "your_robot":` branch in [`data_normalization_interface.py`](data_normalization_interface.py).
4. Ensure the normalization statistics pickle file matches the training data distribution.

Full walkthrough: [docs/walkthroughs/04_add_a_new_robot.md § Phase 4](../../docs/walkthroughs/04_add_a_new_robot.md#phase-4-normalization-bridge).

## Related docs

- [docs/api.md § DataNormalizationInterface](../../docs/api.md#datanormalizationinterface)
- [docs/configuration_cookbook.md § Change normalization stats](../../docs/configuration_cookbook.md#change-normalization-stats)
- [docs/glossary.md § Normalization stats](../../docs/glossary.md#normalization-stats)
