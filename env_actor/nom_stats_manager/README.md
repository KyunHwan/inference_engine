# nom_stats_manager

Data normalization and denormalization using pre-computed statistics from training. Ensures that observations fed to the policy and actions returned from the policy are in the correct scale.

## Structure

```
nom_stats_manager/
├── data_normalization_interface.py    # Factory: robot name → normalization bridge
└── robots/
    ├── igris_b/
    │   └── data_normalization_manager.py   # IGRIS_B normalization logic
    └── igris_c/                            # Stub
```

## DataNormalizationInterface

Factory in `data_normalization_interface.py` that selects the robot-specific normalization bridge:

```python
# Usage in inference actors:
normalizer = DataNormalizationInterface(robot="igris_b", runtime_params=params)
```

## Normalization Logic

### State Normalization

**Proprioception** (z-score normalization):
```
normalized = (state - mean) / (std + epsilon)
```

**Images** (scale to [0, 1]):
```
normalized = image / 255.0
```

### Action Denormalization

Inverse z-score to convert policy output back to robot action space:
```
action = normalized_action * std + mean
```

## Statistics File Format

The normalization statistics are loaded from a pickle file (path configured in `inference_runtime_params.json` as `norm_stats_file_path`). Expected format:

```python
{
    "observation.state": {
        "mean": np.ndarray,  # shape (state_dim,)
        "std": np.ndarray,   # shape (state_dim,)
    },
    "action": {
        "mean": np.ndarray,  # shape (action_dim,)
        "std": np.ndarray,   # shape (action_dim,)
    },
    # Optional additional keys:
    "observation.current": {
        "mean": np.ndarray,
        "std": np.ndarray,
    },
}
```

These statistics are typically computed during training data preprocessing and saved alongside the model checkpoint.

## Adding Normalization for a New Robot

1. Create `robots/your_robot/data_normalization_manager.py` with a `DataNormalizationBridge` class
2. Implement `normalize_state(state: dict) -> dict` and `denormalize_action(action: np.ndarray) -> np.ndarray`
3. Update the factory in `data_normalization_interface.py` to include the new robot
4. Ensure the normalization statistics pickle file matches the training data distribution
