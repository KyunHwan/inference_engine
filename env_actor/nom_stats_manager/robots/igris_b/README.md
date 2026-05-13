# nom_stats_manager/robots/igris_b

**Parent:** [nom_stats_manager](../../README.md)

IGRIS_B `DataNormalizationBridge`. Z-scores proprio (using state + current stats), scales images, denormalizes actions.

## Files

| File | Purpose |
|---|---|
| [`data_normalization_manager.py`](data_normalization_manager.py) | `DataNormalizationBridge` class. Accepts the loaded `norm_stats` dict in its constructor. |

## Methods

### `normalize_state(state)`

```python
state_mean = concat([norm_stats["observation.state"]["mean"],
                     norm_stats["observation.current"]["mean"]], axis=-1)
state_std  = concat([norm_stats["observation.state"]["std"],
                     norm_stats["observation.current"]["std"]], axis=-1)

proprio_len = state["proprio"].shape[-1]
state["proprio"] = (state["proprio"] - state_mean[:proprio_len]) / (state_std[:proprio_len] + 1e-8)

# Every non-"proprio" key (i.e. each camera image) is scaled by /255.0.
for key in state:
    if key != "proprio":
        state[key] = state[key] / 255.0

return state
```

The slice `[:proprio_len]` is what lets the bridge handle the case where `IGRIS_B_STATE_KEYS` includes only position keys (`proprio_len = 24`) vs the full position+current set (`proprio_len = 48`). The stats pickle is sized for the full 48; we just slice off the prefix we need.

> **Caveat.** If your stats pickle does not have `observation.current`, this will raise `KeyError`. Either populate that block (with zeros for mean and ones for std works, since the prefix slice never touches it), or remove the concatenation here.

### `normalize_action(action)` / `denormalize_action(action)`

Standard z-score and its inverse, using `norm_stats["action"]["mean"]` and `["std"]`. `denormalize_action` is the one most policies call at the end of inference (or have the wrapper call internally).

## Stats schema (what the IGRIS_B pickle must contain)

```python
{
    "observation.state":   {"mean": np.ndarray (D1,), "std": np.ndarray (D1,)},
    "observation.current": {"mean": np.ndarray (D2,), "std": np.ndarray (D2,)},
    "action":              {"mean": np.ndarray (24,), "std": np.ndarray (24,)},
}
```

With the default IGRIS_B config (`proprio_state_dim = 24`), `D1 + D2` must be `≥ 24`. With the full state+current key list (uncomment the cur keys in [init_params.py](../../../runtime_settings_configs/robots/igris_b/init_params.py)), set `proprio_state_dim = 48` and ensure `D1 + D2 ≥ 48`.

The training pipeline (in the trainer submodule) writes this pickle as `dataset_stats.pkl` during data prep; the path is whatever you point `norm_stats_file_path` at in the runtime JSON.

## Related docs

- [docs/api.md § DataNormalizationInterface](../../../../docs/api.md#datanormalizationinterface)
- [docs/configuration_cookbook.md § Change normalization stats](../../../../docs/configuration_cookbook.md#change-normalization-stats)
- [trainer docs on data normalization](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/docs/04_concepts.md) — the training side that produces the pickle.
