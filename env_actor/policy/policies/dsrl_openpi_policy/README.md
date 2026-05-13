# dsrl_openpi_policy

**Parent:** [policies](../README.md)

A four-component pipeline that combines a DSRL (Diffusion Steering for Reinforcement Learning) noise actor with a frozen OpenPI/Pi0.5 backbone. Default policy for [run_inference.py](../../../../run_inference.py).

## Table of contents

- [What is DSRL?](#what-is-dsrl)
- [Files](#files)
- [Pipeline](#pipeline)
- [Component YAML fields](#component-yaml-fields)
- [Checkpoint layout](#checkpoint-layout)
- [Related docs](#related-docs)

## What is DSRL?

DSRL is a way to **fine-tune** a large pretrained VLA (here, OpenPI/Pi0.5) with reinforcement learning without modifying the pretrained weights. The trick: a small network learns to produce the *noise* that the pretrained flow-matching decoder is conditioned on. Steering the noise steers the policy's action distribution; the pretrained weights stay frozen.

In this repo, four components compose the policy:

1. **backbone** — Resnet34Group-style image encoder. Outputs spatial features per camera.
2. **noise_processor** — combines image features + proprio into a flat latent vector.
3. **noise_actor** — turns the flat latent into structured noise (`(1, 50, 32)` here).
4. **openpi_model** — the frozen Pi0.5 wrapper, takes the noise as conditioning and emits the action chunk.

The DSRL components are loaded from `params.checkpoint_path` in this policy's YAML; the OpenPI weights are loaded by the OpenPI component itself from `ckpt_dir` in its own YAML.

## Files

| File | Purpose |
|---|---|
| [`dsrl_openpi_policy.py`](dsrl_openpi_policy.py) | `DsrlOpenpiPolicy` class. Implements `Policy`. Owns the 4-step forward pass. |
| [`dsrl_openpi_policy.yaml`](dsrl_openpi_policy.yaml) | Top-level YAML. `policy.type: dsrl_openpi_policy`. Lists 4 component YAMLs. |
| [`components/backbone.yaml`](components/backbone.yaml) | `_type_: dsrl_img_encoder`; image backbone. |
| [`components/noise_processor.yaml`](components/noise_processor.yaml) | `_type_: dsrl_noise_actor_processor`; flattens features + proprio. |
| [`components/noise_actor.yaml`](components/noise_actor.yaml) | `_type_: dsrl_noise_latent_actor`; produces structured noise. |
| [`components/openpi_model.yaml`](components/openpi_model.yaml) | `_type_: openpi_batched`; the frozen Pi0.5 wrapper. |

## Pipeline

```
obs["proprio"] (50, 24) ─────────────────────────────────────────────────────────────┐
obs["head"/"left"/"right"] (1, 3, 240, 320) ─→ backbone ─→ {cam: (1, 512, H', W')}   │
                                                                       │             │
                                                                       ▼             ▼
                                                        noise_processor(features, proprio_norm)
                                                                       │
                                                                       ▼
                                                        flat_features (1, D)
                                                                       │
                                                                       ▼
                                                              noise_actor(flat) → (1, 50, 32)
                                                                       │
                                                                       ▼
openpi_model(observation=raw_obs, noise=structured_noise) ──→ (1, action_horizon, action_dim)
                                                                       │
                                                                       ▼
                                                              [0] → (action_horizon, action_dim)
```

Note that the proprio normalization happens *inside* this policy (via `data_normalization_interface.normalize_state`), and only the DSRL components see the normalized version. OpenPI receives the raw `obs["proprio"][0:1]` because its training-time pipeline normalizes internally.

## Component YAML fields

### `dsrl_openpi_policy.yaml`

```yaml
model:
  component_config_paths:
    backbone:        ./components/backbone.yaml
    noise_processor: ./components/noise_processor.yaml
    noise_actor:     ./components/noise_actor.yaml
    openpi_model:    ./components/openpi_model.yaml

policy:
  type: dsrl_openpi_policy
  params:
    checkpoint_path: /abs/path/to/dsrl_checkpoints/exp1/epoch_10
    obs_proprio_history: 50
```

`checkpoint_path` is a `params:` key (consumed by the policy's `__init__`), not the top-level `checkpoint_path` that `build_policy` reads. The DSRL policy explicitly loads `backbone.pt`, `noise_processor.pt`, `noise_actor.pt` from that directory; it does *not* attempt `openpi_model.pt`.

### `components/backbone.yaml`

`_type_: dsrl_img_encoder`. Params: `resize: true`.

### `components/noise_processor.yaml`

`_type_: dsrl_noise_actor_processor`. Params:

- `proprio_key: 'proprio'`
- `depth_data_keys: []`
- `img_data_keys: ['head', 'left', 'right']`
- `input_img_channel: 512` (matches the backbone output channel)
- `output_img_channel: 24`
- `output_depth_channel: 24`

### `components/noise_actor.yaml`

`_type_: dsrl_noise_latent_actor`. Params:

- `input_dim: 6960` — must match the total flattened output of `noise_processor` at the configured image resolution.
- `action_chunk_size: 50`
- `action_dim: 32` (noise dim, not robot action dim)
- `num_layers: 5`, `num_hidden_dim: 2048`, `dropout: 0.0`

### `components/openpi_model.yaml`

Same fields as `openpi_policy/components/openpi_batched.yaml` — see [openpi_policy README § Component YAML fields](../openpi_policy/README.md#component-yaml-fields).

## Checkpoint layout

Expected at `params.checkpoint_path`:

```
exp1/epoch_10/
├── backbone.pt
├── noise_processor.pt
└── noise_actor.pt
```

OpenPI weights live separately under the path in `components/openpi_model.yaml`'s `params.ckpt_dir`.

## Related docs

- [docs/api.md § Policy protocol](../../../../docs/api.md#policy-protocol)
- [docs/configuration_cookbook.md](../../../../docs/configuration_cookbook.md)
- [openpi_policy README](../openpi_policy/README.md) for the simpler reference policy.
- [policy_constructor MENTAL_MODEL.md](https://github.com/KyunHwan/policy_constructor/blob/00663cc10c91d7614c1a0ea3d68629c38767b167/docs/MENTAL_MODEL.md) — how each component's YAML composes into a `GraphModel`.
