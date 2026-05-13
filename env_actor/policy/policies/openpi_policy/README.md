# openpi_policy

**Parent:** [policies](../README.md)

A single-component wrapper for OpenPI/Pi0.5. The simplest concrete policy in the repo.

## Table of contents

- [What is OpenPI?](#what-is-openpi)
- [Files](#files)
- [How it plugs in](#how-it-plugs-in)
- [Component YAML fields](#component-yaml-fields)
- [Related docs](#related-docs)

## What is OpenPI?

[OpenPI](https://github.com/Physical-Intelligence/openpi) is a pretrained vision-language-action transformer ("Pi0.5"). It takes:

- Images (head/left/right at 224×224 or similar — see the trainer config),
- A natural-language prompt,
- A proprioceptive state vector,

and emits an action chunk via flow matching.

In this repo, OpenPI is **built** by `policy_constructor`'s `openpi_batched` block (registered in [`policy_constructor/.../register.py`](https://github.com/KyunHwan/policy_constructor/blob/00663cc10c91d7614c1a0ea3d68629c38767b167/model_constructor/blocks/register.py)) and **wrapped** here in `OpenPiPolicy`.

## Files

| File | Purpose |
|---|---|
| [`openpi_policy.py`](openpi_policy.py) | `OpenPiPolicy` class. Implements `Policy`. Receives one component (`openpi_model`), unwraps the actual `OpenPiBatchedWrapper` out of its `GraphModel`, delegates `predict` and `guided_inference`. |
| [`openpi_policy.yaml`](openpi_policy.yaml) | Top-level YAML. `policy.type: openpi_policy`. Single component. |
| [`components/openpi_batched.yaml`](components/openpi_batched.yaml) | Component YAML with `_type_: openpi_batched`. Carries `train_config_name`, `ckpt_dir`, `default_prompt`, `action_dim`, `action_horizon`, `num_inference_steps`, `gradient_checkpointing`. |

## How it plugs in

The flow inside `OpenPiPolicy.predict`:

1. Take `obs["proprio"][0]` (newest proprio) and `obs[cam][-1:]` (newest frame per camera).
2. Batch them: `proprio` → `(1, state_dim)`, `cam` → `(1, 3, H, W)`.
3. Optionally attach `obs["prompt"]` (otherwise the wrapper falls back to `default_prompt`).
4. Call `self._wrapper.predict(batched_obs, noise=None)` → `(1, action_horizon, action_dim)`.
5. Strip the batch dim → `(action_horizon, action_dim)`.

`guided_inference` does the same forward pass, then blends with `input_data["prev_action"]` using `compute_guided_prefix_weights(est_delay, min_executed, chunk, "exp")`.

`warmup` delegates to `self._wrapper.warmup(batch_size=1)` which is where `torch.compile` warms up.

`_resolve_wrapper` is non-trivial: `policy_constructor` builds each component as a `GraphModel` whose `graph_modules: ModuleDict` contains the actual `OpenPiBatchedWrapper`. The resolver extracts it by `graph_model.graph_modules["openpi_model"]` (falling back to the only entry if there is just one).

## Component YAML fields

[components/openpi_batched.yaml](components/openpi_batched.yaml) keys:

| Key | Effect |
|---|---|
| `train_config_name` | Name of the OpenPI training config (e.g. `"pi05_igris"`). Identifies the architecture variant. |
| `ckpt_dir` | Absolute path to the directory containing the OpenPI safetensors. `OpenPiBatchedWrapper.__init__` reads from here. |
| `default_prompt` | Used when `obs` has no `"prompt"` key. |
| `action_dim` | Action vector length. Must match `runtime_params.action_dim`. |
| `action_horizon` | Number of action steps per chunk. Must match `runtime_params.action_chunk_size`. |
| `num_inference_steps` | Number of flow-matching ODE steps at inference. Lower = faster, less accurate. |
| `gradient_checkpointing` | Train-time only; keep `false` for inference. |

The trainer-side wrapper is documented in [trainer/README.md](https://github.com/KyunHwan/trainer/blob/3ca051a256c9068f77b556df98f538d9a6185ccf/README.md) and [policy_constructor README](https://github.com/KyunHwan/policy_constructor/blob/00663cc10c91d7614c1a0ea3d68629c38767b167/README.md).

## Related docs

- [docs/api.md § Policy protocol](../../../../docs/api.md#policy-protocol)
- [docs/walkthroughs/03_add_a_new_policy.md](../../../../docs/walkthroughs/03_add_a_new_policy.md) — uses this policy as the template to copy.
- [docs/configuration_cookbook.md § Change the OpenPI checkpoint path](../../../../docs/configuration_cookbook.md#change-the-openpi-checkpoint-path)
- [docs/concepts.md § Flow matching](../../../../docs/concepts.md#flow-matching-in-one-paragraph)
