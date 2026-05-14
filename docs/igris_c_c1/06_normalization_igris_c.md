# 06 — Normalization bridge for IGRIS_C

**What this covers.** The IGRIS_C normalization bridge — where the stats come from, the normalize/denormalize math, which observation keys it touches, and how the normalized action gets back to the controller.
**Who this is for.** Anyone debugging out-of-distribution observations, retraining IGRIS_C, or wondering why the robot's first command jumps.

The bridge is small (~46 lines) but load-bearing. A wrong norm stats file is one of the most common silent failures in this stack — the robot runs, but every joint snaps to its mean.

## Table of contents

- [The IGRIS_C bridge](#the-igris_c-bridge)
- [Stats source: `dataset_stats.pkl`](#stats-source-dataset_statspkl)
- [What gets normalized](#what-gets-normalized)
- [Normalization math](#normalization-math)
- [Denormalization math](#denormalization-math)
- [The proprio-length slicing trick](#the-proprio-length-slicing-trick)
- [Where the bridge is plugged in](#where-the-bridge-is-plugged-in)
- [Action denormalization → `publish_action`](#action-denormalization--publish_action)
- [Differences from IGRIS_B](#differences-from-igris_b)

## The IGRIS_C bridge

Source: [`env_actor/nom_stats_manager/robots/igris_c/data_normalization_manager.py`](../../env_actor/nom_stats_manager/robots/igris_c/data_normalization_manager.py).

Full file (46 lines):

```python
import numpy as np


class DataNormalizationBridge:
    """
    Normalization bridge for igris_c proprio (86-D) and action (17-D).

    Assumes the trainer wrote a single 86-length `observation.state` mean/std
    covering [body_q (31), hand_q (12), body_tau (31), hand_tau (12)] in that
    order, plus an `action` mean/std of length 17. If the pkl uses split keys,
    update normalize_state to concatenate them in the same proprio order before
    slicing.
    """

    def __init__(self, norm_stats):
        self.norm_stats = norm_stats

    def normalize_state(self, state: dict[str, np.ndarray]):
        state_mean = self.norm_stats['observation.state']['mean']
        state_std  = self.norm_stats['observation.state']['std']

        eps = 1e-8

        proprio_len = state['proprio'].shape[-1]
        state['proprio'] = (state['proprio'] - state_mean[:proprio_len]) / (state_std[:proprio_len] + eps)

        for key in state.keys():
            if key != 'proprio':
                state[key] = state[key] / 255.0

        return state

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        action_mean = self.norm_stats['action']['mean']
        action_std  = self.norm_stats['action']['std']

        eps = 1e-8

        return (action - action_mean) / (action_std + eps)

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        action_mean = self.norm_stats['action']['mean']
        action_std  = self.norm_stats['action']['std']

        return action * action_std + action_mean
```

## Stats source: `dataset_stats.pkl`

The pickle path is set in [`inference_runtime_params.json:16`](../../env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_params.json#L16):

```json
"norm_stats_file_path": "/home/robros/Projects/inference_engine/trainer/experiment_training/igris_c/dataset_stats.pkl"
```

**Current value on this branch — this is an absolute path on the author's machine.** You will need to change it. The expected schema:

```python
{
    'observation.state': {
        'mean': np.ndarray[float],   # length >= 86
        'std':  np.ndarray[float],   # length >= 86, same as mean
    },
    'action': {
        'mean': np.ndarray[float],   # length == 17
        'std':  np.ndarray[float],   # length == 17
    },
    # may contain additional keys (image stats, etc.) — bridge ignores them
}
```

`mean` and `std` for `observation.state` may be **longer than 86**; the bridge slices `[:proprio_len]` (see [the slicing trick](#the-proprio-length-slicing-trick) below). For `action`, exact length 17 is required (no slicing on the action side).

How the pickle is loaded ([`runtime_settings_configs/robots/igris_c/inference_runtime_params.py:127-135`](../../env_actor/runtime_settings_configs/robots/igris_c/inference_runtime_params.py#L127)):

```python
def read_stats_file(self):
    norm_stats = None
    norm_stats_file_path = Path(self._norm_stats_file_path)
    if norm_stats_file_path.is_file():
        with norm_stats_file_path.open('rb') as file:
            norm_stats = pickle.load(file)
    else:
        print(f"File not found at: {norm_stats_file_path}")
    return norm_stats
```

**Failure mode:** if the file does not exist, `read_stats_file()` returns `None`. The downstream `DataNormalizationBridge(None)` constructor accepts it without complaint, but the first `normalize_state` call will raise `TypeError: 'NoneType' object is not subscriptable`. The print-only "File not found at: ..." message is easy to miss in a busy log — explicitly grep for it before assuming things are fine. `TODO:` make `read_stats_file()` raise so the failure is loud. Tracked in [09_troubleshooting_igris_c.md § Missing norm stats](09_troubleshooting_igris_c.md#missing-norm-stats-pkl).

## What gets normalized

`normalize_state(state)` operates **in place** on a dict produced by `controller_bridge.read_state()`:

| Key | Treatment | Why |
|---|---|---|
| `proprio` | z-score using `state_mean[:proprio_len]` / `state_std[:proprio_len]` | Continuous-valued physical quantities; policy needs zero-mean unit-variance inputs |
| `head`, `left`, `right` (and any other camera key) | Divide by `255.0` | Match training-time image scaling (`uint8 [0..255] → float [0..1]`) |

The branch on what to do per key is simple: `if key != 'proprio': state[key] = state[key] / 255.0`. There is **no** explicit listing of camera keys, so if `read_state()` ever produces a non-`proprio` key that **isn't** a camera, it will silently be divided by 255. In practice this can't happen — the bridge only ever produces those four keys.

`normalize_action(action)` and `denormalize_action(action)` operate on the **17-D action vector** (or a chunk thereof) as a single numpy array. They are pure functions; the input is not mutated.

## Normalization math

Per-element z-score with `eps` for numerical safety:

```
proprio_norm[i] = (proprio[i] - state_mean[i]) / (state_std[i] + eps)     for i in 0..85
image_norm      = image / 255.0                                            element-wise
action_norm[i]  = (action[i] - action_mean[i]) / (action_std[i] + eps)    for i in 0..16
```

`eps = 1e-8` ([line 22 for state, line 38 for action](../../env_actor/nom_stats_manager/robots/igris_c/data_normalization_manager.py#L22)) prevents division by zero in case any per-dim std is zero (e.g. a joint that never moved in the training dataset).

## Denormalization math

```
action[i] = action_norm[i] * action_std[i] + action_mean[i]               for i in 0..16
```

No `+ eps` here — denormalization is the exact inverse of `normalize_action` only when `std > 0`. If a per-dim std is zero, both normalization and denormalization will produce constant-zero actions for that dimension; the `+ eps` at normalization time still kicks in but is irrelevant to the inverse mapping. In practice IGRIS_C training data should have non-zero std for every action dim (waist yaw and both hands move during demonstrations).

## The proprio-length slicing trick

```python
proprio_len = state['proprio'].shape[-1]   # 86 for IGRIS_C
state['proprio'] = (state['proprio'] - state_mean[:proprio_len]) / (state_std[:proprio_len] + eps)
```

The bridge slices the stats vectors to the runtime proprio length. **Why?** Because the trainer might have written stats for a richer proprio vector (e.g. it had IMU data) and the inference engine consumes a subset. As long as the first 86 entries of the stats vector are `[body_q (31), hand_q (12), body_tau (31), hand_tau (12)]` in that order — the same order the bridge produces — the slice is correct.

If your stats file uses **split keys** (`'observation.state.body_q', 'observation.state.hand_q', ...`) instead of a single concatenated vector, the bridge will fail with `KeyError` on `'observation.state'`. The docstring at lines 5-12 documents the fix: pre-concatenate the four arrays in the bridge before slicing. `TODO:` future versions could detect both layouts.

## Where the bridge is plugged in

The factory at [`nom_stats_manager/data_normalization_interface.py:1-12`](../../env_actor/nom_stats_manager/data_normalization_interface.py#L1):

```python
class DataNormalizationInterface:
    def __init__(self, robot, data_stats):
        if robot == 'igris_b':
            from .robots.igris_b.data_normalization_manager import DataNormalizationBridge
        elif robot == 'igris_c':
            from .robots.igris_c.data_normalization_manager import DataNormalizationBridge

        self.data_normalizer = DataNormalizationBridge(data_stats)
```

The interface is constructed in four places — all four pass the dict returned by `runtime_params.read_stats_file()`:

| Caller | File | Line |
|---|---|---|
| `SequentialActor.__init__` | [`sequential/sequential_actor.py`](../../env_actor/auto/inference_algorithms/sequential/sequential_actor.py#L73) | 73 |
| `SequentialLocalActor.__init__` | [`sequential_local/sequential_local_actor.py`](../../env_actor/auto/inference_algorithms/sequential_local/sequential_local_actor.py#L85) | 85 (currently commented out — see [02 § sequential_local](02_changes_vs_main.md#env_actorautoinference_algorithmssequential_local__init__py-and-sequential_local_actorpy-added--2-files)) |
| `start_inference` (RTC) | [`rtc/actors/inference_loop.py`](../../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py#L94) | 94 |
| `start_inference` (RTC local) | [`rtc_local/actors/inference_loop.py`](../../env_actor/auto/inference_algorithms/rtc_local/actors/inference_loop.py#L90) | 90 |

The **policy** is then expected to call `data_normalization_interface.normalize_state(...)` and `denormalize_action(...)` inside its `predict()` / `guided_inference()` methods. The data manager hands raw observations to the policy; the policy normalizes, runs the network, denormalizes, and returns a real-units action chunk. The data manager and the bridge then store the chunk and execute actions from it.

The pattern is the same on both algorithms — see:
- Sequential: [`sequential_actor.py:127`](../../env_actor/auto/inference_algorithms/sequential/sequential_actor.py#L127) passes `self.data_normalization_interface` to `policy.predict(obs, ...)`.
- RTC: [`rtc/actors/inference_loop.py:120-124`](../../env_actor/auto/inference_algorithms/rtc/actors/inference_loop.py#L120) passes it to `policy.guided_inference(input_data, data_normalization_interface, ...)`.

`TODO:` the exact contract for how the policy uses the bridge lives in the policy code — see [`docs/api.md`](../api.md) for the policy protocol and `docs/walkthroughs/03_add_a_new_policy.md` for the worked example. This branch did not change those.

## Action denormalization → `publish_action`

The denormalized chunk that comes out of `guided_inference` (RTC) or `predict` (sequential) is in **real units** — radians for joints, normalized [0,1] for hand grasp scalars, radians for waist yaw. That is exactly what `publish_action(action_17, prev_joint_31)` expects ([`controller_bridge.py:386`](../../env_actor/robot_io_interface/robots/igris_c/controller_bridge.py#L386)).

So the round-trip is:

```
controller.read_state()                               # raw units (rad, normalized, etc.)
  ↓
bridge.normalize_state(state)                          # z-score (proprio) + /255 (cameras) ← inside policy
  ↓
policy.predict / guided_inference                      # neural network on normalized inputs
  ↓
bridge.denormalize_action(action_chunk)                # back to real units ← inside policy
  ↓
data_manager.buffer_action_chunk(...)                  # store
  ↓
data_manager.get_current_action(t)                     # pick the action for this step
  ↓
controller.publish_action(action_17, prev_joint_31)    # 31-D lift + slew limit + DDS write
```

The bridge does **not** know that some action dims are joints and others are scalar grasp values — it just z-scores them all uniformly. The training stats determine the mean and std per dim, which is why getting the right `dataset_stats.pkl` is critical.

## Differences from IGRIS_B

The IGRIS_B normalization bridge lives at [`env_actor/nom_stats_manager/robots/igris_b/data_normalization_manager.py`](../../env_actor/nom_stats_manager/robots/igris_b/data_normalization_manager.py) (unchanged on this branch). The IGRIS_C version is essentially the same algorithm with three differences:

| Aspect | IGRIS_B | IGRIS_C |
|---|---|---|
| Proprio dim | 24 | 86 |
| Action dim | 24 | 17 |
| Slice trick | Not necessary in the IGRIS_B bridge (proprio is fixed at 24-D, matches stats length) | Slices `[:proprio_len]` so stats files written by trainer with extra entries still work |

The actual math (`(x - mean) / (std + eps)`, images / 255.0) is unchanged.

---

← Back to index: [README.md](README.md) · Next → [07_factory_registration_igris_c.md](07_factory_registration_igris_c.md)
