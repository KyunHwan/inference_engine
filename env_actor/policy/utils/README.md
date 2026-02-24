# policy/utils

Utilities for policy loading and weight management.

## Modules

### `loader.py`

The main API for instantiating policies from YAML configuration.

**`build_policy(policy_yaml_path, *, map_location="cpu") -> Policy`**

Loads a complete policy from a YAML config file. This is the function called by inference actors to create their policy instance.

Flow:
1. `load_policy_config(path)` — loads the YAML, resolving any `defaults` composition (via the trainer's config loader)
2. Resolves component config paths relative to the policy YAML's directory
3. Builds `nn.Module` components using `PolicyConstructorModelFactory` from the trainer submodule
4. Optionally loads checkpoint state dicts from `config["checkpoint_path"]/{component_name}.pt`
5. Looks up the policy class in `POLICY_REGISTRY` by `config["policy"]["type"]`; auto-imports `env_actor.policy.policies.{type}.{type}` if not yet registered
6. Instantiates and returns the policy with `policy_cls(components=components, **params)`

### `weight_transfer.py`

**`load_state_dict_cpu_into_module(module, state_dict)`**

Transfers a CPU-loaded state dict into a module that may already be on GPU. Handles device and dtype matching automatically, including unwrapping DDP-wrapped modules.
