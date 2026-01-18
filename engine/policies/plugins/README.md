# Policy Plugins

Plugins are importable modules that register policy builders into `POLICY_REGISTRY`.

Included plugin
- `component_policy.py` registers the default `component_policy` builder.

Usage
- List plugin modules in inference YAML `plugins`.
- `engine.registry.plugins.load_plugins` imports them once to populate registries.

Example entry
- `engine.policies.plugins.component_policy`
