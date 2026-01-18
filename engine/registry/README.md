# Registry

Registries map string keys to builders or component factories so that YAML configs can refer to components by name without hardcoding imports.

Files
- `core.py`: `Registry` with `add/get/register` and optional base-class enforcement.
- `plugins.py`: `load_plugins` imports plugin modules once to populate registries.

How it is used
- Policy builders register themselves in `engine.policies.registry.POLICY_REGISTRY`.
- The inference YAML lists plugin modules under `plugins`.
- `load_plugins` imports those modules, which execute registrations on import.

Example
```python
from engine.policies.registry import POLICY_REGISTRY

POLICY_REGISTRY.add("my_policy", MyPolicyBuilder())
```

Notes
- `Registry.get` raises with available keys to make config errors actionable.
- `Registry.register` provides a decorator when you want to register classes by name.
