# policy/templates

**Parent:** [policy](../README.md)

The `Policy` protocol — the contract every concrete policy must satisfy.

## Files

| File | What it defines |
|---|---|
| [`policy.py`](policy.py) | `Policy`, a `@runtime_checkable Protocol` with `__init__`, `predict`, `guided_inference`, `warmup`, `freeze_all_model_params`. |

## Why a Protocol, not a base class

`Policy` is defined with `typing.Protocol` rather than as an abstract base class. The implications:

- **No inheritance required.** A concrete policy is *any* class that has the right methods with the right signatures. You don't write `class MyPolicy(Policy):` — you just write `class MyPolicy:` and implement the methods. Python's type checker (and `isinstance(obj, Policy)` at runtime, thanks to `@runtime_checkable`) confirms the structural match.
- **No registry-time enforcement.** The registry passes `expected_base=Policy`, but because `Policy` is a Protocol the `issubclass` check is structural. Practically, a class with the right methods will pass; a class missing one will fail at first call, not at registration.
- **Easier composition.** A policy can wrap an `nn.Module` it owns; you don't have to make the *policy* itself an `nn.Module` subclass. See [`OpenPiPolicy`](../policies/openpi_policy/openpi_policy.py) for an example — it's a plain class that delegates `eval`, `to`, etc. to the wrapped module.

If structural subtyping is unfamiliar, the short version: "Static type checkers say the class is a `Policy` if it has all the right methods." See [PEP 544](https://peps.python.org/pep-0544/).

## What every policy must implement

See [parent README § The Policy protocol](../README.md#the-policy-protocol) and [docs/api.md § Policy protocol](../../../docs/api.md#policy-protocol) for the full method signatures, shapes, and call sites.

## Related docs

- [docs/api.md § Policy protocol](../../../docs/api.md#policy-protocol)
- [docs/concepts.md § Structural subtyping (Protocol)](../../../docs/glossary.md#structural-subtyping-protocol)
- [docs/walkthroughs/03_add_a_new_policy.md](../../../docs/walkthroughs/03_add_a_new_policy.md)
