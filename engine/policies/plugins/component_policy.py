"""Register the default component policy builder."""
from __future__ import annotations

from engine.policies.builders.component_policy import ComponentPolicyBuilder
from engine.policies.registry import POLICY_REGISTRY

POLICY_REGISTRY.add("component_policy", ComponentPolicyBuilder())
