"""Policy builder registry."""
from __future__ import annotations

from engine.registry.core import Registry

from engine.policies.interfaces import PolicyBuilder

POLICY_REGISTRY: Registry[PolicyBuilder] = Registry("policy_builder")
