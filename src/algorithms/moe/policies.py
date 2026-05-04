from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import flax.linen as nn
import jax.numpy as jnp
from sbx.common.policies import Flatten


class ContinuousActorPreTanh(nn.Module):
    """Actor body that exposes pre-tanh logits for mixture-before-nonlinearity aggregation."""

    net_arch: Sequence[int]
    action_dim: int
    use_layer_norm: bool = False
    use_l2_norm: bool = False
    l2_gain_init: float = 1.0
    l2_eps: float = 1e-6
    dropout_rate: Optional[float] = None
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray, return_features: bool = False):
        x = Flatten()(x)
        l2_scale = self.param("l2_scale", nn.initializers.constant(self.l2_gain_init), ())
        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)
            if self.dropout_rate is not None and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = self.activation_fn(x)
            if self.use_l2_norm:
                norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
                x = x / (norm + self.l2_eps) * l2_scale
        features = x
        action_logits = nn.Dense(self.action_dim)(features)
        return (action_logits, features) if return_features else action_logits


class GatingActor(nn.Module):
    """Gating network for MoE actor/critic (depends on observation only)."""

    net_arch: Sequence[int]
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    n_experts: int = 1

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = Flatten()(obs)
        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)
            if self.dropout_rate is not None and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = self.activation_fn(x)
        logits = nn.Dense(self.n_experts)(x)
        return logits, x

