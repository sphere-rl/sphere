from typing import ClassVar, Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from gymnasium import spaces
from sbx import PPO as sbxPPO
from sbx.common.type_aliases import TrainState
from sbx.common.policies import Flatten
from sbx.ppo.policies import PPOPolicy as SBXPPOPolicy
from sbx.ppo.policies import Actor as SBXActor
from sbx.ppo.policies import Critic as SBXCritic
import tensorflow_probability.substrates.jax as tfp


class Actor(SBXActor):
    """Actor with optional LayerNorm after each hidden layer."""

    use_layer_norm: bool = False
    use_l2_norm: bool = False
    l2_gain_init: float = 1.0
    l2_eps: float = 1e-6
    tfd = tfp.distributions

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Any:
        x = Flatten()(x)
        l2_scale = self.param("l2_scale", nn.initializers.constant(self.l2_gain_init), ())
        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = self.activation_fn(x)
            if self.use_l2_norm:
                norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
                x = x / (norm + self.l2_eps) * l2_scale
        if self.ortho_init:
            orthogonal_init = nn.initializers.orthogonal(scale=0.01)
            bias_init = nn.initializers.zeros
            action_logits = nn.Dense(self.action_dim, kernel_init=orthogonal_init, bias_init=bias_init)(x)
        else:
            action_logits = nn.Dense(self.action_dim)(x)
        log_std = self.param("log_std", nn.initializers.constant(self.log_std_init), (self.action_dim,))
        return self.tfd.MultivariateNormalDiag(loc=action_logits, scale_diag=jnp.exp(log_std))


class Critic(SBXCritic):
    """Value function with optional LayerNorm."""

    use_layer_norm: bool = False
    use_l2_norm: bool = False
    l2_gain_init: float = 1.0
    l2_eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = Flatten()(x)
        l2_scale = self.param("l2_scale", nn.initializers.constant(self.l2_gain_init), ())
        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = self.activation_fn(x)
            if self.use_l2_norm:
                norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
                x = x / (norm + self.l2_eps) * l2_scale
        value = nn.Dense(1)(x)
        return value


class PPOPolicy(SBXPPOPolicy):
    """PPO policy using LayerNorm-enabled Actor/Critic by default."""

    def __init__(
        self,
        *args,
        use_layer_norm: bool = False,
        use_l2_norm: bool = False,
        use_actor_layer_norm: bool | None = None,
        use_critic_layer_norm: bool | None = None,
        use_actor_l2_norm: bool | None = None,
        use_critic_l2_norm: bool | None = None,
        **kwargs,
    ):
        # Per-branch LayerNorm configuration (actor / critic) with backward compatibility.
        if use_actor_layer_norm is None and use_critic_layer_norm is None:
            self.use_actor_layer_norm = bool(use_layer_norm)
            self.use_critic_layer_norm = bool(use_layer_norm)
        else:
            self.use_actor_layer_norm = bool(
                use_actor_layer_norm if use_actor_layer_norm is not None else use_layer_norm
            )
            self.use_critic_layer_norm = bool(
                use_critic_layer_norm if use_critic_layer_norm is not None else use_layer_norm
            )

        # Per-branch L2-normalization configuration (actor / critic) with backward compatibility.
        if use_actor_l2_norm is None and use_critic_l2_norm is None:
            self.use_actor_l2_norm = bool(use_l2_norm)
            self.use_critic_l2_norm = bool(use_l2_norm)
        else:
            self.use_actor_l2_norm = bool(
                use_actor_l2_norm if use_actor_l2_norm is not None else use_l2_norm
            )
            self.use_critic_l2_norm = bool(
                use_critic_l2_norm if use_critic_l2_norm is not None else use_l2_norm
            )

        # Keep global flags for diagnostics / legacy code paths.
        self.use_layer_norm = bool(use_layer_norm)
        self.use_l2_norm = bool(use_l2_norm)

        # L2 normalization hyperparameters kept as internal constants.
        self.l2_gain_init = 1.0
        self.l2_eps = 1e-6
        kwargs.setdefault("actor_class", Actor)
        kwargs.setdefault("critic_class", Critic)
        super().__init__(*args, **kwargs)

    def build(self, key, lr_schedule, max_grad_norm):  # type: ignore[override]
        key, actor_key, vf_key = jax.random.split(key, 3)
        key, self.key = jax.random.split(key, 2)
        self.reset_noise()
        obs = jnp.array([self.observation_space.sample()])

        if isinstance(self.action_space, spaces.Box):
            actor_kwargs: dict[str, Any] = {"action_dim": int(np.prod(self.action_space.shape))}
        else:
            raise NotImplementedError("PPOPolicy currently supports continuous Box action spaces only")

        self.actor = self.actor_class(
            net_arch=self.net_arch_pi,
            log_std_init=self.log_std_init,
            activation_fn=self.activation_fn,
            ortho_init=self.ortho_init,
            use_layer_norm=self.use_actor_layer_norm,
            use_l2_norm=self.use_actor_l2_norm,
            l2_gain_init=self.l2_gain_init,
            l2_eps=self.l2_eps,
            **actor_kwargs,
        )
        self.actor.reset_noise = self.reset_noise

        optimizer_class = optax.inject_hyperparams(self.optimizer_class)(
            learning_rate=lr_schedule(1),
            **self.optimizer_kwargs,
        )
        self.actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor.init(actor_key, obs),
            tx=optax.chain(optax.clip_by_global_norm(max_grad_norm), optimizer_class),
        )

        self.vf = self.critic_class(
            net_arch=self.net_arch_vf,
            activation_fn=self.activation_fn,
            use_layer_norm=self.use_critic_layer_norm,
            use_l2_norm=self.use_critic_l2_norm,
            l2_gain_init=self.l2_gain_init,
            l2_eps=self.l2_eps,
        )
        self.vf_state = TrainState.create(
            apply_fn=self.vf.apply,
            params=self.vf.init(vf_key, obs),
            tx=optax.chain(optax.clip_by_global_norm(max_grad_norm), optimizer_class),
        )

        self.actor.apply = jax.jit(self.actor.apply)  # type: ignore[method-assign]
        self.vf.apply = jax.jit(self.vf.apply)  # type: ignore[method-assign]
        return key


class PPO(sbxPPO):
    """Thin wrapper over sbx.PPO with policy alias pass-through."""

    policy_aliases: ClassVar[dict[str, type[PPOPolicy]]] = {  # type: ignore[assignment]
        "MlpPolicy": PPOPolicy,
        "MultiInputPolicy": PPOPolicy,
    }
