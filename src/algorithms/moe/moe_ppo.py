from __future__ import annotations

from functools import partial
from typing import Any, ClassVar

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_probability.substrates.jax as tfp
from gymnasium import spaces
from sbx.common.type_aliases import TrainState
from stable_baselines3.common.utils import explained_variance

from src.algorithms.base.ppo import PPO, Actor as DefaultActor, Critic as DefaultCritic, PPOPolicy
from src.algorithms.moe.policies import ContinuousActorPreTanh, GatingActor
from src.algorithms.moe.sphere import sphere_feature_loss, sphere_output_loss

tfd = tfp.distributions


class TopKMoEActor(nn.Module):
    """Top-K softmax-gated MoE actor for PPO (continuous actions)."""

    action_dim: int
    net_arch: tuple[int, ...]
    n_experts: int = 10
    top_k: int = 1
    temperature: float = 1.0
    log_std_init: float = 0.0
    activation_fn: Any = nn.relu
    use_layer_norm: bool = False
    use_l2_norm: bool = False
    l2_gain_init: float = 1.0
    l2_eps: float = 1e-6
    dropout_rate: float | None = None
    ortho_init: bool = False

    @nn.compact
    def __call__(self, obs: jnp.ndarray, return_aux: bool = False, return_features: bool = False):
        if self.top_k > self.n_experts:
            raise ValueError("top_k must be ≤ n_experts")

        experts = nn.vmap(
            ContinuousActorPreTanh,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=1,
            axis_size=self.n_experts,
        )
        res = experts(
            net_arch=self.net_arch,
            action_dim=self.action_dim,
            use_layer_norm=self.use_layer_norm,
            use_l2_norm=self.use_l2_norm,
            l2_gain_init=self.l2_gain_init,
            l2_eps=self.l2_eps,
            dropout_rate=self.dropout_rate,
            activation_fn=self.activation_fn,
        )(obs, return_features)
        if return_features:
            pre_actions, features = res
        else:
            pre_actions = res
            features = None

        logits, embedding = GatingActor(
            net_arch=self.net_arch,
            use_layer_norm=self.use_layer_norm,
            dropout_rate=self.dropout_rate,
            activation_fn=self.activation_fn,
            n_experts=self.n_experts,
        )(obs)

        topk_val, topk_idx = jax.lax.top_k(logits, self.top_k)
        scores = topk_val / self.temperature
        topk_probs = jax.nn.softmax(scores, axis=-1)

        probs = jnp.zeros_like(logits)
        b_idx = jnp.arange(logits.shape[0])[:, None]
        probs = probs.at[b_idx, topk_idx].set(topk_probs)

        mean = (pre_actions * probs[..., None]).sum(axis=1)
        log_std = self.param("log_std", nn.initializers.constant(self.log_std_init), (self.action_dim,))
        dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))

        if not return_aux:
            return dist

        aux = dict(probs=probs, logits=logits / self.temperature, topk_indices=topk_idx, gating_embedding=embedding)
        if return_features and features is not None:
            aux["expert_features"] = features
        return dist, aux


class ValueExpert(nn.Module):
    """Simple value expert (MLP) for MoE critic."""

    net_arch: tuple[int, ...]
    use_layer_norm: bool = False
    use_l2_norm: bool = False
    l2_gain_init: float = 1.0
    l2_eps: float = 1e-6
    dropout_rate: float | None = None
    activation_fn: Any = nn.relu

    @nn.compact
    def __call__(self, obs: jnp.ndarray, return_features: bool = False):
        x = obs.reshape(obs.shape[0], -1)
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
        value = nn.Dense(1)(features)
        return (value, features) if return_features else value


class TopKValueMoECritic(nn.Module):
    """Top-K softmax-gated MoE value network (observation-only)."""

    net_arch: tuple[int, ...]
    n_experts: int = 10
    top_k: int = 1
    temperature: float = 1.0
    use_layer_norm: bool = False
    use_l2_norm: bool = False
    l2_gain_init: float = 1.0
    l2_eps: float = 1e-6
    dropout_rate: float | None = None
    activation_fn: Any = nn.relu

    @nn.compact
    def __call__(self, obs: jnp.ndarray, return_aux: bool = False, return_features: bool = False):
        if self.top_k > self.n_experts:
            raise ValueError("top_k must be ≤ n_experts")

        experts = nn.vmap(
            ValueExpert,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=1,
            axis_size=self.n_experts,
        )
        res = experts(
            net_arch=self.net_arch,
            use_layer_norm=self.use_layer_norm,
            use_l2_norm=self.use_l2_norm,
            l2_gain_init=self.l2_gain_init,
            l2_eps=self.l2_eps,
            dropout_rate=self.dropout_rate,
            activation_fn=self.activation_fn,
        )(obs, return_features)
        if return_features:
            values, features = res
        else:
            values = res
            features = None

        logits, embedding = GatingActor(
            net_arch=self.net_arch,
            use_layer_norm=self.use_layer_norm,
            dropout_rate=self.dropout_rate,
            activation_fn=self.activation_fn,
            n_experts=self.n_experts,
        )(obs)

        topk_val, topk_idx = jax.lax.top_k(logits, self.top_k)
        topk_probs = jax.nn.softmax(topk_val / self.temperature, axis=-1)

        probs = jnp.zeros_like(logits)
        b_idx = jnp.arange(logits.shape[0])[:, None]
        probs = probs.at[b_idx, topk_idx].set(topk_probs)

        value = (values * probs[..., None]).sum(axis=1)

        if not return_aux:
            return value

        aux = dict(probs=probs, logits=logits / self.temperature, topk_indices=topk_idx, gating_embedding=embedding)
        if return_features and features is not None:
            aux["expert_features"] = features
        return value, aux


class TopKMoEPPOPolicy(PPOPolicy):
    """PPO policy with optional Top-K MoE actor/critic."""

    def __init__(
        self,
        *args,
        n_experts: int = 10,
        top_k: int = 1,
        temperature: float = 1.0,
        use_layer_norm: bool = False,
        dropout_rate: float | None = None,
        apply_to: dict | None = None,
        use_l2_norm: bool = False,
        **kwargs,
    ):
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.dropout_rate = dropout_rate
        self.apply_actor_moe = bool(apply_to.get("actor", True) if apply_to is not None else True)
        self.apply_critic_moe = bool(apply_to.get("critic", False) if apply_to is not None else False)
        super().__init__(*args, use_layer_norm=use_layer_norm, use_l2_norm=use_l2_norm, **kwargs)

    def build(self, key: jax.Array, lr_schedule, max_grad_norm: float) -> jax.Array:  # type: ignore[override]
        key, actor_key, vf_key = jax.random.split(key, 3)
        key, self.key = jax.random.split(key, 2)
        self.reset_noise()

        obs = jnp.array([self.observation_space.sample()])
        if not isinstance(self.action_space, spaces.Box):
            raise NotImplementedError("TopKMoEActor currently supports continuous Box action spaces only")

        actor_kwargs: dict[str, Any] = {"action_dim": int(np.prod(self.action_space.shape))}
        if self.apply_actor_moe:
            actor_cls = TopKMoEActor
            actor_kwargs.update(
                dict(
                    n_experts=self.n_experts,
                    top_k=self.top_k,
                    temperature=self.temperature,
                    use_layer_norm=self.use_actor_layer_norm,
                    use_l2_norm=self.use_actor_l2_norm,
                    l2_gain_init=self.l2_gain_init,
                    l2_eps=self.l2_eps,
                    dropout_rate=self.dropout_rate,
                )
            )
        else:
            actor_cls = DefaultActor
            actor_kwargs.update(
                dict(
                    use_layer_norm=self.use_actor_layer_norm,
                    use_l2_norm=self.use_actor_l2_norm,
                    l2_gain_init=self.l2_gain_init,
                    l2_eps=self.l2_eps,
                )
            )

        self.actor = actor_cls(
            net_arch=tuple(self.net_arch_pi),
            log_std_init=self.log_std_init,
            activation_fn=self.activation_fn,
            ortho_init=self.ortho_init,
            **actor_kwargs,  # type: ignore[arg-type]
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

        if self.apply_critic_moe:
            critic_cls = TopKValueMoECritic
            critic_kwargs = dict(
                net_arch=tuple(self.net_arch_vf),
                n_experts=self.n_experts,
                top_k=self.top_k,
                temperature=self.temperature,
                use_layer_norm=self.use_critic_layer_norm,
                use_l2_norm=self.use_critic_l2_norm,
                l2_gain_init=self.l2_gain_init,
                l2_eps=self.l2_eps,
                dropout_rate=self.dropout_rate,
                activation_fn=self.activation_fn,
            )
        else:
            critic_cls = DefaultCritic
            critic_kwargs = dict(
                net_arch=self.net_arch_vf,
                activation_fn=self.activation_fn,
                use_layer_norm=self.use_critic_layer_norm,
                use_l2_norm=self.use_critic_l2_norm,
                l2_gain_init=self.l2_gain_init,
                l2_eps=self.l2_eps,
            )
        self.vf = critic_cls(**critic_kwargs)
        self.vf_state = TrainState.create(
            apply_fn=self.vf.apply,
            params=self.vf.init(vf_key, obs),
            tx=optax.chain(optax.clip_by_global_norm(max_grad_norm), optimizer_class),
        )

        self.actor.apply = jax.jit(self.actor.apply)  # type: ignore[method-assign]
        self.vf.apply = jax.jit(self.vf.apply)  # type: ignore[method-assign]
        return key


class SphereTopKMoEPPO(PPO):
    """PPO with Top-K MoE actor/critic + SPHERE regularization on expert features."""

    policy_aliases: ClassVar[dict[str, type[PPOPolicy]]] = {  # type: ignore[assignment]
        "TopKMoEPPOPolicy": TopKMoEPPOPolicy,
    }

    def __init__(
        self,
        *args,
        sphere_target_ratio: float = 0.05,
        sphere_gating_ratio: float = 0.0,
        sphere_use_pcgrad: bool = False,
        sphere_scale_mode: str = "loss",
        **kwargs,
    ):
        self.sphere_target_ratio = float(sphere_target_ratio)
        self.sphere_gating_ratio = float(sphere_gating_ratio)
        self.sphere_use_pcgrad = bool(sphere_use_pcgrad)
        self.sphere_scale_mode = str(sphere_scale_mode)
        super().__init__(*args, **kwargs)

    @staticmethod
    @partial(
        jax.jit,
        static_argnames=[
            "normalize_advantage",
            "sphere_target_ratio",
            "sphere_gating_ratio",
            "use_pcgrad",
            "sphere_scale_mode",
        ],
    )
    def _one_update_with_sphere(
        actor_state,
        vf_state,
        observations: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        old_log_prob: np.ndarray,
        clip_range: float,
        ent_coef: float,
        vf_coef: float,
        normalize_advantage: bool = True,
        sphere_target_ratio: float = 0.05,
        sphere_gating_ratio: float = 0.0,
        use_pcgrad: bool = False,
        sphere_scale_mode: str = "loss",
    ):
        if normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        def tree_vdot(a, b):
            leaves = jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x, y: jnp.vdot(x, y), a, b))
            acc = leaves[0]
            for v in leaves[1:]:
                acc = acc + v
            return acc

        def tree_l2_norm(tree) -> jnp.ndarray:
            norm_sq = tree_vdot(tree, tree).real
            return jnp.sqrt(norm_sq + jnp.asarray(1e-8, dtype=jnp.result_type(norm_sq)))

        def actor_base_loss(params):
            dist = actor_state.apply_fn(params, observations, return_aux=False, return_features=False)
            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()
            ratio = jnp.exp(log_prob - old_log_prob)
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * jnp.clip(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -jnp.minimum(policy_loss_1, policy_loss_2).mean()
            entropy_loss = -jnp.mean(entropy)
            return policy_loss + ent_coef * entropy_loss

        def actor_base_metrics(params):
            dist = actor_state.apply_fn(params, observations, return_aux=False, return_features=False)
            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()
            ratio = jnp.exp(log_prob - old_log_prob)
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * jnp.clip(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -jnp.minimum(policy_loss_1, policy_loss_2).mean()
            entropy_loss = -jnp.mean(entropy)
            return policy_loss + ent_coef * entropy_loss, ratio, policy_loss, entropy_loss

        def actor_sphere_feature_loss(params) -> jnp.ndarray:
            if sphere_target_ratio <= 0.0:
                return jnp.array(0.0)
            res = actor_state.apply_fn(params, observations, return_aux=True, return_features=True)
            if not isinstance(res, tuple):
                return jnp.array(0.0)
            _, aux = res
            feats = aux.get("expert_features", None)
            probs = aux.get("probs", None)
            if feats is None or probs is None:
                return jnp.array(0.0)
            weighted = probs[..., None] * feats
            phi = weighted.reshape((weighted.shape[0], weighted.shape[1] * weighted.shape[2]))
            return sphere_feature_loss(phi)

        def actor_sphere_gating_loss(params) -> jnp.ndarray:
            if sphere_gating_ratio <= 0.0:
                return jnp.array(0.0)
            res = actor_state.apply_fn(params, observations, return_aux=True, return_features=False)
            if not isinstance(res, tuple):
                return jnp.array(0.0)
            _, aux = res
            probs = aux.get("probs", None)
            if probs is None:
                return jnp.array(0.0)
            return sphere_output_loss(probs[None, ...])

        def actor_loss(params):
            sphere_loss_actor = jnp.array(0.0)
            sphere_loss_actor_gate = jnp.array(0.0)
            res = actor_state.apply_fn(
                params,
                observations,
                return_aux=(sphere_target_ratio > 0.0) or (sphere_gating_ratio > 0.0),
                return_features=sphere_target_ratio > 0.0,
            )
            if ((sphere_target_ratio > 0.0) or (sphere_gating_ratio > 0.0)) and isinstance(res, tuple):
                dist, aux = res
            else:
                dist, aux = res, None

            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()
            ratio = jnp.exp(log_prob - old_log_prob)
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * jnp.clip(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -jnp.minimum(policy_loss_1, policy_loss_2).mean()
            entropy_loss = -jnp.mean(entropy)

            base_policy_loss = policy_loss + ent_coef * entropy_loss
            total_policy_loss = base_policy_loss

            feats = aux.get("expert_features", None) if aux is not None else None
            probs = aux.get("probs", None) if aux is not None else None

            if sphere_target_ratio > 0.0 and feats is not None and probs is not None:
                weighted = probs[..., None] * feats
                phi = weighted.reshape((weighted.shape[0], weighted.shape[1] * weighted.shape[2]))
                sphere_loss_actor = sphere_feature_loss(phi)
                sphere_scale = jax.lax.stop_gradient(
                    sphere_target_ratio * jnp.abs(base_policy_loss) / (sphere_loss_actor + 1e-8)
                )
                total_policy_loss = total_policy_loss + sphere_scale * sphere_loss_actor

            if sphere_gating_ratio > 0.0 and probs is not None:
                sphere_loss_actor_gate = sphere_output_loss(probs[None, ...])
                sphere_scale_gate = jax.lax.stop_gradient(
                    sphere_gating_ratio * jnp.abs(base_policy_loss) / (sphere_loss_actor_gate + 1e-8)
                )
                total_policy_loss = total_policy_loss + sphere_scale_gate * sphere_loss_actor_gate

            return total_policy_loss, (ratio, policy_loss, entropy_loss, sphere_loss_actor, sphere_loss_actor_gate)

        if sphere_scale_mode == "grad":
            base_policy_loss, ratio, policy_loss, entropy_loss = actor_base_metrics(actor_state.params)
            pg_loss_value = base_policy_loss

            _, grads_base = jax.value_and_grad(actor_base_loss)(actor_state.params)
            sphere_loss_actor = actor_sphere_feature_loss(actor_state.params)
            sphere_loss_actor_gate = actor_sphere_gating_loss(actor_state.params)

            grads_reg = jax.tree_map(jnp.zeros_like, grads_base)
            base_norm = tree_l2_norm(grads_base)

            if sphere_target_ratio > 0.0:
                _, grads_feat = jax.value_and_grad(actor_sphere_feature_loss)(actor_state.params)
                feat_norm = tree_l2_norm(grads_feat)
                feat_scale = jax.lax.stop_gradient(sphere_target_ratio * base_norm / (feat_norm + 1e-8))
                grads_reg = jax.tree_map(lambda gr, gf: gr + feat_scale * gf, grads_reg, grads_feat)
            if sphere_gating_ratio > 0.0:
                _, grads_gate = jax.value_and_grad(actor_sphere_gating_loss)(actor_state.params)
                gate_norm = tree_l2_norm(grads_gate)
                gate_scale = jax.lax.stop_gradient(sphere_gating_ratio * base_norm / (gate_norm + 1e-8))
                grads_reg = jax.tree_map(lambda gr, gg: gr + gate_scale * gg, grads_reg, grads_gate)

            if use_pcgrad and (sphere_target_ratio > 0.0 or sphere_gating_ratio > 0.0):
                dot = tree_vdot(grads_base, grads_reg).real
                norm_sq = tree_vdot(grads_base, grads_base).real
                proj_scale = jnp.where(dot < 0.0, dot / (norm_sq + 1e-8), 0.0)
                grads_reg = jax.tree_map(lambda gr, gb: gr - proj_scale * gb, grads_reg, grads_base)

            grads_final = jax.tree_map(lambda gb, gr: gb + gr, grads_base, grads_reg)
            actor_state = actor_state.apply_gradients(grads=grads_final)
        else:
            if use_pcgrad and (sphere_target_ratio > 0.0 or sphere_gating_ratio > 0.0):
                (pg_loss_value, (ratio, policy_loss, entropy_loss, sphere_loss_actor, sphere_loss_actor_gate)), grads_total = jax.value_and_grad(  # type: ignore[assignment]
                    actor_loss, has_aux=True
                )(actor_state.params)
                _, grads_base = jax.value_and_grad(actor_base_loss)(actor_state.params)

                grads_reg = jax.tree_map(lambda gt, gb: gt - gb, grads_total, grads_base)
                dot = tree_vdot(grads_base, grads_reg).real
                norm_sq = tree_vdot(grads_base, grads_base).real
                proj_scale = jnp.where(dot < 0.0, dot / (norm_sq + 1e-8), 0.0)
                grads_reg_proj = jax.tree_map(lambda gr, gb: gr - proj_scale * gb, grads_reg, grads_base)
                grads_pc = jax.tree_map(lambda gb, grp: gb + grp, grads_base, grads_reg_proj)
                actor_state = actor_state.apply_gradients(grads=grads_pc)
            else:
                (pg_loss_value, (ratio, policy_loss, entropy_loss, sphere_loss_actor, sphere_loss_actor_gate)), grads = jax.value_and_grad(
                    actor_loss, has_aux=True
                )(actor_state.params)
                actor_state = actor_state.apply_gradients(grads=grads)

        def critic_loss(params):
            sphere_loss_critic = jnp.array(0.0)
            sphere_loss_critic_gate = jnp.array(0.0)
            try:
                res = vf_state.apply_fn(
                    params,
                    observations,
                    return_aux=(sphere_target_ratio > 0.0) or (sphere_gating_ratio > 0.0),
                    return_features=sphere_target_ratio > 0.0,
                )
                if ((sphere_target_ratio > 0.0) or (sphere_gating_ratio > 0.0)) and isinstance(res, tuple):
                    vf_values, aux = res
                else:
                    vf_values, aux = res, None
            except TypeError:
                vf_values, aux = vf_state.apply_fn(params, observations), None

            vf_values = vf_values.flatten()
            base_vf_loss = vf_coef * ((returns - vf_values) ** 2).mean()
            loss = base_vf_loss
            feats = aux.get("expert_features", None) if aux is not None else None
            probs = aux.get("probs", None) if aux is not None else None

            if sphere_target_ratio > 0.0 and feats is not None and probs is not None:
                weighted = probs[..., None] * feats
                phi = weighted.reshape((weighted.shape[0], weighted.shape[1] * weighted.shape[2]))
                sphere_loss_critic = sphere_feature_loss(phi)
                sphere_scale = jax.lax.stop_gradient(
                    sphere_target_ratio * jnp.abs(base_vf_loss) / (sphere_loss_critic + 1e-8)
                )
                loss = loss + sphere_scale * sphere_loss_critic

            if sphere_gating_ratio > 0.0 and probs is not None:
                sphere_loss_critic_gate = sphere_output_loss(probs[None, ...])
                sphere_scale_gate = jax.lax.stop_gradient(
                    sphere_gating_ratio * jnp.abs(base_vf_loss) / (sphere_loss_critic_gate + 1e-8)
                )
                loss = loss + sphere_scale_gate * sphere_loss_critic_gate

            return loss, (sphere_loss_critic, sphere_loss_critic_gate)

        def critic_base_loss(params):
            try:
                vf_values = vf_state.apply_fn(params, observations, return_aux=False, return_features=False)
            except TypeError:
                vf_values = vf_state.apply_fn(params, observations)
            vf_values = vf_values.flatten()
            return vf_coef * ((returns - vf_values) ** 2).mean()

        def critic_sphere_feature_loss(params) -> jnp.ndarray:
            if sphere_target_ratio <= 0.0:
                return jnp.array(0.0)
            try:
                res = vf_state.apply_fn(params, observations, return_aux=True, return_features=True)
                if not isinstance(res, tuple):
                    return jnp.array(0.0)
                _, aux = res
            except TypeError:
                return jnp.array(0.0)
            feats = aux.get("expert_features", None) if aux is not None else None
            probs = aux.get("probs", None) if aux is not None else None
            if feats is None or probs is None:
                return jnp.array(0.0)
            weighted = probs[..., None] * feats
            phi = weighted.reshape((weighted.shape[0], weighted.shape[1] * weighted.shape[2]))
            return sphere_feature_loss(phi)

        def critic_sphere_gating_loss(params) -> jnp.ndarray:
            if sphere_gating_ratio <= 0.0:
                return jnp.array(0.0)
            try:
                res = vf_state.apply_fn(params, observations, return_aux=True, return_features=False)
                if not isinstance(res, tuple):
                    return jnp.array(0.0)
                _, aux = res
            except TypeError:
                return jnp.array(0.0)
            probs = aux.get("probs", None) if aux is not None else None
            if probs is None:
                return jnp.array(0.0)
            return sphere_output_loss(probs[None, ...])

        if sphere_scale_mode == "grad":
            vf_loss_value, grads_base = jax.value_and_grad(critic_base_loss)(vf_state.params)
            sphere_loss_critic = critic_sphere_feature_loss(vf_state.params)
            sphere_loss_critic_gate = critic_sphere_gating_loss(vf_state.params)

            grads_reg = jax.tree_map(jnp.zeros_like, grads_base)
            base_norm = tree_l2_norm(grads_base)

            if sphere_target_ratio > 0.0:
                _, grads_feat = jax.value_and_grad(critic_sphere_feature_loss)(vf_state.params)
                feat_norm = tree_l2_norm(grads_feat)
                feat_scale = jax.lax.stop_gradient(sphere_target_ratio * base_norm / (feat_norm + 1e-8))
                grads_reg = jax.tree_map(lambda gr, gf: gr + feat_scale * gf, grads_reg, grads_feat)
            if sphere_gating_ratio > 0.0:
                _, grads_gate = jax.value_and_grad(critic_sphere_gating_loss)(vf_state.params)
                gate_norm = tree_l2_norm(grads_gate)
                gate_scale = jax.lax.stop_gradient(sphere_gating_ratio * base_norm / (gate_norm + 1e-8))
                grads_reg = jax.tree_map(lambda gr, gg: gr + gate_scale * gg, grads_reg, grads_gate)

            if use_pcgrad and (sphere_target_ratio > 0.0 or sphere_gating_ratio > 0.0):
                dot = tree_vdot(grads_base, grads_reg).real
                norm_sq = tree_vdot(grads_base, grads_base).real
                proj_scale = jnp.where(dot < 0.0, dot / (norm_sq + 1e-8), 0.0)
                grads_reg = jax.tree_map(lambda gr, gb: gr - proj_scale * gb, grads_reg, grads_base)

            grads_final = jax.tree_map(lambda gb, gr: gb + gr, grads_base, grads_reg)
            vf_state = vf_state.apply_gradients(grads=grads_final)
        else:
            if use_pcgrad and (sphere_target_ratio > 0.0 or sphere_gating_ratio > 0.0):
                (vf_loss_value, (sphere_loss_critic, sphere_loss_critic_gate)), grads_total = jax.value_and_grad(
                    critic_loss, has_aux=True
                )(vf_state.params)
                _, grads_base = jax.value_and_grad(critic_base_loss)(vf_state.params)

                grads_reg = jax.tree_map(lambda gt, gb: gt - gb, grads_total, grads_base)
                dot = tree_vdot(grads_base, grads_reg).real
                norm_sq = tree_vdot(grads_base, grads_base).real
                proj_scale = jnp.where(dot < 0.0, dot / (norm_sq + 1e-8), 0.0)
                grads_reg_proj = jax.tree_map(lambda gr, gb: gr - proj_scale * gb, grads_reg, grads_base)
                grads_pc = jax.tree_map(lambda gb, grp: gb + grp, grads_base, grads_reg_proj)
                vf_state = vf_state.apply_gradients(grads=grads_pc)
            else:
                (vf_loss_value, (sphere_loss_critic, sphere_loss_critic_gate)), grads = jax.value_and_grad(
                    critic_loss, has_aux=True
                )(vf_state.params)
                vf_state = vf_state.apply_gradients(grads=grads)

        return (
            (actor_state, vf_state),
            (
                pg_loss_value,
                policy_loss,
                entropy_loss,
                vf_loss_value,
                ratio,
                sphere_loss_actor,
                sphere_loss_critic,
                sphere_loss_actor_gate,
                sphere_loss_critic_gate,
            ),
        )

    def train(self) -> None:  # type: ignore[override]
        if self.target_kl is None:
            self._update_learning_rate(
                [self.policy.actor_state.opt_state[1], self.policy.vf_state.opt_state[1]],
                learning_rate=self.lr_schedule(self._current_progress_remaining),
            )
        clip_range = self.clip_range_schedule(self._current_progress_remaining)
        n_updates = 0
        mean_clip_fraction = 0.0
        mean_kl_div = 0.0
        mean_sphere_actor = 0.0
        mean_sphere_critic = 0.0
        mean_sphere_actor_gate = 0.0
        mean_sphere_critic_gate = 0.0

        for _ in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):  # type: ignore[attr-defined]
                n_updates += 1
                actions = (
                    rollout_data.actions.flatten().numpy().astype(np.int32)
                    if isinstance(self.action_space, spaces.Discrete)
                    else rollout_data.actions.numpy()
                )

                (self.policy.actor_state, self.policy.vf_state), (
                    pg_loss,
                    policy_loss,
                    entropy_loss,
                    value_loss,
                    ratio,
                    sphere_loss_actor,
                    sphere_loss_critic,
                    sphere_loss_actor_gate,
                    sphere_loss_critic_gate,
                ) = self._one_update_with_sphere(
                    actor_state=self.policy.actor_state,
                    vf_state=self.policy.vf_state,
                    observations=rollout_data.observations.numpy(),
                    actions=actions,
                    advantages=rollout_data.advantages.numpy(),
                    returns=rollout_data.returns.numpy(),
                    old_log_prob=rollout_data.old_log_prob.numpy(),
                    clip_range=clip_range,
                    ent_coef=self.ent_coef,
                    vf_coef=self.vf_coef,
                    normalize_advantage=self.normalize_advantage,
                    sphere_target_ratio=self.sphere_target_ratio,
                    sphere_gating_ratio=self.sphere_gating_ratio,
                    use_pcgrad=self.sphere_use_pcgrad,
                    sphere_scale_mode=self.sphere_scale_mode,
                )

                eps = 1e-7
                approx_kl_div = jnp.mean((ratio - 1.0 + eps) - jnp.log(ratio + eps)).item()
                clip_fraction = jnp.mean(jnp.abs(ratio - 1) > clip_range).item()
                mean_clip_fraction += (clip_fraction - mean_clip_fraction) / n_updates
                mean_kl_div += (approx_kl_div - mean_kl_div) / n_updates
                if self.target_kl is not None:
                    self.adaptive_lr.update(approx_kl_div)
                    self._update_learning_rate(
                        [self.policy.actor_state.opt_state[1], self.policy.vf_state.opt_state[1]],
                        learning_rate=self.adaptive_lr.current_adaptive_lr,
                    )
                if self.sphere_target_ratio > 0.0:
                    mean_sphere_actor += (sphere_loss_actor.item() - mean_sphere_actor) / n_updates
                    mean_sphere_critic += (sphere_loss_critic.item() - mean_sphere_critic) / n_updates
                if self.sphere_gating_ratio > 0.0:
                    mean_sphere_actor_gate += (sphere_loss_actor_gate.item() - mean_sphere_actor_gate) / n_updates
                    mean_sphere_critic_gate += (sphere_loss_critic_gate.item() - mean_sphere_critic_gate) / n_updates

        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),  # type: ignore[attr-defined]
            self.rollout_buffer.returns.flatten(),  # type: ignore[attr-defined]
        )

        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/policy_gradient_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        self.logger.record("train/approx_kl", mean_kl_div)
        self.logger.record("train/clip_fraction", mean_clip_fraction)
        self.logger.record("train/pg_loss", pg_loss.item())
        self.logger.record("train/explained_variance", explained_var)
        try:
            log_std = self.policy.actor_state.params["params"]["log_std"]
            self.logger.record("train/std", np.exp(log_std).mean().item())
        except KeyError:
            pass
        if self.sphere_target_ratio > 0.0:
            self.logger.record("train/sphere_actor_loss", mean_sphere_actor)
            self.logger.record("train/sphere_critic_loss", mean_sphere_critic)
        if self.sphere_gating_ratio > 0.0:
            self.logger.record("train/sphere_actor_gate_loss", mean_sphere_actor_gate)
            self.logger.record("train/sphere_critic_gate_loss", mean_sphere_critic_gate)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
