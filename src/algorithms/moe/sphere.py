from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def sphere_output_loss(probs: jnp.ndarray) -> jnp.ndarray:
    """SPHERE-style penalty on routing probabilities.

    Args:
        probs: Array of shape [c, b, n] where tokens T=c*b, n=#experts.

    Returns:
        Spectral-variance style penalty on (PᵀP)/T, normalized by n.
    """
    c, b, n = probs.shape
    tokens = c * b
    P = probs.reshape(tokens, n)
    T_f = jnp.asarray(tokens, dtype=P.dtype)
    G_bar = (P.T @ P) / T_f
    fro2 = jnp.sum(G_bar * G_bar)
    tr = jnp.trace(G_bar)
    n_f = jnp.asarray(n, dtype=G_bar.dtype)
    loss = fro2 - (tr * tr) / n_f
    return loss / n_f


@jax.jit
def sphere_feature_loss(x: jnp.ndarray) -> jnp.ndarray:
    """SPHERE penalty on feature Gram matrix.

    Args:
        x: Φ ∈ R^{T×D}, rows are concatenated gating-weighted expert features.
    """
    T, D = x.shape
    T_f = jnp.asarray(T, dtype=x.dtype)

    if D <= T:
        A = (x.T @ x) / T_f
        fro2 = jnp.sum(A * A)
        tr = jnp.trace(A)
    else:
        K = (x @ x.T) / T_f
        fro2 = jnp.sum(K * K)
        tr = jnp.trace(K)

    D_f = jnp.asarray(D, dtype=x.dtype)
    return fro2 - (tr * tr) / D_f
