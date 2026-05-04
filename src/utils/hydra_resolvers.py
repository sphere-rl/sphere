from omegaconf import OmegaConf
import flax
import optax
from src.training.training_utils import parse_linear_scheduler


# Provide a stable identity function for nn resolver
def identity(x):
    return x


def _fair_width(pm, base=256):
    return int(float(base) * (float(pm) ** 0.5) + 0.5)


def _fair_arch(pm, base=256, depth=3):
    w = _fair_width(pm, base)
    return [w] * int(depth)


def _arch_from(w, pm, base=256, depth=3):
    tag = "" if w is None else str(w).lower()
    use_given = tag not in ("", "null", "none")
    width = int(w) if use_given else _fair_width(pm, base)
    return [width] * int(depth)


def register_omega_resolvers() -> None:
    # Generic utilities
    OmegaConf.register_new_resolver("linear_scheduling", lambda v: parse_linear_scheduler(v))
    OmegaConf.register_new_resolver("len", lambda x: len(x))
    OmegaConf.register_new_resolver("plus_one", lambda x: x + 1)
    def nn_resolver(v: str):
        # Allow `${nn:identity}` even though flax.linen has no identity op
        if v == "identity":
            return identity
        return getattr(flax.linen, v)

    OmegaConf.register_new_resolver("nn", nn_resolver)
    OmegaConf.register_new_resolver("optax", lambda v: getattr(optax, v))
    # Fair width/arch utilities
    OmegaConf.register_new_resolver("fair_width", _fair_width)
    OmegaConf.register_new_resolver("fair_arch", _fair_arch)
    OmegaConf.register_new_resolver("arch_from", _arch_from)
