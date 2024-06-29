import optax
from typing import Dict, Any, Optional

def make_optimizer(
    *,
    optimizer: str,
    learning_rate: float = 1e-3,
    max_grad_norm: Optional[float] = None,
    optimizer_kwargs: Dict[str, Any] = None,
    schedule: str = None,
    warmup_steps: int = 0,
    total_steps: int = None,
) -> optax.GradientTransformation:
    @optax.inject_hyperparams
    def _make_optimizer(lr, kwargs):
        if optimizer == "adam":
            opt = optax.adam(
                learning_rate=lr, **kwargs
            )
        elif optimizer == "adamw":
            opt = optax.adamw(
                learning_rate=lr, **kwargs
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        if max_grad_norm is not None:
            opt = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                opt,
            )

        return opt

    if schedule is None or schedule == "constant":
        lr = learning_rate
    elif schedule == "warmup_cosine_decay":
        lr = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
        )
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    return _make_optimizer(lr=lr, kwargs=optimizer_kwargs)
