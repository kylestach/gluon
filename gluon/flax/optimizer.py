import math
from typing import Any, Dict, NamedTuple, Optional

import chex
import jax
import jax.numpy as jnp
import optax


class ScheduleFreeSGDState(NamedTuple):
    """State for the ScheduleFreeAdam optimizer."""

    t: chex.Array
    x: optax.Params

    def z(self, y, beta):
        return jax.tree.map(lambda x, y: x + (y - x) / (1 - beta), self.x, y)


def schedule_free_sgd(
    learning_rate: float,
    warmup_steps: int = 0,
    weight_decay: float = 1e-3,
    beta: float = 0.9,
) -> optax.GradientTransformation:
    peak_learning_rate = learning_rate

    def init_fn(params: optax.Params) -> ScheduleFreeSGDState:
        return ScheduleFreeSGDState(
            t=jnp.zeros([], jnp.int32),
            x=params,
        )

    def update_fn(updates, state, params=None):
        t = state.t + 1
        learning_rate = peak_learning_rate * jnp.minimum(1.0, t / warmup_steps)

        def _adam_update(g, x, y, z):
            # First-order update
            z = z + learning_rate * (-g - weight_decay * y)

            # Accumulated learning rate sum_{i<=t} lr(i)
            #  - t <= warmup_steps: learning_rate * t / 2
            #  - t > warmup_steps: learning_rate * (t - warmup_steps / 2)
            # c_t = lr(t) / sum_{i<=t} lr(i)
            #  - t <= warmup_steps: 1 / (t / 2)
            #  - t > warmup_steps: 1 / (t - warmup_steps / 2)
            c = 1 / jnp.where(
                t <= warmup_steps,
                t / 2,
                t - warmup_steps / 2,
            )

            # Average of the past updates, weighted by learning rate
            x = (1 - c) * x + c * z
            new_y = beta * x + (1 - beta) * z

            return x, y, z, new_y - y

        y = params
        x = state.x
        z = state.z(y, beta=state.t + 1)

        adam_output_results = jax.tree.map(_adam_update, updates, x, y, z)
        x, y, z, y_updates = jax.tree.transpose(
            jax.tree_structure(updates), None, adam_output_results
        )

        return y_updates, ScheduleFreeSGDState(t=state.t + 1, x=x)

    return optax.GradientTransformation(init_fn, update_fn)


class ScheduleFreeAdamState(NamedTuple):
    """State for the ScheduleFreeAdam optimizer."""

    t: chex.Array
    x: optax.Params
    nu: optax.Updates

    def z(self, y, beta):
        return jax.tree.map(lambda x, y: x + (y - x) / (1 - beta), self.x, y)


def schedule_free_adam(
    learning_rate: float,
    warmup_steps: int = 0,
    weight_decay: float = 0,
    b1: float = 0.9,
    b2: float = 0.999,
    epsilon: float = 1e-8,
) -> optax.GradientTransformation:
    peak_learning_rate = learning_rate

    def init_fn(params: optax.Params) -> ScheduleFreeAdamState:
        return ScheduleFreeAdamState(
            t=jnp.zeros([], jnp.int32),
            x=params,
            nu=jax.tree.map(jnp.zeros_like, params),
        )

    def update_fn(updates, state, params=None):
        t = state.t + 1
        learning_rate = peak_learning_rate * jnp.minimum(1.0, t / warmup_steps)

        def _adam_update(g, x, y, z, nu):
            # Compute second-order momentum
            nu = b2 * nu + (1 - b2) * g**2
            nu_hat = nu / (1 - b2 ** (state.t + 1))

            # First-order update
            z = z - learning_rate * g / (jnp.sqrt(nu_hat) + epsilon)
            if weight_decay > 0:
                z = z - learning_rate * weight_decay * y

            # Accumulated learning rate sum_{i<=t} lr(i)
            #  - t <= warmup_steps: learning_rate * t / 2
            #  - t > warmup_steps: learning_rate * (t - warmup_steps / 2)
            # c_t = lr(t) / sum_{i<=t} lr(i)
            #  - t <= warmup_steps: 1 / (t / 2)
            #  - t > warmup_steps: 1 / (t - warmup_steps / 2)
            c = 1 / jnp.where(
                t <= warmup_steps,
                t / 2,
                t - warmup_steps / 2,
            )

            # Average of the past updates, weighted by learning rate
            x = (1 - c) * x + c * z
            new_y = b1 * x + (1 - b1) * z

            return x, y, z, nu, new_y - y

        x = state.x
        y = params
        z = state.z(y, beta=b1)

        adam_output_results = jax.tree.map(_adam_update, updates, x, y, z, state.nu)
        x, y, z, nu, y_updates = jax.tree.transpose(
            jax.tree_structure(updates), None, adam_output_results
        )

        return y_updates, ScheduleFreeAdamState(t=state.t + 1, x=x, nu=nu)

    return optax.GradientTransformation(init_fn, update_fn)


def make_optimizer(
    *,
    optimizer: str,
    learning_rate: float = 1e-3,
    max_grad_norm: Optional[float] = None,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    schedule: Optional[str] = None,
    warmup_steps: int = 0,
    total_steps: Optional[int] = None,
    final_learning_rate: Optional[float] = None,
) -> optax.GradientTransformation:
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    @optax.inject_hyperparams
    def _make_optimizer(lr):
        if optimizer == "adam":
            opt = optax.adam(learning_rate=lr, **optimizer_kwargs)
        elif optimizer == "adamw":
            opt = optax.adamw(learning_rate=lr, **optimizer_kwargs)
        elif optimizer == "schedule_free_adam":
            assert (
                schedule is None
            ), f"schedule is not supported for schedule_free_adam but got {schedule}"
            opt = schedule_free_adam(
                learning_rate=learning_rate,
                warmup_steps=warmup_steps,
                **optimizer_kwargs,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        if max_grad_norm is not None:
            opt = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                opt,
            )

        return opt

    if final_learning_rate is None:
        final_learning_rate = 0.0

    if schedule is None or schedule == "constant":
        lr = optax.linear_schedule(
            init_value=0.0, end_value=learning_rate, transition_steps=warmup_steps
        )
    elif schedule in {"warmup_cosine_decay", "cosine_decay", "cosine"}:
        assert (
            total_steps is not None
        ), "total_steps must be provided for warmup_cosine_decay schedule"
        lr = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
        )
    elif schedule in {"warmup_exponential_decay", "exponential_decay", "exponential"}:
        assert (
            total_steps is not None
        ), "total_steps must be provided for warmup_exponential_decay schedule"
        assert (
            final_learning_rate > 0
        ), "final_learning_rate must be provided and positive"
        decay_rate = math.log(final_learning_rate / learning_rate) / total_steps
        lr = optax.warmup_exponential_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_rate=decay_rate,
            transition_steps=total_steps,
        )
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    return _make_optimizer(lr=lr)
