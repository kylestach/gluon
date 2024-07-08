import dataclasses
import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import tensorflow as tf
from flax import linen as nn
from flax import serialization, struct

from gluon.flax.spec import ModuleSpec, OptimizerSpec
from gluon.typing import Data, Info, Params, Variables


@struct.dataclass
class TrainState:
    module_spec: ModuleSpec = struct.field(pytree_node=False)
    optimizer_spec: OptimizerSpec = struct.field(pytree_node=False)
    example_batch: Any = struct.field(pytree_node=True)

    module: nn.Module = struct.field(pytree_node=False)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    params: Params = struct.field(pytree_node=True)
    variables: Variables = struct.field(pytree_node=True)
    opt_state: optax.OptState = struct.field(pytree_node=True)
    step: jnp.ndarray = struct.field(pytree_node=True)
    rng: jnp.ndarray = struct.field(
        pytree_node=True,
    )

    @classmethod
    def create(
        cls,
        module_spec: ModuleSpec,
        optimizer_spec: OptimizerSpec,
        example_batch: Data,
        rng: jnp.ndarray,
        init_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "TrainState":
        module = module_spec.instantiate()
        tx = optimizer_spec.instantiate()

        if init_kwargs is None:
            init_kwargs = {}

        if isinstance(example_batch, dict):
            module_variables = module.init(rng, **example_batch, **init_kwargs)
        elif isinstance(example_batch, (list, tuple)):
            module_variables = module.init(rng, *example_batch, **init_kwargs)
        else:
            module_variables = module.init(rng, example_batch, **init_kwargs)

        params = module_variables.pop("params")
        assert (
            params is Params
        ), f"Expected params to be of type Params, got {type(params)}"

        optimizer_state = tx.init(params)

        return cls(
            module_spec=module_spec,
            optimizer_spec=optimizer_spec,
            module=module,
            example_batch=example_batch,
            params=params,
            tx=tx,
            opt_state=optimizer_state,
            step=jnp.array(0),
            variables=module_variables,
            rng=rng,
        )

    def replace(self, **updates) -> "TrainState":
        return dataclasses.replace(self, **updates)

    def apply_gradients(self, grads: Any) -> "TrainState":
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            params=new_params,
            opt_state=new_opt_state,
            step=self.step + 1,
        )

    def apply_loss(
        self, loss_fn: Callable, batch: Data, has_aux: bool = False
    ) -> Tuple["TrainState", Info] | "TrainState":
        grad_fn = jax.grad(loss_fn, has_aux=has_aux)
        if has_aux:
            grads, info = grad_fn(self.params, batch)
            return self.apply_gradients(grads=grads), info
        else:
            grads = grad_fn(self.params, batch)
            return self.apply_gradients(grads=grads)

    def apply(
        self,
        variables: Variables,
        *args,
        method: Optional[str | Callable] = None,
        **kwargs,
    ) -> Any:
        if method is None:
            method = self.module.__call__

        variables = {
            **self.variables,
            "params": self.params,
            **variables,
        }
        result = self.module.apply(variables, *args, method=method, **kwargs)
        result = jax.block_until_ready(result)
        return result

    def get_bound_module(self, variables: Optional[Dict[str, Any]] = None) -> Callable:
        if variables is None:
            variables = {"params": self.params, **self.variables}
        return self.module.bind(variables)

    @property
    def state_dict(self) -> Dict[str, Any]:
        return {
            "params": self.params,
            "opt_state": self.opt_state,
            "step": self.step,
            "variables": self.variables,
            "rng": self.rng,
        }

    def save(
        self,
        directory: str,
        manager: Optional[ocp.CheckpointManager] = None,
        wait: bool = False,
    ) -> None:
        tf.io.gfile.makedirs(directory)

        # Save ModuleSpec as JSON
        with tf.io.gfile.GFile(os.path.join(directory, "module_spec.json"), "w") as f:
            f.write(self.module_spec.to_json())
        with tf.io.gfile.GFile(
            os.path.join(directory, "optimizer_spec.json"), "w"
        ) as f:
            f.write(self.optimizer_spec.to_json())

        # Save example_batch using msgpack
        with tf.io.gfile.GFile(
            os.path.join(directory, "example_batch.msgpack"), "wb"
        ) as f:
            f.write(serialization.msgpack_serialize(self.example_batch))

        # Save everything else using CheckpointManager
        if manager is None:
            manager = ocp.CheckpointManager(
                directory=tf.io.gfile.join(directory, "checkpoints"),
                options=ocp.CheckpointManagerOptions(),
            )

        manager.save(self.step.item(), args=ocp.args.StandardSave(self.state_dict))
        if wait:
            manager.wait_until_finished()

    @classmethod
    def load(
        cls,
        directory: str,
        manager: Optional[ocp.CheckpointManager] = None,
        step: Optional[int] = None,
    ) -> "TrainState":
        # Load ModuleSpec from JSON
        with tf.io.gfile.GFile(os.path.join(directory, "module_spec.json"), "r") as f:
            module_spec = ModuleSpec.from_json(f.read())
        with tf.io.gfile.GFile(
            os.path.join(directory, "optimizer_spec.json"), "r"
        ) as f:
            optimizer_spec = OptimizerSpec.from_json(f.read())

        # Load example_batch using msgpack
        with tf.io.gfile.GFile(
            os.path.join(directory, "example_batch.msgpack"), "rb"
        ) as f:
            example_batch = serialization.msgpack_restore(f.read())

        # Load everything else using CheckpointManager
        if manager is None:
            manager = ocp.CheckpointManager(
                directory=os.path.join(directory, "checkpoints"),
                options=ocp.CheckpointManagerOptions(),
            )

        if step is None:
            all_steps = manager.all_steps(read=True)
            if all_steps:
                step = max(all_steps)
            else:
                raise ValueError(f"No checkpoints found in {directory}")

        # Initialize the state
        train_state = cls.create(
            module_spec=module_spec,
            optimizer_spec=optimizer_spec,
            example_batch=example_batch,
            rng=jnp.zeros((2,), dtype=jnp.uint32),
        )

        state_dict = manager.restore(
            step, args=ocp.args.StandardRestore(train_state.state_dict)
        )
        return train_state.replace(**state_dict)

    def next_rng(self) -> "TrainState":
        return self.replace(rng=jax.random.split(self.rng)[0])
