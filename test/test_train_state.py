import os
import tempfile

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import pytest
from flax import serialization

from gluon.flax import ModuleSpec, TrainState


class SimpleModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


class BatchNormModel(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Dense(features=64)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


@pytest.fixture
def example_batch():
    return jnp.ones((32, 784))  # Example MNIST-like batch


@pytest.fixture
def train_state(example_batch):
    rng = jax.random.PRNGKey(0)
    tx = optax.adam(learning_rate=0.001)
    return TrainState.create(
        module_spec=ModuleSpec(SimpleModel, {}),
        optimizer_spec=ModuleSpec(optax.adam, {"learning_rate": 0.001}),
        example_batch=example_batch,
        rng=rng,
    )


def test_create(train_state, example_batch):
    assert isinstance(train_state, TrainState)
    assert isinstance(train_state.module, SimpleModel)
    assert train_state.example_batch.shape == example_batch.shape
    assert train_state.step == 0


def test_apply_gradients(train_state):
    # Dummy gradients
    grads = jax.tree.map(jnp.ones_like, train_state.params)
    new_state = train_state.apply_gradients(grads)

    assert new_state.step == train_state.step + 1
    assert not jax.tree_util.tree_all(
        jax.tree.map(
            lambda x, y: jnp.array_equal(x, y), new_state.params, train_state.params
        )
    )


def test_get_bound_module(train_state, example_batch):
    bound_module = train_state.get_bound_module()
    output = bound_module(example_batch)
    assert output.shape == (32, 10)  # Based on SimpleModel output


def test_save_and_load_with_manager(train_state, example_batch):
    with tempfile.TemporaryDirectory() as tmpdir:
        options = ocp.CheckpointManagerOptions(max_to_keep=5, create=True)
        tmpdir = ocp.test_utils.erase_and_create_empty(tmpdir)
        checkpoint_dir = os.path.join(tmpdir, "checkpoints")
        with ocp.CheckpointManager(checkpoint_dir, options=options) as manager:
            # Save the state
            train_state.save(tmpdir, manager=manager, wait=True)

            # Check if files were created
            assert os.path.exists(os.path.join(tmpdir, "module_spec.json"))
            assert os.path.exists(os.path.join(tmpdir, "example_batch.msgpack"))
            assert os.path.exists(os.path.join(tmpdir, "checkpoints"))

            # Load the state
            loaded_state = TrainState.load(tmpdir, train_state.tx, manager=manager)

            # Check if loaded state matches the original
            assert isinstance(loaded_state, TrainState)
            assert isinstance(loaded_state.module, SimpleModel)
            assert jnp.array_equal(
                loaded_state.example_batch, train_state.example_batch
            )
            assert loaded_state.step == train_state.step
            assert jax.tree_util.tree_all(
                jax.tree.map(
                    lambda x, y: jnp.array_equal(x, y),
                    loaded_state.params,
                    train_state.params,
                )
            )


def test_save_and_load_without_manager(train_state, example_batch):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save the state
        train_state.save(tmpdir, wait=True)

        # Check if files were created
        assert os.path.exists(os.path.join(tmpdir, "module_spec.json"))
        assert os.path.exists(os.path.join(tmpdir, "example_batch.msgpack"))
        assert os.path.exists(os.path.join(tmpdir, "checkpoints"))

        # Load the state
        loaded_state = TrainState.load(tmpdir, train_state.tx)

        # Check if loaded state matches the original
        assert isinstance(loaded_state, TrainState)
        assert isinstance(loaded_state.module, SimpleModel)
        assert jnp.array_equal(loaded_state.example_batch, train_state.example_batch)
        assert loaded_state.step == train_state.step
        assert jax.tree_util.tree_all(
            jax.tree.map(
                lambda x, y: jnp.array_equal(x, y),
                loaded_state.params,
                train_state.params,
            )
        )


def test_next_rng(train_state):
    new_state = train_state.next_rng()
    assert not jnp.array_equal(new_state.rng, train_state.rng)


def test_apply_property(train_state):
    assert callable(train_state.apply)


def test_module_spec_serialization():
    spec = ModuleSpec(SimpleModel, {})
    json_str = spec.to_json()
    loaded_spec = ModuleSpec.from_json(json_str)
    assert loaded_spec.ctor == spec.ctor
    assert loaded_spec.config == spec.config


def test_with_variables(example_batch):
    rng = jax.random.PRNGKey(0)
    model = TrainState.create(
        module_spec=ModuleSpec(BatchNormModel, {}),
        optimizer_spec=ModuleSpec(optax.adam, {"learning_rate": 0.001}),
        example_batch=example_batch,
        rng=rng,
        init_kwargs={"train": True},
    )

    assert "batch_stats" in model.variables


if __name__ == "__main__":
    pytest.main([__file__])
