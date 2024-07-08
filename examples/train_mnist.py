from functools import partial
from typing import Dict, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch.utils.data
from datasets.utils.py_utils import tqdm
from torch.utils.data import DataLoader

from gluon.flax import ModuleSpec, OptimizerSpec, TrainState, make_optimizer


def build_dataset() -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    # Load MNIST dataset from huggingface
    from datasets import load_dataset

    train_dataset = load_dataset("mnist", split="train")
    val_dataset = load_dataset("mnist", split="test")
    return train_dataset, val_dataset


def build_model():
    from flax import linen as nn

    model = nn.Sequential(
        [
            nn.Conv(32, (3, 3), strides=2),
            nn.relu,
            nn.Conv(64, (3, 3), strides=2),
            nn.relu,
            nn.Conv(128, (3, 3), strides=2),
            nn.relu,
            nn.Conv(256, (3, 3), strides=2),
            nn.relu,
            partial(nn.avg_pool, window_shape=(2, 2)),
            partial(jnp.squeeze, axis=(-2, -3)),
            nn.Dense(10),
        ]
    )
    return model


def build_train_state(example_batch, seed=0):
    module_spec = ModuleSpec(
        ctor=build_model,
        config={},
    )
    optimizer_spec = OptimizerSpec(
        make_optimizer,
        config=dict(
            optimizer="schedule_free_adam",
            learning_rate=1e-3,
            warmup_steps=100,
        ),
    )
    train_state = TrainState.create(
        module_spec,
        optimizer_spec,
        example_batch=example_batch,
        rng=jax.random.PRNGKey(seed),
    )

    return train_state


def collate_fn(examples):
    images = []
    labels = []
    for example in examples:
        images.append(np.asarray(example["image"]))
        labels.append(example["label"])

    pixel_values = np.stack(images)
    labels = np.array(labels)
    return {"images": pixel_values[..., None] / 255, "labels": labels}


Data = Dict[str, Union[np.ndarray, "Data"]]


@jax.jit
def update_fn(train_state: TrainState, batch: Data):
    def loss_fn(params, images, labels):
        logits = train_state.apply({"params": params}, images)
        loss = jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=labels
            )
        )
        return loss, {
            "loss": loss,
            "accuracy": jnp.mean(jnp.argmax(logits, axis=-1) == labels),
        }

    grad_fn = jax.grad(loss_fn, has_aux=True)
    grad, info = grad_fn(train_state.params, batch["images"], batch["labels"])

    info["learning_rate"] = train_state.opt_state.hyperparams["lr"]
    info["grad_norm"] = optax.global_norm(grad)

    return train_state.apply_gradients(grads=grad), info


def train():
    train_dataset, val_dataset = build_dataset()
    batch_size = 128

    example_image = np.asarray(next(iter(train_dataset))["image"])[None, ..., None]
    train_state = build_train_state(example_image)

    losses = []

    for epoch in range(10):
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, collate_fn=collate_fn
        )
        accuracy = []
        for batch in val_loader:
            _, info = update_fn(train_state, batch)
            accuracy.append(info["accuracy"])
        print(f"Validation accuracy: {np.mean(accuracy)}")

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, collate_fn=collate_fn
        )
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            train_state, info = update_fn(train_state, batch)
            pbar.set_postfix({k: f"{v:.2f}" for k, v in info.items()})
            losses.append(info["loss"])

    import matplotlib.pyplot as plt

    plt.plot(losses)
    plt.show()


train()
