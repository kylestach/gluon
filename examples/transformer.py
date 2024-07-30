%load_ext autoreload
%autoreload 2

#|%%--%%| <EF4WZMh74o|aWFu4PxL46>

from flax.typing import Array
from gluon.flax.transformers import DiscreteSequenceTransformer
from gluon.flax import make_optimizer
import numpy as np
import jax.numpy as jnp
from flax import nnx
import tqdm

#|%%--%%| <aWFu4PxL46|m9HjOSrRV6>

model = DiscreteSequenceTransformer(d_model=64, vocab_size=16, max_len=16, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, make_optimizer(optimizer="adam", warmup_steps=100, total_steps=10000, schedule="cosine"))

#|%%--%%| <m9HjOSrRV6|4RFZDKMFjE>

@nnx.jit
def train_step(model: DiscreteSequenceTransformer, optimizer: nnx.Optimizer, batch: Array, rngs: nnx.Rngs):
    def loss_fn(model, batch, rngs=rngs):
        return model.loss(batch, train=True, rngs=rngs, self_mask="causal", cross_mask=None)

    loss, grad = nnx.value_and_grad(loss_fn)(model, batch)

    optimizer.update(grad)

    return loss


for epoch in range(10):
    with tqdm.trange(100) as pbar:
        for i in pbar:
            batch = (jnp.arange(16)[None, :] + np.random.randint(0, 16, size=(16,))[:, None]) % 16
            loss = train_step(model, optimizer, batch, nnx.Rngs(0))
            pbar.set_postfix({"loss": loss})

    model.init_cache((16, 16))
    print(model.decode(start_token=jnp.arange(16), train=False, rngs=nnx.Rngs(0), decode_length=16))


#|%%--%%| <4RFZDKMFjE|eQlPddlyMY>

model.init_cache((16, 15))
print(model.decode(start_token=jnp.arange(16)[:, None], train=False, rngs=nnx.Rngs(0), decode_length=15))

