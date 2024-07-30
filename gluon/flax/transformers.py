from typing import Literal, Optional, Callable

import jax
from jax import Array, lax
import jax.numpy as jnp

import einops

from flax import nnx
from flax.typing import Dtype, PrecisionLike, Shape
from flax.nnx.nnx.nn.attention import dot_product_attention_weights
import optax


AttentionType = Literal["dot_product", "rope", "alibi"]
PositionalEncodingType = Literal["sinusoidal", "learned", "rope", "alibi"]


def sin_embed_init(key, shape, dtype):
    # sin/cos positional encoding
    max_len, d_model = shape

    embed = jnp.zeros((max_len, d_model), dtype=dtype)
    pos = jnp.arange(max_len)[:, None]
    dim = jnp.arange(d_model // 2)[None, :]
    ts = 10000 ** -(2 * dim / d_model)
    embed = jnp.concatenate(
        [
            jnp.sin(pos * ts),
            jnp.cos(pos * ts),
        ],
        axis=-1,
    )
    return embed


def rope_attention_weights(
    query: Array,
    key: Array,
    **kwargs,
) -> Array:
    """Computes attention weights according to RoPE."""

    # rotate qkv
    def rotate(x):
        dim = x.shape[-1]
        idcs = jnp.arange(x.shape[-2])
        ts = 10000 ** -(2 * jnp.linspace(0, 1, x.shape[-1] // 2) / dim)
        ct = jnp.cos(ts * idcs[:, None])
        st = jnp.sin(ts * idcs[:, None])
        return jnp.concatenate(
            [
                x[..., : dim // 2] * ct - x[..., dim // 2 :] * st,
                x[..., : dim // 2] * st + x[..., dim // 2 :] * ct,
            ],
            axis=-1,
        )

    return dot_product_attention_weights(
        rotate(query),
        rotate(key),
        **kwargs,
    )


def rope_attention(
    query: Array,
    key: Array,
    value: Array,
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    broadcast_dropout: bool = True,
    dropout_rng: Optional[Array] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Optional[Dtype] = None,
    precision: PrecisionLike = None,
    module: Optional[nnx.Module] = None,
):
    """Computes Rotary Positional Encoding (RoPE) from query, key, and value.

    See `nnx.dot_product_attention` for more details.
    """
    query, key, value = promote_dtype((query, key, value), dtype=dtype)  # type: ignore[bad-unpacking]
    dtype = query.dtype
    assert key.ndim == query.ndim == value.ndim, "q, k, v must have same rank."
    assert (
        query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
    ), "q, k, v batch dims must match."
    assert (
        query.shape[-2] == key.shape[-2] == value.shape[-2]
    ), "q, k, v num_heads must match."
    assert key.shape[-3] == value.shape[-3], "k, v lengths must match."

    # compute attention weights
    attn_weights = rope_attention_weights(
        query,
        key,
        bias=bias,
        mask=mask,
        broadcast_dropout=broadcast_dropout,
        dropout_rng=dropout_rng,
        dropout_rate=dropout_rate,
        deterministic=deterministic,
        dtype=dtype,
        precision=precision,
        module=module,
    )

    # return weighted sum over values for each query position
    return jnp.einsum("...hqk,...khd->...qhd", attn_weights, value, precision=precision)


def alibi_attention(
    query: Array,
    key: Array,
    value: Array,
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    broadcast_dropout: bool = True,
    dropout_rng: Optional[Array] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Optional[Dtype] = None,
    precision: PrecisionLike = None,
    module: Optional[nnx.Module] = None,
):
    """Computes Attention with Linear Biases (ALiBi) from query, key, and value.

    See `nnx.dot_product_attention` for more details.
    """
    num_heads, query_len, _ = query.shape[-3:]
    kv_len = key.shape[-2]

    q_index = jnp.arange(query_len)
    kv_index = jnp.arange(kv_len)
    alibi_base = q_index[..., None] - kv_index[..., None, :]

    alibi_slopes = 2 ** jnp.linspace(1, 8, num_heads)
    alibi_bias = alibi_slopes[:, None, None] * alibi_base[None, :, :]

    bias = bias + alibi_bias if bias is not None else alibi_bias

    return nnx.dot_product_attention(
        query,
        key,
        value,
        bias=bias,
        mask=mask,
        broadcast_dropout=broadcast_dropout,
        dropout_rng=dropout_rng,
        dropout_rate=dropout_rate,
        deterministic=deterministic,
        dtype=dtype,
        precision=precision,
        module=module,
    )


class AdditivePositionalEncoding(nnx.Module):
    def __init__(self, max_len, d_model, learned: bool, rngs: nnx.Rngs):
        self.learned = learned
        self.embed = nnx.Param(
            sin_embed_init(rngs.params(), (max_len, d_model), jnp.float32),
        )
        self.cache_index: nnx.Cache[Array] | None = None

    def __call__(self, x, *, decode: bool = False):
        if decode:
            assert self.cache_index is not None, "cache_index must be initialized"
            embed = self.embed[self.cache_index.value]
            self.cache_index.value += 1
        else:
            embed = self.embed[: x.shape[-2]]

        if not self.learned:
            embed = lax.stop_gradient(embed)

        return x + embed

    def init_cache(self, input_shape):
        self.cache_index = nnx.Cache(jnp.zeros(input_shape[:-1], jnp.uint32))


class TransformerLayer(nnx.Module):
    def __init__(
        self,
        *,
        d_model,
        n_heads=None,
        d_ff=None,
        ratio_ff=4.0,
        d_head=64,
        dropout: float = 0.0,
        d_cross_attn=None,
        attention_fn: Callable = nnx.dot_product_attention,
        rngs: nnx.Rngs,
    ):
        if n_heads is None:
            n_heads = d_model // d_head
        if d_ff is None:
            d_ff = int(d_model * ratio_ff)
        self.self_attn = nnx.MultiHeadAttention(
            num_heads=n_heads,
            in_features=d_model,
            dropout_rate=dropout,
            rngs=rngs,
            attention_fn=attention_fn,
        )
        self.ffn = nnx.Sequential(
            nnx.Linear(d_model, d_ff, rngs=rngs),
            nnx.gelu,
            nnx.Linear(d_ff, d_model, rngs=rngs),
        )
        self.ln1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.ln2 = nnx.LayerNorm(d_model, rngs=rngs)

        if d_cross_attn is not None:
            self.cross_attn = nnx.MultiHeadAttention(
                num_heads=n_heads,
                in_features=d_model,
                dropout_rate=dropout,
                rngs=rngs,
                attention_fn=attention_fn,
            )
            self.ln3 = nnx.LayerNorm(d_model, rngs=rngs)
            self.ln4 = nnx.LayerNorm(d_model, rngs=rngs)

    def __call__(
        self,
        x,
        y=None,
        *,
        self_mask: Literal["causal"] | Array | None = None,
        cross_mask: Array | None = None,
        train: bool,
        decode: bool = False,
        rngs: nnx.Rngs,
    ):
        if self_mask == "causal":
            self_mask = nnx.make_causal_mask(x[..., 0])

        x = x + self.self_attn(
            self.ln1(x),
            deterministic=not train,
            decode=decode,
            mask=self_mask,
            rngs=rngs,
        )
        x = x + self.ffn(self.ln2(x), rngs=rngs)

        if hasattr(self, "cross_attn"):
            x = x + self.cross_attn(
                self.ln3(x),
                self.ln4(y),
                deterministic=not train,
                decode=False,
                mask=cross_mask,
                rngs=rngs,
            )

        return x

    def init_cache(self, input_shape: Shape, *, dtype: Dtype = jnp.float32):
        self.self_attn.init_cache(input_shape, dtype=dtype)


class TransformerBackbone(nnx.Module):
    def __init__(
        self,
        *,
        d_model,
        n_heads=None,
        d_ff=None,
        ratio_ff=4.0,
        d_head=64,
        dropout: float = 0.0,
        n_layers=6,
        d_cross_attn=None,
        attention_fn: Callable,
        rngs: nnx.Rngs,
    ):
        self.layers = [
            TransformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                ratio_ff=ratio_ff,
                d_head=d_head,
                dropout=dropout,
                d_cross_attn=d_cross_attn,
                attention_fn=attention_fn,
                rngs=rngs,
            )
            for _ in range(n_layers)
        ]

    def __call__(self, x, y=None, *, rngs: nnx.Rngs, **kwargs):
        for layer in self.layers:
            x = layer(x, y, rngs=rngs, **kwargs)
        return x

    def init_cache(self, input_shape: Shape, *, dtype: Dtype = jnp.float32):
        for layer in self.layers:
            layer.init_cache(input_shape, dtype=dtype)


class DiscreteSequenceTransformer(nnx.Module):
    def __init__(
        self,
        *,
        d_model: int,
        vocab_size: int,
        max_len: int,
        rngs: nnx.Rngs,
        position_encoding_type: PositionalEncodingType = "learned",
        **backbone_kwargs,
    ):
        self.embedding = nnx.Embed(vocab_size, d_model, rngs=rngs)

        if position_encoding_type == "learned":
            self.position_encoding = AdditivePositionalEncoding(
                max_len, d_model, learned=True, rngs=rngs
            )
            self.attention_fn = nnx.dot_product_attention
        elif position_encoding_type == "sinusoidal":
            self.position_encoding = AdditivePositionalEncoding(
                max_len, d_model, learned=False, rngs=rngs
            )
            self.attention_fn = nnx.dot_product_attention
        elif position_encoding_type == "rope":
            self.attention_fn = rope_attention
        elif position_encoding_type == "alibi":
            self.attention_fn = alibi_attention

        self.backbone = TransformerBackbone(
            rngs=rngs,
            attention_fn=self.attention_fn,
            d_model=d_model,
            **backbone_kwargs,
        )
        self.d_model = d_model

    def __call__(self, x, y=None, *, rngs: nnx.Rngs, **kwargs):
        x = self.embedding(x)
        for layer in self.backbone.layers:
            x = layer(x, y, **kwargs, rngs=rngs)
        return einops.einsum(x, self.embedding.embedding.value, "... d, v d -> ... v")

    def step_decode(
        self, x, y=None, *, train: bool, temperature: float = 1.0, rngs: nnx.Rngs
    ):
        output_logits = self(x, y, train=train, decode=True, rngs=rngs)

        if temperature == 0:
            new_token = jnp.argmax(output_logits, axis=-1)
        else:
            new_token = jax.random.categorical(
                rngs.sample(), output_logits / temperature
            )

        return new_token

    def decode(
        self,
        start_token,
        y=None,
        *,
        train: bool,
        temperature: float = 1.0,
        decode_length: int,
        rngs: nnx.Rngs,
    ):
        def _step_decode(carry, _, *, rngs):
            module: "DiscreteSequenceTransformer"
            token: Array
            module, token = carry
            new_token = module.step_decode(
                token, y=y, train=train, temperature=temperature, rngs=rngs
            )
            return (module, new_token), new_token

        scan_fn = nnx.scan(
            _step_decode, length=decode_length, in_axes=None, out_axes=-1, state_axes={}
        )

        _, output = scan_fn(
            (self, start_token),
            None,
            rngs=rngs,
        )

        return output

    def loss_with_logits(self, x, y=None, *, train: bool, rngs: nnx.Rngs, self_mask=None, cross_mask=None):
        inputs = x[..., :-1]
        labels = x[..., 1:]
        output_logits = self(inputs, y, train=train, decode=False, rngs=rngs, self_mask=self_mask, cross_mask=cross_mask)

        return (
            jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(output_logits, labels)
            ),
            output_logits,
        )

    def loss(self, x, y=None, *, train: bool, rngs: nnx.Rngs, self_mask=None, cross_mask=None):
        return self.loss_with_logits(x, y, train=train, rngs=rngs, self_mask=self_mask, cross_mask=cross_mask)[0]

    def init_cache(self, input_shape: Shape, *, dtype: Dtype = jnp.float32):
        if hasattr(self, "position_encoding") and hasattr(
            self.position_encoding, "init_cache"
        ):
            self.position_encoding.init_cache(input_shape)
        self.backbone.init_cache((*input_shape, self.d_model), dtype=dtype)
