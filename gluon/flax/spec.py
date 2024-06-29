from typing import Any, Callable, Dict, Generic, TypeVar
from flax import struct
import json
import importlib
import optax
from flax import linen as nn

T = TypeVar("T")

@struct.dataclass
class CtorSpec(Generic[T]):
    ctor: Callable[..., T]
    config: Dict[str, Any]

    @classmethod
    def from_name(cls, ctor_full_name: str, config: Dict[str, Any]):
        ctor_module = importlib.import_module(
            ".".join(ctor_full_name.split(".")[:-1])
        )
        ctor_name = ctor_full_name.split(".")[-1]
        ctor = getattr(ctor_module, ctor_name)
        return cls(ctor=ctor, config=config)

    def instantiate(self) -> T:
        return self.ctor(**self.config)

    def to_json(self) -> str:
        ctor_str = f"{self.ctor.__module__}.{self.ctor.__name__}"
        return json.dumps({"ctor": ctor_str, "config": self.config})

    @classmethod
    def from_json(cls, json_str: str) -> "CtorSpec":
        data = json.loads(json_str)
        return cls.from_name(data["ctor"], data["config"])


OptimizerSpec = CtorSpec[optax.GradientTransformation]
ModuleSpec = CtorSpec[nn.Module]
