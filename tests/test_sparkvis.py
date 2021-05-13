import pytest
import sparkvis
from functools import partial

def test_version():
    assert sparkvis.__version__ == '0.1.0'


class API:
    def __getattr__(self, name: str):
        if name in self.__dict__:
            return self[name]
        else:
            return getattr(self.lib, name)


import numpy as np


class NumpyAPI(API):
    lib: np

    def __init__(self):
        super().__init__()
        self.lib = np

    def rand(self, *shape):
        return self.lib.random.rand(*shape)

    def cat(self, *values, dim=0):
        return self.lib.concatenate(sparkvis.flatlist(values), axis=dim)


try:
    import torch
except ImportError:
    torch = None


class TorchAPI(API):
    lib: torch

    def __init__(self):
        super().__init__()
        self.lib = torch


try:
    import tensorflow as tf2
    tf = tf2.compat.v1
except ImportError:
    tf2 = None


class TensorflowAPI(API):
    lib: tf

    def __init__(self):
        super().__init__()
        self.lib = tf

    def rand(self, *shape):
        return self.lib.random.uniform(shape)

    def cat(self, *values, dim=0):
        return self.lib.concat(sparkvis.flatlist(values), axis=dim)


class TestAPI:
    def __new__(cls, name):
        print(f'TestAPI({name!r})')
        if name not in "torch numpy tensorflow".split():
            raise ValueError("Unexpected name")
        if name == "torch" and torch is not None:
            return TorchAPI()
        if name == "tensorflow" and tf2 is not None:
            return TensorflowAPI()
        return NumpyAPI()


apis = [
    partial(TestAPI, "torch"),
    partial(TestAPI, "tensorflow"),
    partial(TestAPI, "numpy"),
]


@pytest.mark.parametrize("api", apis)
def test_sparkvis(api):
    lib = api()
    x = lib.rand(7, 7)
    v = sparkvis.Sparkvis(x)
    assert 0 <= v.min() <= 1
    assert 0 <= v.max() <= 1
    sparkvis.sparkvis(lib.ones_like(x), x, lib.zeros_like(x))
    print(x.shape)
    x = lib.cat([lib.ones_like(x), x, lib.zeros_like(x)], dim=-1)
    sparkvis.sparkvis(lib.ones_like(x), x, lib.zeros_like(x))
    print(x.shape)
    x = lib.rand(3,7,7)
    sparkvis.sparkvis(lib.ones_like(x), x, lib.zeros_like(x))
    print(x.shape)
    x = lib.rand(7,7, 3)
    sparkvis.sparkvis(lib.ones_like(x), x, lib.zeros_like(x))
    print(x.shape)

