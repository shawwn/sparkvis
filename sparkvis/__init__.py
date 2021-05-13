__version__ = '0.1.0'

import numpy as np
from sparklines import sparklines


def flatten(x):
    if isinstance(x, (list, tuple)):
        l = []
        for ele in x:
            l.extend(flatten(ele))
        return l
    else:
        return [x]


def flatlist(x):
    if not isinstance(x, (list, tuple)):
        x = [x]
    return flatten(x)


def to_numpy(x, concat_axis=-1) -> np.ndarray:
    x = flatlist(x)
    x = [v.numpy() if hasattr(v, 'numpy') else v for v in x]
    # reshape to 2D.
    x = [v.reshape([-1, v.shape[-1]]) for v in x]
    x = np.concatenate(x, axis=concat_axis)
    return x


class Sparkvis:
    value: np.ndarray

    def __init__(self, value):
        self.value = to_numpy(value)

    def min(self):
        return self.value.min()

    def max(self):
        return self.value.max()

    def get_lines(self):
        lo, hi = self.min(), self.max()
        value = self.value - self.min()
        return [sparklines(x, minimum=lo, maximum=hi) for x in value]

    def print(self, aspect_ratio=3):
        print('')
        for sparklines in self.get_lines():
            for sparkline in sparklines:
                final = ''.join([''.join([c] * aspect_ratio) for c in sparkline])
                print(final)
        h, w = list(self.value.shape)
        print(f'{h}x{w} min={self.min():.6} max={self.max():.6}')

    def __call__(self):
        self.print()


def sparkvis(*values, aspect_ratio=3) -> Sparkvis:
    v = Sparkvis(values)
    v.print(aspect_ratio=aspect_ratio)
    return v
