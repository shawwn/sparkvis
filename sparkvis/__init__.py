__version__ = '0.3.4'

import builtins
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

    def __init__(self, value, *, aspect_ratio=3):
        self.value = to_numpy(value)
        self.aspect_ratio = aspect_ratio

    def min(self):
        return self.value.min()

    def max(self):
        return self.value.max()

    def get_lines(self):
        value = self.value - self.min()
        lo, hi = value.min(), value.max()
        return [sparklines(x, minimum=lo, maximum=hi) for x in value]

    def to_string(self):
        lines = list()
        lines.append('')
        for sparklines in self.get_lines():
            for sparkline in sparklines:
                final = ''.join([''.join([c] * self.aspect_ratio) for c in sparkline])
                lines.append(final)
        h, w = list(self.value.shape)
        lines.append('{h}x{w} min={min:.6} max={max:.6}'.format(
            h=h, w=w, min=self.min(), max=self.max(),
        ))
        return '\n'.join(lines)

    def print(self, **print_kwargs):
        output = self.to_string()
        print(output, **print_kwargs)


    def __call__(self):
        self.print()


def sparkvis(*values, aspect_ratio=3, print=True, file=None) -> Sparkvis:
    v = Sparkvis(values, aspect_ratio=aspect_ratio)
    if print:
        builtins.print(v.to_string(), file=file)
    return v
