# sparkvis

See [here](https://twitter.com/theshawwn/status/1392730448682524672) for usage instructions. Maybe someday I'll write a real readme, but not at 2am on a Thursday.


Ok fine, I'll write a readme. This is a library for visualizing tensors in a plain Python REPL using sparklines. I was sick of having to install jupyter on servers just to see a damn tensor.

E.g. the FFT of MNIST looks like this:

![](https://pbs.twimg.com/media/E1P4TC3WEAApxDU?format=jpg&name=large)


## Quickstart


```
pip3 install -U sparkvis
python3
from sparkvis import sparkvis as vis
vis(foo)
```

`foo` can be a torch tensor, tf tensor, numpy array, etc. It supports anything with a .numpy() method.

`vis(a, b)` will put 'a' and 'b' side by side. For example,

```py
import numpy as np
from sparkvis import sparkvis as vis
x = np.random.rand(7,7)
vis(x, np.zeros_like(x), np.ones_like(x))
```

will print this:

```
▅▅▅▄▄▄▂▂▂▅▅▅▄▄▄▅▅▅▅▅▅▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█████████████████████
▄▄▄▃▃▃▃▃▃▆▆▆▁▁▁▃▃▃███▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█████████████████████
▆▆▆▇▇▇▆▆▆▂▂▂▇▇▇▅▅▅▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█████████████████████
███▇▇▇▃▃▃▇▇▇▄▄▄▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█████████████████████
▆▆▆▅▅▅▇▇▇▅▅▅███▆▆▆▄▄▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█████████████████████
▂▂▂▇▇▇▇▇▇▆▆▆▆▆▆▁▁▁▃▃▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█████████████████████
▅▅▅▇▇▇▆▆▆▅▅▅▅▅▅▁▁▁▇▇▇▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█████████████████████
7x21 min=0.0 max=1.0
<sparkvis.Sparkvis object at 0x10230f310>
```

You can pass `to_string=True` if you want the string instead of
printing to stdout. Or you can pass `file=f` like the normal python
`print` function.

### Note on Tensorflow in Graph mode

Currently this library only supports Tensorflow in eager mode, since those are the only tensors that have a .numpy() method. Graph-based tensorflow tensors use .eval() rather than .numpy(). (Sorry, I'll get around to it sometime, otherwise PRs welcome.)


## License

MIT.
