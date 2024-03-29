# bonn.randopt
Bayesian Optimization for [Randopt](http://seba-1511.github.io/randopt) with

* Gaussian Processes,
* Tree of Parzen Estimators, and
* Neural Networks.

Here's a simple demo that exposes the universal API (just swap `Bonn` for `Bogp` or `Botpe`)

~~~python
#!/usr/bin/env python

import randopt as ro
from bonn import Bonn

def loss(x, y, z):
    return x**2 + y**2 + z**2

if __name__ == '__main__':
    e = ro.Experiment('bo_simple', {
        'x': ro.Choice([0.0, 1, 2, 3, 4, 5, 6, 7]),
        'y': ro.Gaussian(0.0, 3.0),
        'z': ro.Uniform(0.0, 1.0),
        })

    bo = Bonn(e)

    e.sample_all_params()
    res = loss(e.x, e.y, e.z)
    e.add_result(res)

    for i in xrange(200):
        bo.fit()
        bo.sample(e)
        res = loss(e.x, e.y, e.z)
        print res
        e.add_result(res, {
            'trial': i
            })
~~~
