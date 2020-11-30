"""Initializer of parameters."""
import numpy as np

import tvm
from tvm import relay

class Initializer(object):
    """The base class of an initializer."""
    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def __call__(self, desc, arr):
        """Initialize an array
        Parameters
        ----------
        desc : str
            Initialization pattern descriptor.
        arr : NDArray
            The array to be initialized.
        """
        if desc.endswith('weight'):
            self._init_weight(desc, arr)
        elif desc.endswith('bias'):
            self._init_bias(desc, arr)
        elif desc.endswith('gamma'):
            self._init_gamma(desc, arr)
        elif desc.endswith('beta'):
            self._init_beta(desc, arr)
        elif desc.endswith('mean'):
            self._init_mean(desc, arr)
        elif desc.endswith('var'):
            self._init_var(desc, arr)
        elif desc.endswith('scale'):
            self._init_scale(desc, arr)
        elif desc.endswith('shift'):
            self._init_shift(desc, arr)
        else:
            self._init_default(desc, arr)

    def _init_bias(self, _, arr):
        arr[:] = 0.0

    def _init_gamma(self, _, arr):
        arr[:] = 1.0

    def _init_beta(self, _, arr):
        arr[:] = 0.0

    def _init_mean(self, _, arr):
        arr[:] = 0.0

    def _init_var(self, _, arr):
        arr[:] = 1.0

    def _init_scale(self, _, arr):
        arr[:] = 2

    def _init_shift(self, _, arr):
        arr[:] = 2

    def _init_weight(self, name, arr):
        """Abstract method to Initialize weight."""
        raise NotImplementedError("Must override it")

    def _init_default(self, name, _):
        raise ValueError(
            'Unknown initialization pattern for %s. ' \
            'Default initialization is now limited to '\
            '"weight", "bias", "gamma" (1.0), and "beta" (0.0).' \
            'Please use mx.sym.Variable(init=mx.init.*) to set initialization pattern' % name)

class Xavier(Initializer):
    """ "Xavier" initialization for weights
    Parameters
    ----------
    rnd_type: str, optional
        Random generator type, can be ``'gaussian'`` or ``'uniform'``.
    factor_type: str, optional
        Can be ``'avg'``, ``'in'``, or ``'out'``.
    magnitude: float, optional
        Scale of random number.
    """
    def __init__(self, rnd_type="uniform", factor_type="avg", magnitude=3):
        super(Xavier, self).__init__(rnd_type=rnd_type,
                                     factor_type=factor_type,
                                     magnitude=magnitude)
        self.rnd_type = rnd_type
        self.factor_type = factor_type
        self.magnitude = float(magnitude)

    def _init_weight(self, name, arr):
        shape = arr.shape
        hw_scale = 1.
        if len(shape) < 2:
            raise ValueError('Xavier initializer cannot be applied to vector {0}. It requires at'
                             ' least 2D.'.format(name))
        if len(shape) > 2:
            hw_scale = np.prod(shape[2:])
        fan_in, fan_out = shape[1] * hw_scale, shape[0] * hw_scale
        factor = 1.
        if self.factor_type == "avg":
            factor = (fan_in + fan_out) / 2.0
        elif self.factor_type == "in":
            factor = fan_in
        elif self.factor_type == "out":
            factor = fan_out
        else:
            raise ValueError("Incorrect factor type")
        # Hack for mobilenet, because there is less connectivity
        if "depthwise" in name:
            factor = hw_scale
        scale = np.sqrt(self.magnitude / factor)
        if self.rnd_type == "uniform":
            arr[:] = np.random.uniform(-scale, scale, size=arr.shape)
        else:
            raise ValueError("Unknown random type")

class QuantizeInitializer(Initializer):
    def _init_weight(self, name, arr):
        if arr.dtype == np.float32:
            arr[:] = np.random.uniform(-1., 1., size=arr.shape)
        elif arr.dtype == np.int8:
            arr[:] = np.random.randint(-127, 128, size=arr.shape)
        elif arr.dtype == np.uint8:
            arr[:] = np.random.randint(0, 256, size=arr.shape)
        elif arr.dtype == np.int32:
            arr[:] = np.random.randint(-2**31, 2**31, size=arr.shape)
        else:
            raise ValueError("Unknown random type %s" % (arr.dtype))

    def _init_bias(self, name, arr):
        if arr.dtype == np.int32:
            arr[:] = np.random.randint(-200, 200, size=arr.shape)
        else:
            raise ValueError("Unknown random type %s" % (arr.dtype))

    def _init_scale(self, _, arr):
        arr[:] = np.random.randint(-256, 256, size=arr.shape)

    def _init_shift(self, _, arr):
        arr[:] = np.random.randint(-256, 256, size=arr.shape)

def create_workload(net, initializer=None, seed=0):
    """Helper function to create benchmark image classification workload.
    Parameters
    ----------
    net : tvm.relay.Function
        The selected function of the network.
    initializer : Initializer
        The initializer used
    seed : int
        The seed used in initialization.
    Returns
    -------
    mod : tvm.IRModule
        The created relay module.
    params : dict of str to NDArray
        The parameters.
    """
    mod = tvm.IRModule.from_expr(net)
    mod = relay.transform.InferType()(mod)
    shape_dict = {
        v.name_hint : v.checked_type for v in mod["main"].params}
    np.random.seed(seed)
    initializer = initializer if initializer else Xavier()
    params = {}
    for k, v in shape_dict.items():
        if k == "data":
            continue

        if v.dtype == 'int4' or v.dtype == 'uint4':
            pack_shape = list(v.concrete_shape)
            pack_shape[-1] = pack_shape[-1] // 8
            init_value = np.zeros(pack_shape).astype('int32')
        else:
            init_value = np.zeros(v.concrete_shape).astype(v.dtype)

        initializer(k, init_value)
        params[k] = tvm.nd.array(init_value, ctx=tvm.cpu(0))

    return mod, params