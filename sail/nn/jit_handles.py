import typing
from collections import Counter, OrderedDict
from numbers import Number
from typing import Any, Callable, List, Optional, Union
import numpy as np

try:
    from math import prod
except ImportError:
    from numpy import prod
Handle = Callable[[List[Any], List[Any]], Union[typing.Counter[str], Number]]


def get_shape(val: Any) -> Optional[List[int]]:
    """
    Get the shapes from a jit value object.

    Args:
        val (torch._C.Value): jit value object.

    Returns:
        list(int): return a list of ints.
    """
    if val.isCompleteTensor():
        return str(val.dtype).sizes()
    else:
        return None


"""
Below are flop/activation counters for various ops. Every counter has the following signature:

Args:
    inputs (list(torch._C.Value)): The inputs of the op in the form of a list of jit object.
    outputs (list(torch._C.Value)): The outputs of the op in the form of a list of jit object.

Returns:
    number: The number of flops/activations for the operation.
    or Counter[str]
"""


def generic_activation_jit(op_name: Optional[str] = None) -> Handle:
    """
    This method return a handle that counts the number of activation from the
    output shape for the specified operation.

    Args:
        op_name (str): The name of the operation. If given, the handle will
            return a counter using this name.

    Returns:
        Callable: An activation handle for the given operation.
    """

    def _generic_activation_jit(
        i: Any, outputs: List[Any]
    ) -> Union[typing.Counter[str], Number]:
        """
        This is a generic jit handle that counts the number of activations for any
        operation given the output shape.
        """
        out_shape = get_shape(outputs[0])
        ac_count = prod(out_shape)
        if op_name is None:
            return ac_count
        else:
            return Counter({op_name: ac_count})

    return _generic_activation_jit


def addmm_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for fully connected layers.
    """
    input_shapes = [get_shape(v) for v in inputs[1:3]]
    assert len(input_shapes[0]) == 2, input_shapes[0]
    assert len(input_shapes[1]) == 2, input_shapes[1]
    batch_size, input_dim = input_shapes[0]
    output_dim = input_shapes[1][1]
    flops = batch_size * input_dim * output_dim
    return flops


def linear_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for the aten::linear operator.
    """
    input_shapes = [get_shape(v) for v in inputs[0:2]]
    assert input_shapes[0][-1] == input_shapes[1][-1]
    flops = prod(input_shapes[0]) * input_shapes[1][0]
    return flops


def bmm_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for the bmm operation.
    """
    assert len(inputs) == 2, len(inputs)
    input_shapes = [get_shape(v) for v in inputs]
    n, c, t = input_shapes[0]
    d = input_shapes[-1][-1]
    flop = n * c * t * d
    return flop


def conv_flop_count(
    x_shape: List[int],
    w_shape: List[int],
    out_shape: List[int],
    transposed: bool = False,
) -> Number:
    """
    Count flops for convolution. Note only multiplication is
    counted. Computation for addition and bias is ignored.

    Flops for a transposed convolution are calculated as
    flops = (x_shape[2:] * prod(w_shape) * batch_size).

    Args:
        x_shape (list(int)): The input shape before convolution.
        w_shape (list(int)): The filter shape.
        out_shape (list(int)): The output shape after convolution.
        transposed (bool): is the convolution transposed
    Returns:
        int: the number of flops
    """
    batch_size = x_shape[0]
    conv_shape = (x_shape if transposed else out_shape)[2:]
    flop = batch_size * prod(w_shape) * prod(conv_shape)
    return flop


def conv_flop_jit(inputs: List[Any], outputs: List[Any]) -> typing.Counter[str]:
    """
    Count flops for convolution.
    """
    assert len(inputs) == 12 or len(inputs) == 13, len(inputs)
    x, w = inputs[:2]
    x_shape, w_shape, out_shape = get_shape(x), get_shape(w), get_shape(outputs[0])
    transposed = inputs[6].toIValue()
    return Counter(
        {"conv": conv_flop_count(x_shape, w_shape, out_shape, transposed=transposed)}
    )


def einsum_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for the einsum operation.
    """
    assert len(inputs) >= 2, len(inputs)
    equation = inputs[0].toIValue()
    equation = equation.replace(" ", "")
    input_shapes_jit = inputs[1].node().inputs()
    input_shapes = [get_shape(v) for v in input_shapes_jit]
    letter_order = OrderedDict((k, 0) for k in equation if k.isalpha()).keys()
    mapping = {ord(x): (97 + i) for i, x in enumerate(letter_order)}
    equation = equation.translate(mapping)
    if equation == "abc,abd->acd":
        n, c, t = input_shapes[0]
        p = input_shapes[-1][-1]
        flop = n * c * t * p
        return flop
    elif equation == "abc,adc->adb":
        n, t, g = input_shapes[0]
        c = input_shapes[-1][1]
        flop = n * t * g * c
        return flop
    else:
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
        raise NotImplementedError("Unsupported einsum operation.")


def matmul_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for matmul.
    """
    input_shapes = [get_shape(v) for v in inputs]
    assert len(input_shapes) == 2, input_shapes
    assert input_shapes[0][-1] == input_shapes[1][-2], input_shapes
    flop = prod(input_shapes[0]) * input_shapes[-1][-1]
    return flop


def norm_flop_counter(affine_arg_index: int) -> Handle:
    """
    Args:
        affine_arg_index: index of the affine argument in inputs
    """

    def norm_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
        """
        Count flops for norm layers.
        """
        input_shape = get_shape(inputs[0])
        has_affine = get_shape(inputs[affine_arg_index]) is not None
        assert 2 <= len(input_shape) <= 5, input_shape
        flop = prod(input_shape) * (5 if has_affine else 4)
        return flop

    return norm_flop_jit


def batchnorm_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    training = inputs[5].toIValue()
    assert isinstance(training, bool), "Signature of aten::batch_norm has changed!"
    if training:
        return norm_flop_counter(1)(inputs, outputs)
    has_affine = get_shape(inputs[1]) is not None
    input_shape = prod(get_shape(inputs[0]))
    return input_shape * (2 if has_affine else 1)


def elementwise_flop_counter(input_scale: float = 1, output_scale: float = 0) -> Handle:
    """
    Count flops by
        input_tensor.numel() * input_scale + output_tensor.numel() * output_scale

    Args:
        input_scale: scale of the input tensor (first argument)
        output_scale: scale of the output tensor (first element in outputs)
    """

    def elementwise_flop(inputs: List[Any], outputs: List[Any]) -> Number:
        ret = 0
        if input_scale != 0:
            shape = get_shape(inputs[0])
            ret += input_scale * prod(shape)
        if output_scale != 0:
            shape = get_shape(outputs[0])
            ret += output_scale * prod(shape)
        return ret

    return elementwise_flop
