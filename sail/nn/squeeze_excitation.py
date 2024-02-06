import paddle
from typing import Optional


def max(*args, **kwargs):
    if "input" in kwargs:
        kwargs["x"] = kwargs.pop("input")

    out_v = None
    if "out" in kwargs:
        out_v = kwargs.pop("out")

    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.maximum(*args, **kwargs)
    elif len(args) == 2 and isinstance(args[1], paddle.Tensor):
        ret = paddle.maximum(*args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 2:
            if out_v:
                ret = paddle.max(*args, **kwargs), paddle.argmax(*args, **kwargs)
                paddle.assign(ret[0], out_v[0])
                paddle.assign(ret[1], out_v[1])
                return out_v
            else:
                ret = paddle.max(*args, **kwargs), paddle.argmax(*args, **kwargs)
                return ret
            return out_v
        else:
            ret = paddle.max(*args, **kwargs)
            return ret

    if out_v:
        paddle.assign(ret, out_v)
        return out_v
    else:
        return ret


class SqueezeExcitation(paddle.nn.Layer):
    """
    Generic 2d/3d extension of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    Squeezing spatially and exciting channel-wise
    """

    block: paddle.nn.Layer
    is_3d: bool

    def __init__(
        self,
        num_channels: int,
        num_channels_reduced: Optional[int] = None,
        reduction_ratio: float = 2.0,
        is_3d: bool = False,
        activation: Optional[paddle.nn.Layer] = None,
    ) -> None:
        """
        Args:
            num_channels (int): Number of input channels.
            num_channels_reduced (int):
                Number of reduced channels. If none, uses reduction_ratio to calculate.
            reduction_ratio (float):
                How much num_channels should be reduced if num_channels_reduced is not provided.
            is_3d (bool): Whether we're operating on 3d data (or 2d), default 2d.
            activation (nn.Module): Activation function used, defaults to ReLU.
        """
        super().__init__()
        if num_channels_reduced is None:
            num_channels_reduced = int(num_channels // reduction_ratio)
        if activation is None:
            activation = paddle.nn.ReLU()
        if is_3d:
            conv1 = paddle.nn.Conv3D(
                in_channels=num_channels,
                out_channels=num_channels_reduced,
                kernel_size=1,
                bias_attr=True,
            )
            conv2 = paddle.nn.Conv3D(
                in_channels=num_channels_reduced,
                out_channels=num_channels,
                kernel_size=1,
                bias_attr=True,
            )
        else:
            conv1 = paddle.nn.Conv2D(
                in_channels=num_channels,
                out_channels=num_channels_reduced,
                kernel_size=1,
                bias_attr=True,
            )
            conv2 = paddle.nn.Conv2D(
                in_channels=num_channels_reduced,
                out_channels=num_channels,
                kernel_size=1,
                bias_attr=True,
            )
        self.is_3d = is_3d
        self.block = paddle.nn.Sequential(conv1, activation, conv2, paddle.nn.Sigmoid())

    def forward(self, input_tensor: paddle.Tensor) -> paddle.Tensor:
        """
        Args:
            input_tensor: X, shape = (batch_size, num_channels, H, W).
                For 3d X, shape = (batch_size, num_channels, T, H, W).
            output tensor
        """
        mean_tensor = (
            input_tensor.mean(axis=[2, 3, 4], keepdim=True)
            if self.is_3d
            else input_tensor.mean(axis=[2, 3], keepdim=True)
        )
        output_tensor = paddle.multiply(
            x=input_tensor, y=paddle.to_tensor(self.block(mean_tensor))
        )
        return output_tensor


class SpatialSqueezeExcitation(paddle.nn.Layer):
    """
    Generic 2d/3d extension of SE block
        squeezing channel-wise and exciting spatially described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation
        in Fully Convolutional Networks, MICCAI 2018*
    """

    block: paddle.nn.Layer

    def __init__(self, num_channels: int, is_3d: bool = False) -> None:
        """
        Args:
            num_channels (int): Number of input channels.
            is_3d (bool): Whether we're operating on 3d data.
        """
        super().__init__()
        if is_3d:
            conv = paddle.nn.Conv3D(
                in_channels=num_channels, out_channels=1, kernel_size=1, bias_attr=True
            )
        else:
            conv = paddle.nn.Conv2D(
                in_channels=num_channels, out_channels=1, kernel_size=1, bias_attr=True
            )
        self.block = paddle.nn.Sequential(conv, paddle.nn.Sigmoid())

    def forward(self, input_tensor: paddle.Tensor) -> paddle.Tensor:
        """
        Args:
            input_tensor: X, shape = (batch_size, num_channels, H, W).
                For 3d X, shape = (batch_size, num_channels, T, H, W).
            output tensor
        """
        output_tensor = paddle.multiply(
            x=input_tensor, y=paddle.to_tensor(self.block(input_tensor))
        )
        return output_tensor


class ChannelSpatialSqueezeExcitation(paddle.nn.Layer):
    """
    Generic 2d/3d extension of concurrent spatial and channel squeeze & excitation:
         *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation
         in Fully Convolutional Networks, arXiv:1803.02579*
    """

    def __init__(
        self,
        num_channels: int,
        num_channels_reduced: Optional[int] = None,
        reduction_ratio: float = 16.0,
        is_3d: bool = False,
        activation: Optional[paddle.nn.Layer] = None,
    ) -> None:
        """
        Args:
            num_channels (int): Number of input channels.
            num_channels_reduced (int):
                Number of reduced channels. If none, uses reduction_ratio to calculate.
            reduction_ratio (float):
                How much num_channels should be reduced if num_channels_reduced is not provided.
            is_3d (bool): Whether we're operating on 3d data (or 2d), default 2d.
            activation (nn.Module): Activation function used, defaults to ReLU.
        """
        super().__init__()
        self.channel = SqueezeExcitation(
            num_channels=num_channels,
            num_channels_reduced=num_channels_reduced,
            reduction_ratio=reduction_ratio,
            is_3d=is_3d,
            activation=activation,
        )
        self.spatial = SpatialSqueezeExcitation(num_channels=num_channels, is_3d=is_3d)

    def forward(self, input_tensor: paddle.Tensor) -> paddle.Tensor:
        """
        Args:
            input_tensor: X, shape = (batch_size, num_channels, H, W)
                For 3d X, shape = (batch_size, num_channels, T, H, W)
            output tensor
        """
        output_tensor = max(self.channel(input_tensor), self.spatial(input_tensor))
        return output_tensor
