import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch_lightning import Callback

matplotlib.use("Agg")


def save_figure_to_numpy(fig: plt.Figure) -> np.ndarray:
    """
    【升级版】将 Matplotlib figure 保存为 NumPy 数组。
    此版本使用现代 API 替代已弃用的 tostring_rgb()。

    Args:
        fig (Figure): Matplotlib figure 对象。

    Returns:
        ndarray: 代表图像的 NumPy 数组，形状为 (height, width, 3)，格式为 RGB。
    """
    # 1. 确保 figure 被绘制到了 canvas 上，这是获取缓冲区前的重要一步
    fig.canvas.draw()

    # 2. 从 canvas 获取 RGBA 格式的缓冲区
    # buffer_rgba() 返回一个 memoryview 对象，非常高效
    buf = fig.canvas.buffer_rgba()

    # 3. 将缓冲区转换为 NumPy 数组
    # 得到的数组形状是 (height, width, 4)，数据类型是 uint8
    X = np.asarray(buf)

    # 4. 从 RGBA 转换为 RGB
    # 我们只需要前3个通道 (R, G, B)，丢弃第4个通道 (Alpha)
    # [:, :, :3] 表示在所有高度和宽度上，只取前3个通道
    X_rgb = X[:, :, :3]

    # 5. 为了完全匹配旧代码的内存布局，可以创建一个副本
    # np.fromstring 总是创建一个新副本，而 np.asarray 可能不会
    # 这在大多数情况下不是必需的，但为了100%安全，可以加上
    return X_rgb.copy()


def plot_spectrogram_to_numpy(spectrogram: np.ndarray) -> np.ndarray:
    """
    Plot a spectrogram and convert it to a numpy array.

    Args:
        spectrogram (ndarray): Spectrogram data.

    Returns:
        ndarray: Numpy array representing the plotted spectrogram.
    """
    spectrogram = spectrogram.astype(np.float32)
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


class GradNormCallback(Callback):
    """
    Callback to log the gradient norm.
    """

    def on_after_backward(self, trainer, model):
        model.log("grad_norm", gradient_norm(model))


def gradient_norm(model: torch.nn.Module, norm_type: float = 2.0) -> torch.Tensor:
    """
    Compute the gradient norm.

    Args:
        model (Module): PyTorch model.
        norm_type (float, optional): Type of the norm. Defaults to 2.0.

    Returns:
        Tensor: Gradient norm.
    """
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type) for g in grads]), norm_type)
    return total_norm
