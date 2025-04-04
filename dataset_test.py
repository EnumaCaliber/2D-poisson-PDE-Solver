from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import torch


class PoissonImageDataset(Dataset):
    def __init__(self, path, normalize=True):
        super().__init__()
        data = np.load(path)
        self.images = data["image"]  # shape: (N, H, W)
        self.normalize = normalize

        # 如果是灰度图，给它加个 channel 维度变成 (N, 1, H, W)
        if self.images.ndim == 3:
            self.images = self.images[:, None, :, :]  # -> (N, 1, H, W)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]
        img = torch.from_numpy(img).float()

        if self.normalize:
            img = (
                img * 2 - 1
            )  # normalize to [-1, 1] as required by most diffusion models

        return img


dataset = PoissonImageDataset("poisson_dataset_200.npz")
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 查看一张图
import matplotlib.pyplot as plt

img = next(iter(dataloader))[3].squeeze().numpy()  # (H, W)
plt.imshow((img + 1) / 2, cmap="jet")  # 还原到 [0, 1]
plt.title("u(x,y)")
plt.colorbar()
plt.show()
