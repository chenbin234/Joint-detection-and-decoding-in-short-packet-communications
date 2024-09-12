# This file is to build the pytorch dataset for the channel model

import torch
from torch.utils.data import Dataset
from pathlib import Path


class InfobitDataset(Dataset):
    """
    A PyTorch dataset for information bits of length k.

    This dataset loads the information bits (which are randomly generated)
    and returns them as a PyTorch tensor.

    Attributes:
        root (Path): The root directory of the dataset.
        device (torch.device): The device on which tensors will be loaded.
        data (torch.Tensor): The tensor containing the information bits.


    Raises:
        FileNotFoundError: If the specified root directory does not exist.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns the sample at the specified index.

    """

    def __init__(self, root):

        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"The {root} directory does not exist.")

        self.root = Path(root)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the information bits of size (num_samples, 1, k)
        self.data = torch.load(self.root, map_location=self.device)

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """

        return self.data.shape[0]

    def __getitem__(self, idx):
        """
        Retrieve a specific item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            torch.tensor: The information bits at the specified index.
        """

        return self.data[idx, :, :].to(self.device)


if __name__ == "__main__":

    dataset = InfobitDataset("data/processed/train_dataset_info_bits_2000_1_64.pt")
    print(len(dataset))
    print(dataset[0].shape)
    print(dataset[0])

    print("done")
