import glob
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_directory, return_directory=False):
        self.data_directory = data_directory
        self.files = []
        for directory in self.data_directory:
            self.files.extend(glob.glob(directory + "*.pt"))
        self.size = len([name for name in self.files])
        self.loaded = {}
        self.return_directory = return_directory

    def __len__(self):
        return self.size

    def __getitem__(self, idx, scale=False):
        items = []
        single = False
        if not isinstance(idx, list):
            idx = [idx]
            single = True
        for idx_single in idx:
            if idx_single not in self.loaded.keys():
                content = torch.load(self.files[idx_single], weights_only=False)
                self.loaded[idx_single] = content
            # spectrogram = self.loaded[idx_single]
            # spectrogram[:45, :] = 0
            # items.append(spectrogram)
            items.append(self.loaded[idx_single].tolist())
        
        if single:
            if self.return_directory:
                return torch.tensor(items[0]).float(), self.files[idx[0]]
            else:
                return torch.tensor(items[0]).float()
        if self.return_directory:
            return torch.tensor(items).float(), self.files[idx[0]]
        return torch.tensor(items).float()

