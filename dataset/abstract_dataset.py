import torch.utils.data as data
import torch


class AbstractDataset(data.Dataset):
    '''
    Abstract Base Class for Dataset. All Dataset class will inherit from this class.
    '''
    def __init__(self, data_frame, transform=None) -> None:
        '''
        Args:
            data_frame(pandas.DataFrame)
            transform(Callable)
        '''
        super().__init__()
        self.data_frame = data_frame
        self.transform = transform
        self._keys = list(self.data_frame.keys())

    def __len__(self):
        return len(self.data_frame)

    @property
    def keys(self):
        return self._keys

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist(index)

        info_dict = self.data_frame.loc[index, :]
        item_dict = {'index': index}

        for key in self.keys:
            item_dict[key] = info_dict[key]

        return item_dict
