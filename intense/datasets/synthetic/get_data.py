from torch.utils.data import Dataset, random_split, DataLoader
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import torch
import numpy as np

from intense.utils.util import get_split_lengths


def get_dataloader(dataset: Dataset, batch_size: int, num_workers=0,
                   val_split: float = 0.25, seed=42,
                   pin_memory=True) -> list[DataLoader]:

    split_lengths = get_split_lengths([val_split], len(dataset))
    train_set, val_set = random_split(
        dataset,
        split_lengths,
        generator=torch.Generator().manual_seed(seed)
        )
    print(f'Number of datapoints in the train set : {len(train_set)}')
    print(f'Number of datapoints in the validation set: {len(val_set)}')
    print(f'Data Modalities in dataset: {dataset.modalities}')
    train = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=pin_memory)
    valid = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=pin_memory)
    test = valid
    return train, valid, test


class MutltiModalCharDataset(Dataset):
    vocabulary: list = ['A', 'C', 'G', 'T']
    encoder = OneHotEncoder(categories=[vocabulary])

    def __init__(self,
                 data_csv_path: str,
                 transform=None) -> None:
        self.data_csv_path = data_csv_path
        self.transform = transform
        self.data = self.make_dataset_df(self.data_csv_path)

    @property
    def modalities(self) -> list:
        mods = list(self.data.columns)
        mods.remove('label')
        return mods

    @modalities.setter
    def modalities(self, mods: list):
        if len(self.data.colums) - 1 != len(mods):
            raise ValueError("Number of modalities in the list doesnt match"
            "the number of modalities present in the dataset file")
        self.modalities = mods

    def __getitem__(self, index: int):
        data_list, target = self._get_sample(index)
        data_list = [ torch.from_numpy(self._one_hot_encode(value)).float() 
                        for value in data_list]
        if self.transform is not None:
            data_list, target = self.transform(data_list, target)
        #TODO:return a datastructure instead of a list, make a custom datasrtucture
        return data_list, target
        
    def __len__(self):
        return len(self.data)

    def _get_sample(self,index):
        row = self.data.iloc[index]
        target = row['label']
        #TODO:Make this work when modality names different the column in csv file
        #data_dict = {modality: row[modality] for modality in self.modalities}
        data_list = [row[modality] for modality in self.modalities]
        return data_list, target

    def _one_hot_encode(self, sequence):
        seq_arr = np.array(list(sequence)).reshape(-1, 1)
        return self.__class__.encoder.fit_transform(seq_arr).toarray().T

    @staticmethod
    def make_dataset_df(data_csv_path):
        df = pd.read_csv(data_csv_path,  index_col=0)
        df.loc[df['label']==-1, 'label']=0
        df = df.rename(columns=lambda col: col.replace('.','_'))
        return df
