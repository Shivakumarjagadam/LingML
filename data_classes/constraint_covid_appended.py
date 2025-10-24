import pandas as pd

from .base import BaseDataset


class ConstraintAppended(BaseDataset):

    def __init__(
        self,
        root: str,
        **kwargs,
    ):

        self.datasets = dict()
        for split in ('train', 'val', 'test'):
            fn = f'{root}/{split}.csv'
            try:
                df = pd.read_csv(fn, index_col=0).drop(['Segment',], axis=1)
                df.label = df.label.replace({'real': 1, 'fake': 0})
                self.datasets[split] = df
            except:
                print(f'file {fn} not found.')