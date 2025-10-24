from typing import Callable, Union, Iterable

from numpy import vectorize


class BaseDataset:

    def __init__(
        self
    ):

        pass

    def __getitem__(
        self,
        key
    ):

        return self.datasets.get(key, None)
    
    def __setitem__(
        self, 
        key, 
        value
    ):
        
        self.datasets[key] = value
    
    def __iter__(
        self
    ):

        return iter(self.datasets)

    def apply(
        self, 
        func: Callable, 
        splits: Union[str, Iterable] = 'all', 
        col: str = 'tweet', 
        new_col: str = None
    ):

        vec_func = vectorize(func)
        if isinstance(splits, str):
            if splits == 'all':
                splits = ('train', 'val', 'test')
            elif splits in self.datasets:
                splits = (split,)
            else:
                print(f"split '{splits}' unidentified.")
                return

        if not isinstance(new_col, str):
            new_col = f'{func.__name__}_{col}'

        for split in set(splits).intersection(self.datasets.keys()):
            df = self.datasets[split]
            df[new_col] = vec_func(df[col].values)