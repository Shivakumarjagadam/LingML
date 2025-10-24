from typing import Union
from yacs.config import CfgNode as CN


def default_cfg():

    _C = CN()

    _C.DEVICE_INDEX = None
    _C.SPLITS = None
    _C.ADD_NEW_TOKENS = None
    _C.LR = None

    _C.DATA = CN()
    _C.DATA.ADD_DENSE = None
    _C.DATA.BATCH_SIZE = None
    _C.DATA.N_EPOCHS = None
    _C.DATA.TEST_EVERY = None
    _C.DATA.SAVE_EVERY = None

    _C.DATA.args = CN()
    _C.DATA.args.root = None
    _C.DATA.args.train_pct = None
    _C.DATA.args.valid_pct = None

    _C.MODEL = CN()
    _C.MODEL.HF_PATH = None
    _C.MODEL.MAX_LENGTH = None
    _C.MODEL.SAVE_NAME = None

    return _C.clone()


class Config:
    
    def __init__(self, root: str, dataset: str, model: str, override: Union[list, None] = None):

        self.cfg = default_cfg()
        self.cfg.merge_from_file(f'{root}/config.yaml')
        self.cfg.DATA.merge_from_file(f'{root}/datasets/{dataset}.yaml')
        self.cfg.MODEL.merge_from_file(f'{root}/models/{model}.yaml')

        if isinstance(override, list):
            self.cfg.merge_from_list(override)

    def __getattr__(self, name: str):

        return self.cfg.__getattr__(name)