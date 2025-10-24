# usage:

# python3 -B inference.py \
#    --model covid-twitter-bert-v2 \
#    --dataset aaai-constraint-covid \
#    --weights "./results/aaai-constraint-covid/CT-BERT/2023-12-19-00-28-08/ckpt5350.pth"

# python3 -B inference.py \
#    --model covid-twitter-bert-v2 \
#    --dataset aaai-constraint-covid-appended \
#    --weights "./results/aaai-constraint-covid-appended/CT-BERT/2023-12-19-02-26-31/ckpt5350.pth"


import argparse
import os
import warnings; warnings.filterwarnings('ignore')

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from data_classes import dataclass_map 
from data_classes.preprocess import PreProcessor
from model_classes import modelclass_map
from utils.config import Config
from utils.logger import Logger
from utils.metrics import Results
from utils.tokenizer import tokenize
from utils.data_loaders import data_loaders
from utils.test import test_epoch


parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset', type=str, 
    help='dataset directory name'
)
parser.add_argument(
    '--model', type=str, 
    help='hugging face model name'
)
parser.add_argument(
    '--weights', type=str, 
    help='path to learnt model weights'
)
parser.add_argument(
    'opts', default=None, nargs=argparse.REMAINDER, 
    help='modify config options using the command-line'
)
args = parser.parse_args()

cfg = Config(
    root='config', 
    dataset=args.dataset, 
    model=args.model,
    override=args.opts,
)

Dataset = dataclass_map(args.dataset)
preprocessor = PreProcessor()
Model = modelclass_map(args.model)
DEVICE = torch.device(f'cuda:{cfg.DEVICE_INDEX}' if torch.cuda.is_available() and cfg.DEVICE_INDEX is not None else 'cpu')

logger = Logger(
    dataset=args.dataset,
    model=cfg.MODEL.SAVE_NAME,
    add_new_tokens=cfg.ADD_NEW_TOKENS
)

logger.log(f'DATASET = {os.path.basename(cfg.DATA.args.root)}', date=False)
logger.log(f"COVID WORDS = {', '.join((w for w in preprocessor.COVID_WORDS if not w.startswith('#')))}", date=False)
logger.log(f"TOKENS = {', '.join(preprocessor.TOKENS)}", date=False)
logger.log(f'ADD NEW TOKENS = {cfg.ADD_NEW_TOKENS}', date=False)
logger.log(f'MAX LENGTH = {cfg.MODEL.MAX_LENGTH}', date=False)
logger.log(f'BATCH SIZE = {cfg.DATA.BATCH_SIZE}', date=False)
logger.log(f'MODEL = {os.path.basename(cfg.MODEL.HF_PATH)}', date=False)


logger.log('Loading and pre-processing dataset...')
datasets = Dataset(
    root=cfg.DATA.args.root,
    train_pct=cfg.DATA.args.train_pct,
    valid_pct=cfg.DATA.args.valid_pct,
)
datasets.apply(
    func=preprocessor.preprocess,
    splits=['test'],
    col='tweet',
    new_col='tweet'
)
logger.log('Finished pre-processing dataset.\n')


logger.log('Tokenizing dataset...')
tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.HF_PATH)
if cfg.ADD_NEW_TOKENS:
    tokenizer.add_tokens(preprocessor.TOKENS)
encoded_inputs, dense_features, binarized_labels = tokenize(
    tokenizer=tokenizer,
    dataset=datasets,
    splits=['test'],
    max_length=cfg.MODEL.MAX_LENGTH,
    add_dense=cfg.DATA.ADD_DENSE
)
logger.log('Finished tokenizing dataset.\n')


logger.log('Preparing data-loader...')
test_loader, = data_loaders(
    splits=['test'],
    encoded_input=encoded_inputs,
    dense_features=dense_features,
    binarized_labels=binarized_labels,
    batch_size=cfg.DATA.BATCH_SIZE,
    shuffle=False
)
logger.log('Finished preparing data-loader.\n')


logger.log('Loading and preparing model...')
base_model = AutoModelForSequenceClassification.from_pretrained(cfg.MODEL.HF_PATH)
model = Model(
    base_model,
    emb_table_size=len(tokenizer),
    dense_size=dense_features['test'].size(1) if dense_features['test'] is not None else None,
    device=DEVICE
)
if os.path.exists(args.weights):
    if not args.weights.endswith('.pth'):
        logger.log(f'File {args.weights} is not a PyTorch state dictionary (must have a .pth extension). Continuing with downloaded weights.', print_text=True)
    else:
        model.base_model.load_state_dict(torch.load(args.weights))
else:
    logger.log(f'File {args.weights} does not exist. Continuing with downloaded weights.', print_text=True)
logger.log('Finshed preparing model.\n')


results = Results()

logger.log('Starting inference...')
labels, preds, loss = test_epoch(model, test_loader, DEVICE)
logger.log('Finished inference.')

results.update('testing', 0, labels, preds, loss)
accuracy, f1_score, loss = results.metrics('testing', last=1)
logger.log(f'Testing (total {labels.shape[0]} samples): accuracy = {accuracy:.6f}, f1-score = {f1_score:.6f}, loss = {loss:.6f}', print_text=True)

df = pd.DataFrame({'Labels': labels.numpy().astype(int), 'Predictions': preds.numpy().astype(int)})
df.index = df.index.rename('Sample Index')
df.to_csv(f'{logger.SAVE_DIR}/predictions.csv')
print(f'Results saved at {logger.SAVE_DIR}')