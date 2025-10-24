import argparse
import os
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.optim import Adam

from data_classes import dataclass_map 
from data_classes.preprocess import PreProcessor
from model_classes import modelclass_map
from utils import adjust
from utils.config import Config
from utils.logger import Logger
from utils.tokenizer import tokenize
from utils.data_loaders import data_loaders
from utils.train import train_batch
from utils.test import test_epoch 
from utils.metrics import Results


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
Model = modelclass_map(args.model)
DEVICE = torch.device(f'cuda:{cfg.DEVICE_INDEX}' if torch.cuda.is_available() and cfg.DEVICE_INDEX is not None else 'cpu')


logger = Logger(
    dataset=args.dataset,
    model=cfg.MODEL.SAVE_NAME,
    add_new_tokens=cfg.ADD_NEW_TOKENS
)
preprocessor = PreProcessor()

logger.log(f'DATASET = {os.path.basename(cfg.DATA.args.root)}', date=False)
logger.log(f"COVID WORDS = {', '.join((w for w in preprocessor.COVID_WORDS if not w.startswith('#')))}", date=False)
logger.log(f"TOKENS = {', '.join(preprocessor.TOKENS)}", date=False)
logger.log(f'ADD NEW TOKENS = {cfg.ADD_NEW_TOKENS}', date=False)
logger.log(f'MAX LENGTH = {cfg.MODEL.MAX_LENGTH}', date=False)
logger.log(f'BATCH SIZE = {cfg.DATA.BATCH_SIZE}', date=False)
logger.log(f'MODEL = {os.path.basename(cfg.MODEL.HF_PATH)}', date=False)
logger.log(f'LEARNING RATE = {cfg.LR}', date=False)
logger.log(f'TEST EVERY = {cfg.DATA.TEST_EVERY}', date=False)
logger.log(f'SAVE EVERY = {cfg.DATA.SAVE_EVERY}\n', date=False)


logger.log('Loading and pre-processing datasets...')
datasets = Dataset(
    root=cfg.DATA.args.root,
    train_pct=cfg.DATA.args.train_pct,
    valid_pct=cfg.DATA.args.valid_pct,
)
datasets.apply(
    func=preprocessor.preprocess,
    splits=cfg.SPLITS,
    col='tweet',
    new_col='tweet'
)
logger.log('Finished pre-processing datasets.\n')


logger.log('Tokenizing datasets...')
tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.HF_PATH)
if cfg.ADD_NEW_TOKENS:
    tokenizer.add_tokens(preprocessor.TOKENS)
encoded_inputs, dense_features, binarized_labels = tokenize(
    tokenizer,
    datasets,
    splits=cfg.SPLITS,
    max_length=cfg.MODEL.MAX_LENGTH,
    add_dense=cfg.DATA.ADD_DENSE
)
logger.log('Finished tokenizing datasets.\n')

logger.log('Preparing data-loaders...')
train_loader, val_loader, test_loader = data_loaders(
    cfg.SPLITS,
    encoded_inputs,
    dense_features,
    binarized_labels,
    batch_size=cfg.DATA.BATCH_SIZE
)
logger.log('Finished preparing data-loaders.\n')


logger.log('Loading and preparing model...')
base_model = AutoModelForSequenceClassification.from_pretrained(cfg.MODEL.HF_PATH)
model = Model(
    base_model,
    emb_table_size=len(tokenizer),
    dense_size=dense_features[cfg.SPLITS[0]].size(1) if dense_features[cfg.SPLITS[0]] is not None else None,
    device=DEVICE
)
logger.log('Finshed preparing model.\n')

optimizer = Adam(model.base_model.parameters(), lr=cfg.LR)


logger.log(f"Starting training...\n", print_text=True)
results, batch = Results(), 0
TOTAL_BATCHES = cfg.DATA.N_EPOCHS*len(train_loader)

for epoch in range(cfg.DATA.N_EPOCHS):

    for input_ids, attention_mask, dense_features, labels in tqdm(train_loader):

        batch += 1
        input_ids, attention_mask, labels = map(lambda x: x.to(DEVICE, non_blocking=True), (input_ids, attention_mask, labels,))
        if dense_features is not None:
            dense_features = dense_features.to(DEVICE, non_blocking=True)
        labels, preds, loss = train_batch(model, optimizer, input_ids, attention_mask, dense_features, labels)
        results.update('training', batch, labels, preds, loss)

        if batch % cfg.DATA.TEST_EVERY == 0 or batch == TOTAL_BATCHES:
            accuracy, f1_score, loss = results.metrics('training', last=cfg.DATA.TEST_EVERY)
            logger.log(f'Training (last {cfg.DATA.TEST_EVERY} batches): accuracy = {accuracy:.6f}, f1-score = {f1_score:.6f}, loss = {loss:.6f}', print_text=True)
            labels, preds, loss = test_epoch(model, val_loader, DEVICE)
            results.update('validation', batch, labels, preds, loss)
            accuracy, f1_score, loss = results.metrics('validation', last=1)
            logger.log(f'Validation (total {len(val_loader)} batches): accuracy = {accuracy:.6f}, f1-score = {f1_score:.6f}, loss = {loss:.6f}', print_text=True)
            labels, preds, loss = test_epoch(model, test_loader, DEVICE)
            results.update('testing', batch, labels, preds, loss)
            accuracy, f1_score, loss = results.metrics('testing', last=1)
            logger.log(f'Testing (total {len(test_loader)} batches): accuracy = {accuracy:.6f}, f1-score = {f1_score:.6f}, loss = {loss:.6f}', print_text=True)
            logger.log(f'Finished batch {batch}.\n', print_text=True)

        if cfg.DATA.SAVE_EVERY is not None and (batch % cfg.DATA.SAVE_EVERY == 0 or batch == TOTAL_BATCHES):
            ckpt_fn = f'{logger.SAVE_DIR}/ckpt{adjust(batch, TOTAL_BATCHES)}.pth'
            logger.log(f'Saving model at {ckpt_fn}...', print_text=True)
            torch.save(model.base_model.state_dict(), ckpt_fn)
            logger.log(f'Finished saving model at {ckpt_fn}.\n', print_text=True)


for type in ('training', 'validation', 'testing'):
    logger.save(f'{type}_results', results.results[type])