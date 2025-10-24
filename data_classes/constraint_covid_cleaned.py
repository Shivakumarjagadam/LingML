import pandas as pd

from .base import BaseDataset
from .preprocess import MENTION_TOKEN, HASHTAG_TOKEN, URL_TOKEN


def add_tokens(row):

    tweet, mention, hashtag, included_url, ref_url = row
    tweet = tweet.strip()
    if mention and not pd.isna(mention):
        tweet = MENTION_TOKEN + ' ' + tweet
    if hashtag and not pd.isna(hashtag):
        tweet = tweet + ' ' + HASHTAG_TOKEN
    if included_url and not pd.isna(included_url) or ref_url and not pd.isna(ref_url):
        tweet = tweet + ' ' + URL_TOKEN
    
    return tweet


class ConstraintCleaned(BaseDataset):

    def __init__(
        self, 
        root: str,
        train_pct: int,
        valid_pct: int,
    ):

        df = pd.read_csv(f'{root}/dataset.csv', usecols=['tweet_denoised', 'account', 'hashtags', 'Included_URL', 'Reference_URL', 'veracity_finetuned'])
        df = df.sample(frac=1).reset_index(drop=True).rename({'tweet_denoised': 'tweet', 'veracity_finetuned': 'label'}, axis=1)
        df = df.loc[~df.tweet.isna() & df.label.isin(('real', 'fake')), :]
        df.label = df.label.replace({'real': 1, 'fake': 0})
        df.tweet = df.iloc[:, df.columns!='label'].apply(add_tokens, axis=1, raw=True)
        df.drop(['account', 'hashtags', 'Included_URL', 'Reference_URL'], axis=1, inplace=True)
        
        train_end, val_end = int(train_pct*df.shape[0]/100), int((train_pct+valid_pct)*df.shape[0]/100)
        self.datasets = {
            'train': df.iloc[:train_end, :],
            'val': df.iloc[train_end:val_end, :],
            'test': df.iloc[val_end:, :]
        }