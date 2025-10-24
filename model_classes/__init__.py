from .albert import ALBERT
from .bart import BART
from .bert import BERT
from .bertweet import BERTweet
from .ct_bert import CTBERT
from .distilbert import DistilBERT
from .longformer import Longformer
from .roberta import RoBERTa
from .twitter_roberta import TwitterRoBERTa
from .xlm import XLM
from .xlm_roberta import XLMRoBERTa
from .xlnet import XLNet


map = {
    'albert-base-v2': ALBERT,
    'bart-base': BART,
    'bert-base-uncased': BERT,
    'bertweet-covid19-base-uncased': BERTweet,
    'covid-twitter-bert-v2': CTBERT,
    'distilbert-base-uncased': DistilBERT,
    'longformer-base-4096': Longformer,
    'roberta-base': RoBERTa,
    'twitter-roberta-base-sentiment-latest': TwitterRoBERTa,
    'xlm-mlm-en-2048': XLM,
    'xlm-roberta-base': XLMRoBERTa,
    'xlnet-base-cased': XLNet,
}


def modelclass_map(model_name):

    model_name = model_name.lower()

    return map[model_name]