import re
import string
from typing import Union, Iterable

from emoji import replace_emoji


URL_REGEX = r'https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)'
HASHTAG_REGEX = r'#[^\s]+'
MENTION_REGEX = r'@[^\s]+'

URL_TOKEN = '[URL]'
HASHTAG_TOKEN = '[HASHTAG]'
MENTION_TOKEN = '[MENTION]'
EMOJI_TOKEN = '[EMOJI]'
COVID_TOKEN = '[COVID]'

COVID_WORDS = [
    'covid 19', 'covid-19', 'covid19', 'covid',
    'corona virus', 'coronavirus', 'corona',
]


class PreProcessor:

    def __init__(
        self, 
        url_token: Union[str, None] = URL_TOKEN, hashtag_token: Union[str, None] = HASHTAG_TOKEN,
        mention_token: Union[str, None] = MENTION_TOKEN, emoji_token: Union[str, None] = EMOJI_TOKEN,
        covid_token: Union[str, None] = COVID_TOKEN, covid_words: Union[str, Iterable, None] = COVID_WORDS
    ):

        self.URL_TOKEN = url_token
        self.HASHTAG_TOKEN = hashtag_token
        self.MENTION_TOKEN = mention_token
        self.EMOJI_TOKEN = emoji_token
        self.COVID_TOKEN = covid_token
        self.TOKENS = [self.URL_TOKEN, self.HASHTAG_TOKEN, self.MENTION_TOKEN, self.EMOJI_TOKEN, self.COVID_TOKEN]
        self.COVID_WORDS = covid_words

    def preprocess(
        self,
        text: str, uncase: bool = True, remove_newline=True, remove_space: bool = True,
        condense_letters: bool = True, remove_punctuation: Union[str, None] = None,
        verbose: bool = True
    ):

        if uncase:
            text = text.lower()
        if remove_newline:
            text = re.sub(r'\n+', ' ', text)

        if self.URL_TOKEN is not None:
            text = re.sub(URL_REGEX, self.URL_TOKEN, text)
        if self.MENTION_TOKEN is not None:
            text = re.sub(MENTION_REGEX, ' ' + self.MENTION_TOKEN, text)
        if self.COVID_TOKEN is not None:
            COVID_WORDS = self.COVID_WORDS + ['#' + w for w in self.COVID_WORDS if not w.startswith('#')]
            text = re.sub('(?i)'+'|'.join(COVID_WORDS), ' ' + self.COVID_TOKEN + ' ', text)
        if self.HASHTAG_TOKEN is not None:
            text = re.sub(r'#\s', ' ', text)
            text = re.sub(HASHTAG_REGEX, ' ' + self.HASHTAG_TOKEN, text)

        if condense_letters:
            text = re.sub(r'(\w)\1{2,}', r'\1\1', text)

        if remove_punctuation is not None and remove_punctuation != '':
            try:
                remove_punctuation = set(string.punctuation).intersection(remove_punctuation)
            except:
                if verbose:
                    print(f'remove_punctuation = {remove_punctuation} is not iterable, defaulting to string.punctuation')
                remove_punctuation = string.punctuation
            remove_punctuation = ''.join(('\\' + p for p in remove_punctuation))
            text = re.sub(f"[{remove_punctuation}]", '', text)

        emoji_token = ' ' + self.EMOJI_TOKEN + ' '
        if emoji_token is not None:
            replace = lambda _, data_dict: emoji_token + ' '.join(data_dict['en'].split('_')).strip(':') + emoji_token
        else:
            replace = lambda _: emoji_token
        text = replace_emoji(text, replace=replace)

        if remove_space:
            text = re.sub(r'^\s+|\s+$', '', text)
            text = re.sub(r'\s+', ' ', text)

        return text