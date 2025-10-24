import os
from datetime import datetime
import pickle


class Logger:

    def __init__(self, dataset, model, add_new_tokens):

        self.SAVE_DIR = f'./results/{dataset}/{model}'

        if add_new_tokens:
            self.SAVE_DIR += '-NT'

        self.SAVE_DIR += f"/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        
        os.makedirs(self.SAVE_DIR)

    def log(self, text, date=True, print_text=False):

        if print_text:
            print(text)
            
        if date:
            text = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}: {text}"

        with open(f'{self.SAVE_DIR}/logs', 'a') as f:
            f.write(text + '\n')

    def save(self, fn, obj):

        if not fn.endswith('.pkl'):
            fn += '.pkl'
            
        with open(f'{self.SAVE_DIR}/{fn}', 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)