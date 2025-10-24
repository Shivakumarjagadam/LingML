from tqdm import tqdm

from torch import Tensor, long


def tokenize(tokenizer, dataset, splits='all', max_length=128, add_dense=False):

    encoded_input, binarized_labels, dense_features = dict(), dict(), dict()

    for split in splits:
    
        df = dataset[split]
        input_ids, attention_mask = list(), list()
    
        for tweet in tqdm(df.tweet):
    
            encoded = tokenizer(
                tweet,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True
            )
    
            input_ids.append(encoded['input_ids'])
            attention_mask.append(encoded['attention_mask'])
    
        encoded_input[split] = {'input_ids': Tensor(input_ids).to(long), 'attention_mask': Tensor(attention_mask)}
        binarized_labels[split] = Tensor(df.label.values).to(long)
        dense_features[split] = Tensor(df[df.columns.difference(('tweet', 'label'))].values) if add_dense else None

    return encoded_input, dense_features, binarized_labels