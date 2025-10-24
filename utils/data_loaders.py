from torch import stack
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):

    def __init__(self, encoded_input, dense_features, binarized_labels):

        self.encoded_input = encoded_input
        self.dense_features = dense_features
        self.binarized_labels = binarized_labels

    def __len__(self):

        return self.binarized_labels.size(0)
    
    def __getitem__(self, idx):

        return self.encoded_input['input_ids'][idx], \
            self.encoded_input['attention_mask'][idx], \
            self.dense_features[idx] if self.dense_features is not None else None, \
            self.binarized_labels[idx]


def data_loaders(splits, encoded_input, dense_features, binarized_labels, batch_size, shuffle=True):

    def collate_fn(batch):

        input_ids, attention_mask, dense_features, labels = zip(*batch)
        if any((x is None for x in dense_features)):
            dense_features = None

        input_ids, attention_mask, labels = map(lambda x: stack(x, 0), (input_ids, attention_mask, labels,))
        if dense_features is not None:
            dense_features = stack(dense_features, 0)

        return input_ids, attention_mask, dense_features, labels

    for split in splits:
    
        if split in encoded_input and split in binarized_labels:
    
            dataset = CustomDataset(
                encoded_input[split],
                dense_features[split],
                binarized_labels[split],
            )
    
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=collate_fn,
                pin_memory=True,
            )
    
            yield data_loader