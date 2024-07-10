from torch.utils.data import Dataset


class GTEDataset(Dataset):
    def __init__(self, config, data, tokenizer):
        self.config = config
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = config.max_length

    def __getitem__(self, index):
        query, document = self.data[index]
        query = self.tokenizer(
            query,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        document = self.tokenizer(
            document,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        return {
            "queries": query,
            "documents": document
        }


    def __len__(self):
        return len(self.data)