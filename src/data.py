import ignite.distributed as idist
import torch
from transformers import AutoTokenizer
import pandas as pd
 
from get_data_frame import get_dataset


class TransformerDataset(torch.utils.data.Dataset):
    def __init__(self, left, right, labels, tokenizer, max_length):
        self.left = left
        self.right = right
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def get_encoded(self, text):
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
        )

        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]
        padding_lenght = self.max_length - len(ids)

        ids = ids + ([0] * padding_lenght)
        mask = mask + ([0] * padding_lenght)
        token_type_ids = token_type_ids + ([0] * padding_lenght)

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long) ,
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        }

    def __getitem__(self, idx):
        left = self.left[idx]
        right = self.right[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
    
        return {
            "left": self.get_encoded(left),
            "right": self.get_encoded(right),
            "label": label,
        }

    def __len__(self):
        return len(self.labels)


def setup_data(config):
    df = pd.read_csv("data/train.csv")

    dataset_train, dataset_eval = get_dataset()
    tokenizer = AutoTokenizer.from_pretrained(
        config.model, cache_dir=config.tokenizer_dir, do_lower_case=True 
    )

    train_left_texts, train_right_texts, train_labels = (
        dataset_train["left"],
        dataset_train["right"],
        dataset_train["label"],
    )
    test_left_texts, test_right_texts, test_labels = (
        dataset_eval["left"],
        dataset_eval["right"],
        dataset_eval["label"],
    )
    # train_texts, train_labels = dataset_train["text"], dataset_train["label"]
    # test_texts, test_labels = dataset_eval["text"], dataset_eval["label"]
    dataset_train = TransformerDataset(
        train_left_texts.to_list(), train_right_texts.to_list(), train_labels.to_list(), tokenizer, config.max_length
    )
    dataset_eval = TransformerDataset(
        test_left_texts.to_list(), test_right_texts.to_list(), test_labels.to_list(), tokenizer, config.max_length
    )
    # dataloader_train = torch.utils.data.DataLoader(
    #     dataset_train, batch_size=32, shuffle=True, num_workers=4
    # )

    dataloader_train = idist.auto_dataloader(
        dataset_train,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        drop_last=True 
    )
    dataloader_eval = idist.auto_dataloader(
        dataset_eval,
        batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        shuffle=False 
    )

    return dataloader_train, dataloader_eval


if __name__ == "__main__":
    from omegaconf import OmegaConf

    config = OmegaConf.load("config.yaml")
    dataloader_train, dataloader_eval = setup_data(config)
    for batch in dataloader_train:
        print(batch)
        break
    # for batch in dataloader_eval:
    #     print(batch)
    #     break
