import ignite.distributed as idist
import torch
from transformers import AutoTokenizer
import pandas as pd

from get_data_frame import get_dataset
from utils import build_prompt, get_tokenizer


class TransformerDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def crop_text(self, text, length):
        tokens = self.tokenizer(
            text,
            max_length=int(length),
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        decoded = self.tokenizer.decode(tokens["input_ids"][0])

        return decoded

    def get_encoded(self, prompt, response_a, response_b):
        text = build_prompt("", prompt, "", "")
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=False,
            return_tensors="pt",
        )

        length_without_responses = inputs["input_ids"].shape[1]

        length_per_response = (self.max_length - length_without_responses) // 2

        response_a = self.crop_text(response_a, length_per_response * 0.8)
        response_b = self.crop_text(response_b, length_per_response * 0.8)

        text = build_prompt("", text, response_a, response_b)
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=False,
            return_tensors="pt",
        )
        decoded = self.tokenizer.decode(inputs["input_ids"][0])
        assert inputs["input_ids"].shape[1] <= self.max_length

        # presented ### Evaluation: in decoded text
        assert "Evaluation:" in decoded

        # ids = inputs["input_ids"]
        # # token_type_ids = inputs["token_type_ids"]
        # mask = inputs["attention_mask"]
        # padding_lenght = self.max_length - len(ids)

        # ids = ids + ([0] * padding_lenght)
        # mask = mask + ([0] * padding_lenght)
        # token_type_ids = token_type_ids + ([0] * padding_lenght)

        return inputs

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prompt = row["prompt"]
        response_a = row["response_a"]
        response_b = row["response_b"]
        label = row["label"]  # 0 for a, 1 for b, 2 for equal

        # if random < 0.5, response_a is first model otherwise response_b
        random = torch.rand(1).item()

        if random < 0.5:
            response_a = row["response_b"]
            response_b = row["response_a"]
            if label != 2:
                label = 1 if label == 0 else 0

        label = torch.tensor(label, dtype=torch.long)
        encoded = self.get_encoded(prompt, response_a, response_b)
        # add lables to dict encoded
        encoded["label"] = label

        return encoded

    def __len__(self):
        return len(self.df)


def setup_data(config):
    # df = pd.read_csv("data/train.csv")

    dataset_train, dataset_eval = get_dataset()
    tokenizer = get_tokenizer(config.model)

    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "right"
    # tokenizer.add_eos_token = True

    # train_texts, train_labels = dataset_train["text"], dataset_train["label"]
    # test_texts, test_labels = dataset_eval["text"], dataset_eval["label"]
    dataset_train = TransformerDataset(dataset_train, tokenizer, config.max_length)
    dataset_eval = TransformerDataset(dataset_eval, tokenizer, config.max_length)
    # dataloader_train = torch.utils.data.DataLoader(
    #     dataset_train, batch_size=32, shuffle=True, num_workers=4
    # )

    # add distributed sampler

    dataloader_train = idist.auto_dataloader(
        dataset_train,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        drop_last=True,
    )
    dataloader_eval = idist.auto_dataloader(
        dataset_eval,
        batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        shuffle=False,
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
