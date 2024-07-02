import ignite.distributed as idist
import torch
from transformers import AutoTokenizer
import pandas as pd
 
from get_data_frame import get_dataset


class TransformerDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length): 
        self.df = df 
        self.tokenizer = tokenizer
        self.max_length = max_length

    def get_encoded(self, text):
        inputs = self.tokenizer(
            text, 
            padding='max_length', 
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
        )

        ids = inputs["input_ids"]
        # token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]
        # padding_lenght = self.max_length - len(ids)

        # ids = ids + ([0] * padding_lenght)
        # mask = mask + ([0] * padding_lenght)
        # token_type_ids = token_type_ids + ([0] * padding_lenght)

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long) ,
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            # "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        }

    def __getitem__(self, idx): 
        row = self.df.iloc[idx]
        prompt = row["prompt"]
        response_a = row["response_a"]
        response_b = row["response_b"]
        label = row["label"] # 0 for a, 1 for b, 2 for equal
        
        # if random < 0.5, response_a is first model otherwise response_b
        random = torch.rand(1).item()
        text = ""

        if random < 0.5:
            text = f"Prompt: {prompt}\n\n\n\n Model A: {response_a} \n\n\n\n  Model B: {response_b}" 
        else:
            text = f"Prompt: {prompt}\n\n\n\n Model A: {response_b} \n\n\n\n  Model B: {response_a}"
            if label != 2:
                label = 1 if label == 0 else 0

        label = torch.tensor(label, dtype=torch.long)
        encoded = self.get_encoded(text)
        # add lables to dict encoded 
        encoded["label"] = label

        return encoded

    def __len__(self):
        return len(self.df)


def setup_data(config):
    # df = pd.read_csv("data/train.csv")

    dataset_train, dataset_eval = get_dataset()
    tokenizer = AutoTokenizer.from_pretrained(config.model, do_lower_case=True)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    tokenizer.add_eos_token = True
 
    # train_texts, train_labels = dataset_train["text"], dataset_train["label"]
    # test_texts, test_labels = dataset_eval["text"], dataset_eval["label"]
    dataset_train = TransformerDataset(
        dataset_train, tokenizer, config.max_length
    )
    dataset_eval = TransformerDataset(
        dataset_eval, tokenizer, config.max_length
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
