import ignite.distributed as idist
import torch
import pandas as pd
from tqdm import tqdm
from get_data_frame import get_dataset
from utils import build_prompt, build_prompt2, get_tokenizer

class TransformerDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def crop_text(self, text, length):
        try:
            tokens = self.tokenizer(
                text,
                max_length=int(length),
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
        except:
            print(f"Text: {text}")
            print(f"Length: {length}")
            raise
        decoded = self.tokenizer.decode(tokens["input_ids"][0])

        return decoded

    def get_encoded(self, prompt, response_a, response_b):
        # Tokenize the prompt without truncation
        prompt_inputs = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=False,
            return_tensors="pt",
        )

        prompt_length = prompt_inputs["input_ids"].shape[1]

        # Calculate remaining length for responses
        remaining_length = self.max_length - prompt_length

        if remaining_length <= 0 or remaining_length < self.max_length * 0.7:
            prompt = self.crop_text(prompt, self.max_length * 0.3)
            prompt_inputs = self.tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=False,
                return_tensors="pt",
            )
            prompt_length = prompt_inputs["input_ids"].shape[1]
            remaining_length = self.max_length - prompt_length
            # raise ValueError("Prompt itself exceeds max_length")

        # Allocate length for each response
        length_per_response = remaining_length // 2

        if length_per_response <= 0:
            raise ValueError("Prompt itself exceeds max_length")

        percent = 1.0
        tokenized_length = 0
        loop_count = 0
        while tokenized_length > self.max_length or tokenized_length == 0:
            # Truncate responses to fit within the allocated length
            response_a = self.crop_text(response_a, length_per_response * percent)
            response_b = self.crop_text(response_b, length_per_response * percent)

            # Build the final combined text
            text = build_prompt("", prompt, response_a, response_b)
            inputs = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=False, 
                return_tensors="pt",
            )
            percent -= 0.1
            tokenized_length = inputs["input_ids"].shape[1]
            loop_count += 1
            if loop_count > 10:
                print("Loop count exceeded 10")
            
            if percent < 0.1:
                print("Percent less than 0.1")


        # # Truncate responses to fit within the allocated length
        # response_a = self.crop_text(response_a, length_per_response)
        # response_b = self.crop_text(response_b, length_per_response)

        # # Build the final combined text
        # text = build_prompt("", prompt, response_a, response_b)
        # inputs = self.tokenizer(
        #     text,
        #     add_special_tokens=True,
        #     max_length=self.max_length,
        #     truncation=False, 
        #     return_tensors="pt",
        # )

        decoded = self.tokenizer.decode(inputs["input_ids"][0])

        try:
            assert inputs["input_ids"].shape[1] <= self.max_length
        except:
            print(f"Prompt: {prompt}")
            print(f"Response A: {response_a}")
            print(f"Response B: {response_b}")
            print(f"Text: {text}")
            print(f"Decoded: {decoded}")
            print(f"Length: {inputs['input_ids'].shape[1]}")
            raise
        assert inputs["input_ids"].shape[1] <= self.max_length

        # Ensure "Evaluation:" is in the decoded text
        assert "Evaluation:" in decoded

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

        prompt = build_prompt2(
            self.tokenizer,
            self.max_length,
            "",
            prompt,
            response_a,
            response_b,
        )
        encoded = self.tokenizer(
                prompt,
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=False, 
                return_tensors="pt",
            )

        # print(f"Shape: {encoded['input_ids'].shape}")
        try:
            assert encoded["input_ids"].shape[1] <= self.max_length
        except:
            print(f"Prompt: {prompt}")
            print(f"Length: {encoded['input_ids'].shape[1]}")
            raise
        
        # encoded = self.get_encoded(prompt, response_a, response_b)
        # add lables to dict encoded
        encoded["label"] = label
        encoded["input_ids"] = encoded["input_ids"].squeeze()
        encoded["attention_mask"] = encoded["attention_mask"].squeeze()

        return encoded

    def __len__(self):
        return len(self.df)


def setup_data(config):
    # df = pd.read_csv("data/train.csv")

    dataset_train, dataset_eval = get_dataset()
    tokenizer = get_tokenizer(config.model)
 
    dataset_train = TransformerDataset(dataset_train, tokenizer, config.max_length)
    dataset_eval = TransformerDataset(dataset_eval, tokenizer, config.max_length)
 
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
    tokenizer = get_tokenizer(config.model)
    asd = next(iter(dataloader_train))
    decodedl = []
    # show progress bar
    for i, batch in tqdm(enumerate(dataloader_train)):
        # print(batch.shape)
        decoded = tokenizer.decode(batch["input_ids"].squeeze())
        decodedl.append(decoded)
    
    # create df from decoded
    df = pd.DataFrame(decodedl)

    df.to_csv("./decoded.csv", index=False)
         
 
