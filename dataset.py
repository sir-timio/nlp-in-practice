from typing import Callable

import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import T5Tokenizer


class ReceiptsDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: T5Tokenizer,
        source_max_token_length: int = 256,
        target_max_token_length: int = 32,
    ):
        super().__init__()
        self.is_predict = "target_text" not in df.columns
        self.data = (
            df[["input_text", "target_text"]]
            if not self.is_predict
            else df[["input_text"]]
        )
        self.data = self.data.values
        self.tokenizer = tokenizer
        self.source_max_token_length = source_max_token_length
        self.target_max_token_length = target_max_token_length

    def __getitem__(self, index):
        input_text = self.data[index][0]
        target_text = self.data[index][1] if not self.is_predict else ""

        source_encoding = self.tokenizer(
            input_text,
            max_length=self.source_max_token_length,
            padding="max_length",
            truncation="only_second",
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        if target_text != "":
            target_encoding = self.tokenizer(
                target_text,
                max_length=self.target_max_token_length,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )

            labels = target_encoding["input_ids"]
            labels[labels == 0] = -100
        else:
            target_encoding = torch.zeros(1)
            labels = torch.zeros(1)

        return dict(
            input_text=input_text,
            target_text=target_text,
            input_ids=source_encoding["input_ids"].flatten(),
            attention_mask=source_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
        )

    def __len__(self):
        return len(self.data)


class ReceiptsDataModule(pl.LightningDataModule):
    def __init__(self, hparam):
        super().__init__()
        self.hparam = hparam
        self.tokenizer = T5Tokenizer.from_pretrained(self.hparam.tokenizer_name)

    def prepare_data(self):
        self.train_df = pd.read_csv(self.hparam.train_dataset_path).fillna("")
        self.test_df = pd.read_csv(self.hparam.test_dataset_path)

    def setup(self, stage: str):
        self.train_df, self.val_df = train_test_split(
            self.train_df, test_size=self.hparam.val_split_size, random_state=42
        )

        self.train_dataset = ReceiptsDataset(self.train_df, self.tokenizer)
        self.val_dataset = ReceiptsDataset(self.val_df, self.tokenizer)
        self.predict_dataset = ReceiptsDataset(self.test_df, self.tokenizer)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparam.batch_size,
            num_workers=self.hparam.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparam.batch_size,
            num_workers=self.hparam.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.hparam.batch_size,
            num_workers=self.hparam.num_workers,
        )
