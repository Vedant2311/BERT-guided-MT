import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, T5Tokenizer

class NepaliEnglishDataset(Dataset):
    def __init__(self, bert_tokenizer, mT5_tokenizer, max_length, src_data_path, trg_data_path, isTest = False):
        self.bert_tokenizer = bert_tokenizer
        self.mT5_tokenizer = mT5_tokenizer
        self.max_length = max_length
        self.src_data_path = src_data_path
        self.trg_data_path = trg_data_path
        self.isTest = isTest

        # read the .txt files 
        with open(self.src_data_path, "r") as f:
            source_lines = f.readlines()
            # Adding a prompt as typically suggested for fine-tuning mT5
            self.src_data = ["Translate Nepali to English: " + line for line in source_lines]

        with open(self.trg_data_path, "r") as f:
            self.trg_data = f.readlines()

        # convert the data into pandas dataframe
        self.src_data = pd.DataFrame(self.src_data, columns=["src"])
        self.trg_data = pd.DataFrame(self.trg_data, columns=["trg"])

        # merge the dataframes
        self.data = pd.concat([self.src_data, self.trg_data], axis=1)

    def tokenize_data(self, text, tokenizer):
        tokens = tokenizer.encode(text, add_special_tokens=True, max_length=self.max_length, truncation=True)
        text_len = len(tokens)
        padding = [tokenizer.pad_token_id] * (self.max_length - text_len)
        return torch.tensor(tokens + padding), torch.tensor(text_len)
    
    def tokenize_mT5_label(self, src, trg, tokenizer):
        inputs = tokenizer(src, text_target=trg)
        mT5_input_ids = inputs["input_ids"]
        mT5_input_ids_len = len(mT5_input_ids)
        padding = [tokenizer.pad_token_id] * (self.max_length - mT5_input_ids_len)
        labels = inputs["labels"]
        labels_padding = [-100] * (self.max_length - len(labels))
        return torch.tensor(mT5_input_ids + padding), torch.tensor(labels + labels_padding), torch.tensor(mT5_input_ids_len)

    def tokenize_test(self, text, tokenizer):
        tokens = tokenizer.encode(text, add_special_tokens=True, max_length=self.max_length, truncation=True)
        text_len = len(tokens)
        return torch.tensor(tokens), torch.tensor(text_len)
    
    def tokenize_mT5_label_test(self, src, trg, tokenizer):
        inputs = tokenizer(src, text_target=trg)
        mT5_input_ids = inputs["input_ids"]
        mT5_input_ids_len = len(mT5_input_ids)
        labels = inputs["labels"]
        return torch.tensor(mT5_input_ids), torch.tensor(labels), torch.tensor(mT5_input_ids_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.isTest:
            row = self.data.iloc[idx]
            src_text, trg_text = row["src"], row["trg"]
            input_ids, input_ids_len = self.tokenize_test(src_text, self.bert_tokenizer)
            mT5_input_ids, labels, mT5_input_ids_len = self.tokenize_mT5_label_test(src_text, trg_text, self.mT5_tokenizer)
            if input_ids_len < mT5_input_ids_len:
                padding = [self.bert_tokenizer.pad_token_id] * (mT5_input_ids_len - input_ids_len)
                input_ids = torch.tensor(input_ids.tolist() + padding)
            elif input_ids_len > mT5_input_ids_len:
                padding = [self.mT5_tokenizer.pad_token_id] * (input_ids_len - mT5_input_ids_len)
                mT5_input_ids = torch.tensor(mT5_input_ids.tolist() + padding)

            return input_ids, labels, mT5_input_ids
        
        else:
            row = self.data.iloc[idx]
            src_text, trg_text = row["src"], row["trg"]

            input_ids, input_ids_len = self.tokenize_data(src_text, self.bert_tokenizer)
            encoder_mask = (input_ids != self.bert_tokenizer.pad_token_id).long()

            mT5_input_ids, labels, _ = self.tokenize_mT5_label(src_text, trg_text, self.mT5_tokenizer)
            mT5_input_mask = (mT5_input_ids != self.mT5_tokenizer.pad_token_id).long()

            return input_ids, encoder_mask, labels, mT5_input_ids, mT5_input_mask, input_ids_len
        
if __name__ == "__main__":
    # test the dataset
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    mT5_tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")
    max_length = 128
    src_data_path = "../dataset/test_raw/test.ne_NP"
    trg_data_path = "../dataset/test_raw/test.en_XX"

    dataset = NepaliEnglishDataset(bert_tokenizer, mT5_tokenizer, max_length, src_data_path, trg_data_path)

    print(dataset[0])
    print(dataset[1])
    print(dataset[2])
    print(dataset[3])