import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

class NepaliEnglishDataset(Dataset):
    def __init__(self, bert_tokenizer, mbart_tokenizer, max_length, src_data_path, trg_data_path, isTest = False):
        self.bert_tokenizer = bert_tokenizer
        self.mbart_tokenizer = mbart_tokenizer
        self.max_length = max_length
        self.src_data_path = src_data_path
        self.trg_data_path = trg_data_path
        self.isTest = isTest

        # read the .txt files 
        with open(self.src_data_path, "r") as f:
            self.src_data = f.readlines()
        
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
    
    def tokenize_mbart_label(self, src, trg, tokenizer):
        inputs = tokenizer(src, text_target=trg)
        mbart_input_ids = inputs["input_ids"]
        mbart_input_ids_len = len(mbart_input_ids)
        padding = [tokenizer.pad_token_id] * (self.max_length - mbart_input_ids_len)
        labels = inputs["labels"]
        labels_padding = [-100] * (self.max_length - len(labels))
        return torch.tensor(mbart_input_ids + padding), torch.tensor(labels + labels_padding), torch.tensor(mbart_input_ids_len)

    def tokenize_test(self, text, tokenizer):
        tokens = tokenizer.encode(text, add_special_tokens=True, max_length=self.max_length, truncation=True)
        text_len = len(tokens)
        return torch.tensor(tokens), torch.tensor(text_len)
    
    def tokenize_mbart_label_test(self, src, trg, tokenizer):
        inputs = tokenizer(src, text_target=trg)
        mbart_input_ids = inputs["input_ids"]
        mbart_input_ids_len = len(mbart_input_ids)
        labels = inputs["labels"]
        return torch.tensor(mbart_input_ids), torch.tensor(labels), torch.tensor(mbart_input_ids_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.isTest:
            row = self.data.iloc[idx]
            src_text, trg_text = row["src"], row["trg"]
            input_ids, input_ids_len = self.tokenize_test(src_text, self.bert_tokenizer)
            mbart_input_ids, labels, mbart_input_ids_len = self.tokenize_mbart_label_test(src_text, trg_text, self.mbart_tokenizer)
            if input_ids_len < mbart_input_ids_len:
                padding = [self.bert_tokenizer.pad_token_id] * (mbart_input_ids_len - input_ids_len)
                input_ids = torch.tensor(input_ids.tolist() + padding)

            return input_ids, None, labels, mbart_input_ids, None, input_ids_len
        
        else:
            row = self.data.iloc[idx]
            src_text, trg_text = row["src"], row["trg"]

            input_ids, input_ids_len = self.tokenize_data(src_text, self.bert_tokenizer)
            encoder_mask = (input_ids != self.bert_tokenizer.pad_token_id).long()

            mbart_input_ids, labels, mbart_input_ids_len = self.tokenize_mbart_label(src_text, trg_text, self.mbart_tokenizer)
            mbart_input_mask = (mbart_input_ids != self.mbart_tokenizer.pad_token_id).long()

            return input_ids, encoder_mask, labels, mbart_input_ids, mbart_input_mask, input_ids_len
        
def custom_collate_fn(batch):
    input_ids, encoder_masks, labels, mbart_input_ids, mbart_input_masks, input_ids_lens = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    encoder_masks = torch.stack(encoder_masks, dim=0)
    labels = torch.stack(labels, dim=0)
    mbart_input_ids = torch.stack(mbart_input_ids, dim=0)
    mbart_input_masks = torch.stack(mbart_input_masks, dim=0)
    input_ids_lens = torch.stack(input_ids_lens, dim=0)
    return input_ids, encoder_masks, labels, mbart_input_ids, mbart_input_masks, input_ids_lens

if __name__ == "__main__":
    # test the dataset
    from transformers import BertTokenizer, MBartTokenizer
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    mbart_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="ne_NP", tgt_lang="en_XX")
    max_length = 128
    src_data_path = "../dataset/train_raw/train.ne_NP"
    trg_data_path = "../dataset/train_raw/train.en_XX"

    dataset = NepaliEnglishDataset(bert_tokenizer, mbart_tokenizer, max_length, src_data_path, trg_data_path)

    print(dataset[0])
    print(dataset[1])
    print(dataset[2])
    print(dataset[3])