from transformers import MT5Model, AutoTokenizer, AutoModelForMaskedLM
from NBT5 import NBT5
import argparse
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--finetune", action="store_true")
parser.add_argument("--test", action="store_true")

args = parser.parse_args()

def load_model():
    t5_model = MT5Model.from_pretrained("google/mt5-base")
    t5_tokenizer = AutoTokenizer.from_pretrained("google/mt5-base", src_lang="ne_NP", tgt_lang="en_XX")

    nepBerta_model = AutoModelForMaskedLM.from_pretrained("NepBERTa/NepBERTa")
    nepBerta_tokenizer = AutoTokenizer.from_pretrained("NepBERTa/NepBERTa")

    nbt5_model = NBT5(t5_model, nepBerta_model, device=device)

    # combine t5 and nepBerta tokenizer
    nbt5_tokenizer = t5_tokenizer
    nbt5_tokenizer.add_tokens(nepBerta_tokenizer.additional_special_tokens)
    nbt5_tokenizer.add_tokens(nepBerta_tokenizer.all_special_tokens)
    nbt5_tokenizer.add_tokens(nepBerta_tokenizer.all_special_ids)
    nbt5_tokenizer.add_tokens(nepBerta_tokenizer.all_vocab_tokens)
    nbt5_tokenizer.add_tokens(nepBerta_tokenizer.all_vocab_ids)
    
    return nbt5_model, nbt5_tokenizer

def finetune():
    pass

def test():
    nbt5_model, nbt5_tokenizer = load_model()
    print(nbt5_model)
    print(nbt5_tokenizer)
    
    # run a test with random tensor
    input_ids = torch.randint(0, 100, (1, 1024)).to(device)
    attention_mask = torch.randint(0, 2, (1, 1024)).to(device)
    decoder_input_ids = torch.randint(0, 100, (1, 1024)).to(device)
    decoder_attention_mask = torch.randint(0, 2, (1, 1024)).to(device)

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask
    }

    output = nbt5_model(batch)

    print(output)


def main():
    if args.finetune:
        finetune()
    elif args.test:
        test()
    else:
        print("Please specify --finetune or --test")

if __name__ == "__main__":
    main()