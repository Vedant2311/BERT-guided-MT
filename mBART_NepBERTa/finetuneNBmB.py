from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, AutoTokenizer, AutoModelForMaskedLM
from NBmB import NBmB
import argparse
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--finetune", "-f", action="store_true")
parser.add_argument("--test", "-t", action="store_true")

args = parser.parse_args()

def load_model():
    mBart_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
    mBart_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="ne_NP", tgt_lang="en_XX")

    nepBerta_model = AutoModelForMaskedLM.from_pretrained("tf_model.h5")
    nepBerta_tokenizer = AutoTokenizer.from_pretrained("vocab.txt")

    nbmb_model = NBmB(mBart_model, nepBerta_model, device=device)

    # combine mBart and nepBerta tokenizer
    nbmb_tokenizer = mBart_tokenizer
    nbmb_tokenizer.add_tokens(nepBerta_tokenizer.additional_special_tokens)
    nbmb_tokenizer.add_tokens(nepBerta_tokenizer.all_special_tokens)
    nbmb_tokenizer.add_tokens(nepBerta_tokenizer.all_special_ids)
    nbmb_tokenizer.add_tokens(nepBerta_tokenizer.all_vocab_tokens)
    nbmb_tokenizer.add_tokens(nepBerta_tokenizer.all_vocab_ids)
    
    return nbmb_model, nbmb_tokenizer

def finetune():
    pass

def test():
    nbmb_model, nbmb_tokenizer = load_model()
    print(nbmb_model)
    print(nbmb_tokenizer)
    
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

    output = nbmb_model(batch)

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