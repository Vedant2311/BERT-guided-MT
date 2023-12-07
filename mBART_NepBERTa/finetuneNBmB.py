from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from NBmB import NBmB
import argparse
import torch
from customDataset import NepaliEnglishDataset
from tqdm import tqdm
import wandb
from nltk.translate.bleu_score import corpus_bleu

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--finetune", "-f", action="store_true")
parser.add_argument("--test", "-t", action="store_true")
parser.add_argument("--epoch", "-e", type=int, default=50)
parser.add_argument("--batch_size", "-b", type=int, default=8)
parser.add_argument("--lr", "-l", type=float, default=1e-5)
parser.add_argument("--wandb", "-w", action="store_true", default=False)

args = parser.parse_args()

if args.wandb:
    wandb.init(project="Nepali-English-MT",
               config = {
                    "epoch": args.epoch,
                    "batch_size": args.batch_size,
                    "lr": args.lr
               })

def load_model():
    mBart_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50", output_hidden_states=True)
    mBart_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="ne_NP", tgt_lang="en_XX")

    nepBerta_config = AutoConfig.from_pretrained("NepBERTa/NepBERTa", output_hidden_states=True) # typo in their config 'state' => 'states'
    nepBerta_model = AutoModelForMaskedLM.from_pretrained("NepBERTa/NepBERTa", from_tf=True, config=nepBerta_config)

    nepBerta_tokenizer = AutoTokenizer.from_pretrained("NepBERTa/NepBERTa")
    nepBerta_tokenizer.model_max_length = 512

    nbmb_model = NBmB(mBart_model, nepBerta_model, mBart_tokenizer)

    num_device = torch.cuda.device_count()
    if num_device > 1:
        nbmb_model = torch.nn.DataParallel(nbmb_model)
        print("Using DataParallel for NBmB")

    # Loading the saved checkpoint for the zeroth epoch since the previous script was killed
    checkpoint = torch.load("saved_models/nbmb_model_3.pth")
    nbmb_model.load_state_dict(checkpoint)

    nbmb_model.to(device)

    return nbmb_model, nepBerta_tokenizer, mBart_tokenizer

def run_dev(nbmb_model, nepBerta_tokenizer, mBart_tokenizer):
    dataset = NepaliEnglishDataset(nepBerta_tokenizer, mBart_tokenizer, max_length=512, src_data_path="../dataset/valid_raw/valid.ne_NP", trg_data_path="../dataset/valid_raw/valid.en_XX")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    dev_loss = 0
    nbmb_model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Dev Loop"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, encoder_mask, labels, mbart_input_ids, mbart_input_mask, input_ids_len = batch
            output = nbmb_model(input_ids, encoder_mask, labels, mbart_input_ids, mbart_input_mask, input_ids_len)

            loss = output.loss.sum()
            dev_loss += loss.item()

    dev_loss /= len(dataloader)
    print(f"Dev Loss: {dev_loss}")

    if args.wandb:
        wandb.log({"dev_loss": dev_loss})

    return dev_loss

def finetune():
    nbmb_model, nepBerta_tokenizer, mBart_tokenizer = load_model()

    dataset = NepaliEnglishDataset(nepBerta_tokenizer, mBart_tokenizer, max_length=512, src_data_path="../dataset/train_raw/train.ne_NP", trg_data_path="../dataset/train_raw/train.en_XX")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(nbmb_model.parameters(), lr=args.lr)
    epochs = args.epoch

    for epoch in tqdm(range(epochs), desc="Training Loop"):

        nbmb_model.train()

        train_loss = 0
        for batch in tqdm(dataloader, desc="Batch Splitting Loop"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, encoder_mask, labels, mbart_input_ids, mbart_input_mask, input_ids_len = batch
            output = nbmb_model(input_ids, encoder_mask, labels, mbart_input_ids, mbart_input_mask, input_ids_len)

            optimizer.zero_grad()
            loss = output.loss.sum()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(dataloader)

        if args.wandb:
            wandb.log({"train_loss": train_loss})

        print(f"Epoch: {epoch+4} Train Loss: {train_loss}")

        # save model
        torch.save(nbmb_model.state_dict(), f"saved_models/nbmb_model_{epoch+4}.pth")

        # run dev
        run_dev(nbmb_model, nepBerta_tokenizer, mBart_tokenizer)

        # test
        test(nbmb_model, nepBerta_tokenizer, mBart_tokenizer)

def test(nbmb_model, nepBerta_tokenizer, mBart_tokenizer):
    import string
    def smart_split(s):
        # treat most punctuation as separate words
        spaced_string = ""
        for c in s:
            if c in string.punctuation and c != "'":
                spaced_string += " " + c + " "
            else:
                spaced_string += c
        return spaced_string.split()

    #init generated translations
    generated_translations, target = [], []
    
    test_dataset = NepaliEnglishDataset(nepBerta_tokenizer, mBart_tokenizer, max_length=512, src_data_path="../dataset/test_raw/test.ne_NP", trg_data_path="../dataset/test_raw/test.en_XX", isTest=True)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    nbmb_model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Test Loop"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, labels, mbart_input_ids = batch
            model_translation = nbmb_model.module.generate(input_ids, mbart_input_ids)
            generated_translations.append(model_translation)
            target.append(mBart_tokenizer.batch_decode(labels))

    split_predictions = [smart_split(pred) for pred in generated_translations]
    split_references = [[smart_split(ref) for ref in ref_list] for ref_list in target]

    bleu_score = corpus_bleu(split_references, split_predictions) * 100

    print(f"BLEU Score: {bleu_score}")

    if args.wandb:
        wandb.log({"bleu_score": bleu_score})


def main():
    if args.finetune:
        finetune()
    elif args.test:
        nbmb_model, nepBerta_tokenizer, mBart_tokenizer = load_model()
        test(nbmb_model, nepBerta_tokenizer, mBart_tokenizer)
    else:
        print("Please specify --finetune or --test")

if __name__ == "__main__":

    main()