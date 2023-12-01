from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from NBmT5 import NBmT5
import argparse
import torch
from customDataset import NepaliEnglishDataset
from tqdm import tqdm
import wandb
from nltk.translate.bleu_score import corpus_bleu
import re

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--finetune", "-f", action="store_true")
parser.add_argument("--test", "-t", action="store_true")
parser.add_argument("--epoch", "-e", type=int, default=20)
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
    mT5_tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")
    mT5_model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")

    nepBerta_config = AutoConfig.from_pretrained("NepBERTa/NepBERTa", output_hidden_states=True) # typo in their config 'state' => 'states'
    nepBerta_model = AutoModelForMaskedLM.from_pretrained("NepBERTa/NepBERTa", from_tf=True, config=nepBerta_config)

    nepBerta_tokenizer = AutoTokenizer.from_pretrained("NepBERTa/NepBERTa")
    nepBerta_tokenizer.model_max_length = 512

    nbmt5_model = NBmT5(mT5_model, nepBerta_model, mT5_tokenizer)

    num_device = torch.cuda.device_count()
    if num_device > 1:
        nbmt5_model = torch.nn.DataParallel(nbmt5_model)
        print("Using DataParallel for NBmT5")

    nbmt5_model.to(device)

    return nbmt5_model, nepBerta_tokenizer, mT5_tokenizer

def run_dev(nbmt5_model, nepBerta_tokenizer, mT5_tokenizer):
    dataset = NepaliEnglishDataset(nepBerta_tokenizer, mT5_tokenizer, max_length=512, src_data_path="../dataset/valid_raw/valid.ne_NP", trg_data_path="../dataset/valid_raw/valid.en_XX")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    dev_loss = 0
    nbmt5_model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Dev Loop"):
            input_ids, encoder_mask, labels, mT5_input_ids, mT5_input_mask, input_ids_len = batch
            output = nbmt5_model(input_ids, encoder_mask, labels, mT5_input_ids, mT5_input_mask, input_ids_len)

            loss = output.loss.sum()
            dev_loss += loss.item()

    dev_loss /= len(dataloader)
    print(f"Dev Loss: {dev_loss}")

    if args.wandb:
        wandb.log({"dev_loss": dev_loss})

    return dev_loss

def finetune():
    nbmt5_model, nepBerta_tokenizer, mT5_tokenizer = load_model()

    dataset = NepaliEnglishDataset(nepBerta_tokenizer, mT5_tokenizer, max_length=512, src_data_path="../dataset/train_raw/train.ne_NP", trg_data_path="../dataset/train_raw/train.en_XX")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(nbmt5_model.parameters(), lr=args.lr)
    epochs = args.epoch

    for epoch in tqdm(range(epochs), desc="Training Loop"):

        nbmt5_model.train()

        train_loss = 0
        for batch in tqdm(dataloader, desc="Batch Splitting Loop"):
            input_ids, encoder_mask, labels, mT5_input_ids, mT5_input_mask, input_ids_len = batch
            output = nbmt5_model(input_ids, encoder_mask, labels, mT5_input_ids, mT5_input_mask, input_ids_len)

            optimizer.zero_grad()
            loss = output.loss.sum()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(dataloader)

        if args.wandb:
            wandb.log({"train_loss": train_loss})

        print(f"Epoch: {epoch+1} Train Loss: {train_loss}")

        # save model
        torch.save(nbmt5_model.state_dict(), f"saved_models/nbmt5_model_{epoch}.pth")

        # run dev
        run_dev(nbmt5_model, nepBerta_tokenizer, mT5_tokenizer)

        # test
        test(nbmt5_model, nepBerta_tokenizer, mT5_tokenizer)

def test(nbmt5_model, nepBerta_tokenizer, mT5_tokenizer):
    #init generated translations
    generated_translations, target = [], []
    
    test_dataset = NepaliEnglishDataset(nepBerta_tokenizer, mT5_tokenizer, max_length=512, src_data_path="../dataset/test_raw/test.ne_NP", trg_data_path="../dataset/test_raw/test.en_XX", isTest=True)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    nbmt5_model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Test Loop"):
            batch = tuple(t.to(device) for t in batch if t is not None)
            input_ids, labels, mT5_input_ids = batch
            model_translation = nbmt5_model.generate(input_ids, mT5_input_ids)
            generated_translations.append(model_translation)
            target.append(mT5_tokenizer.batch_decode(labels))

    # A post-processing step to remove the unnecessary tokens like <str> that occur during generation
    pattern = r'<.*?>'
    generated_translations = [re.sub(pattern, '', input_string) for input_string in generated_translations]
    target = [re.sub(pattern, '', input_string) for input_string in target]

    bleu_score = corpus_bleu([[t] for t in target], generated_translations) * 100

    print(f"BLEU Score: {bleu_score}")

    if args.wandb:
        wandb.log({"bleu_score": bleu_score})


def main():
    if args.finetune:
        finetune()
    elif args.test:
        nbmt5_model, nepBerta_tokenizer, mT5_tokenizer = load_model()
        test(nbmt5_model, nepBerta_tokenizer, mT5_tokenizer)
    else:
        print("Please specify --finetune or --test")

if __name__ == "__main__":
    main()