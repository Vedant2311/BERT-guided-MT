from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from NBmB import NBmB
import torch
from customDataset import NepaliEnglishDataset
from tqdm import tqdm
import re
from nltk.translate.bleu_score import corpus_bleu
from collections import OrderedDict

device = "cuda"
print(device)

def test(nbmt5_model, nepBerta_tokenizer, mT5_tokenizer):
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

    split_predictions = [smart_split(pred) for pred in generated_translations]
    split_references = [[smart_split(ref) for ref in ref_list] for ref_list in target]
    bleu_score = corpus_bleu(split_references, split_predictions) * 100

    print(f"BLEU Score: {bleu_score}")

mBart_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", output_hidden_states=True)
mBart_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="ne_NP", tgt_lang="en_XX")

nepBerta_config = AutoConfig.from_pretrained("NepBERTa/NepBERTa", output_hidden_states=True) # typo in their config 'state' => 'states'
nepBerta_model = AutoModelForMaskedLM.from_pretrained("NepBERTa/NepBERTa", from_tf=True, config=nepBerta_config)

nepBerta_tokenizer = AutoTokenizer.from_pretrained("NepBERTa/NepBERTa")
nepBerta_tokenizer.model_max_length = 512

nbmb_model = NBmB(mBart_model, nepBerta_model, mBart_tokenizer)

# Loading the saved checkpoint for the zeroth epoch since the previous script was killed
checkpoint = torch.load("saved_models_pretrained/nbmb_model_3.pth", map_location=device)
checkpoint = OrderedDict((key.replace('module.', ''), value) for key, value in checkpoint.items())
nbmb_model.load_state_dict(checkpoint)

nbmb_model.to(device)

test(nbmb_model, nepBerta_tokenizer, mBart_tokenizer)