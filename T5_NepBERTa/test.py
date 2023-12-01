from NBmT5 import NBmT5
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from transformers import MT5ForConditionalGeneration, T5Tokenizer
import torch
from customDataset import NepaliEnglishDataset
from tqdm import tqdm
import re
from nltk.translate.bleu_score import corpus_bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def test(nbmt5_model, nepBerta_tokenizer, mT5_tokenizer):
    #init generated translations
    generated_translations, target = [], []
    
    test_dataset = NepaliEnglishDataset(nepBerta_tokenizer, mT5_tokenizer, max_length=512, src_data_path="../dataset/train_raw/train.ne_NP", trg_data_path="../dataset/train_raw/train.en_XX", isTest=True)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    nbmt5_model.eval()
    i = 0
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Test Loop"):
            if i > 100:
                break
            i += 1
            batch = tuple(t.to(device) for t in batch if t is not None)
            input_ids, labels, mT5_input_ids = batch
            model_translation = nbmt5_model.generate(input_ids, mT5_input_ids)
            generated_translations.append(model_translation)
            print(i, model_translation)
            target.append(mT5_tokenizer.batch_decode(labels))

    # A post-processing step to remove the unnecessary tokens like <str> that occur during generation
    pattern = r'<.*?>'
    generated_translations = [re.sub(pattern, '', input_string[0]) for input_string in generated_translations]
    target = [re.sub(pattern, '', input_string[0]) for input_string in target]

    hypotheses = [gen.split() for gen in generated_translations]
    references = [[ref.split()] for ref in target]

    print(hypotheses[30:40])
    print(references[30:40])
    bleu_score = corpus_bleu(references, hypotheses) * 100

    print(f"BLEU Score: {bleu_score}")


mT5_tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")
mT5_model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")

nepBerta_config = AutoConfig.from_pretrained("NepBERTa/NepBERTa", output_hidden_states=True) # typo in their config 'state' => 'states'
nepBerta_model = AutoModelForMaskedLM.from_pretrained("NepBERTa/NepBERTa", from_tf=True, config=nepBerta_config)

nepBerta_tokenizer = AutoTokenizer.from_pretrained("NepBERTa/NepBERTa")
nepBerta_tokenizer.model_max_length = 512

nbmt5_model = NBmT5(mT5_model, nepBerta_model, mT5_tokenizer)

checkpoint = torch.load("saved_models/nbmt5_model_1.pth")
nbmt5_model.load_state_dict(checkpoint)

nbmt5_model.eval()
nbmt5_model.to(device)

test(nbmt5_model, nepBerta_tokenizer, mT5_tokenizer)