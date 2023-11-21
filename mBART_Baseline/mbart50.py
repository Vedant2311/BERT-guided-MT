from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from datasets import load_dataset
from nltk.translate.bleu_score import corpus_bleu

#initlaize model and tokenizer
model_name = "facebook/mbart-large-50-many-to-many-mmt"
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

#english and nepali language codes
tokenizer.src_lang = "ne_NP"
tokenizer.tgt_lang = "en_XX"

#text that says hi my name is trevor in nepali
nepali_text = "नमस्ते, मेरो नाम त्रेवर हो।"

#input tokens in nepali created by tokenizer
input_ids = tokenizer(nepali_text, return_tensors="pt").input_ids

#find forced beginning of sentence token id
forced_bos_token_id = tokenizer.lang_code_to_id["en_XX"]

#generate english translation for "hi my name is trevor"
outputs = model.generate(input_ids=input_ids, forced_bos_token_id=forced_bos_token_id,max_length=128)

#decode generated english translation back into a sentence
english_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#print english translation
print(english_text)

#get a dataset from dataloader we create
dataset = load_dataset("csv", data_files={"train": "/path/to/train.csv", "test": "/path/to/test.csv"})

#get nepali sentences and english references
nepali_sentences = dataset["test"]["nepali"]
english_references = dataset["test"]["english"]

#init generated translations
generated_translations = []

#for each nepali sentence, generate english translation, and add to generated translations
for nepali_sentence in nepali_sentences:

    #input tokens in nepali created by tokenizer
    input_ids = tokenizer(nepali_sentence, return_tensors="pt").input_ids

    #find forced beginning of sentence token id
    forced_bos_token_id = tokenizer.lang_code_to_id["en_XX"]

    #generate english translation
    outputs = model.generate(input_ids=input_ids, forced_bos_token_id=forced_bos_token_id, max_length=128)

    #decode generated english translation back into a sentence
    english_translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # add english sentence to generated translations
    generated_translations.append(english_translation)

#calculate bleu score
bleu_score = corpus_bleu([[reference] for reference in english_references], generated_translations)

#print bleu score
print("bleu score: ", bleu_score)