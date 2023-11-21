from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers.optimization import AdamW
from datasets import load_dataset
from nltk.translate.bleu_score import corpus_bleu
import torch

# A flag to see whether we are fine-tuning the model or not
fine_tune = False

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

#get a dataset from dataloader we create.
#TODO look into how can we load the dataset from the individual separate raw data files for Nepali and English
dataset = load_dataset("csv", data_files={"train": "/path/to/train.csv", "test": "/path/to/test.csv", "valid": "/path/to/valid.csv"})

'''
Perform fine-tuning of mBART in case the flag is set appropriately
You can check out references like these to understand the code better:
    - https://colab.research.google.com/drive/1d2mSWLiv93X2hgt9WV8IJFms9dQ6arLn?usp=sharing
    - https://github.com/huggingface/transformers/issues/23185#issuecomment-1537690520
'''
if fine_tune:
    # Moving the model to CUDA
    model = model.cuda()

    # Get the train nepali and english sentences
    nepali_sentences_train = dataset["train"]["nepali"]
    english_references_train = dataset["train"]["english"]

    # Set up the optimizer and training settings
    optimizer = AdamW(model.parameters(), lr=1e-4)
    model.train()

    # Setting the number of epochs and batch-size
    num_epochs = 20
    batch_size = 32

    # Getting the tokens corresponding to all the Input and output sentences and creating a tuple
    train_set = []
    raw_data_train = list(zip(nepali_sentences_train, english_references_train))
    for data in raw_data_train:
        source, target = data
        # Test and check if we would need model_inputs.input_ids or if this should be fine
        model_inputs = tokenizer(source, return_tensors="pt")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(target, return_tensors="pt").input_ids
        train_set.append((model_inputs, labels))

    # Creating a training batch using torch dataloader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Fine-tune for the specified number of epochs
    for i in range(num_epochs):
        for batch in train_loader:
            model_inputs, labels = batch
            model_inputs = model_inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            output = model(**model_inputs, labels=labels) # forward pass
            loss = output.loss
            loss.backward() # Backward pass

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