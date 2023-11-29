# datasets
DIR_PATH = "/workspace"
BATCH_SIZE = 100

# A flag to see whether we are fine-tuning the model or not
fine_tune = True

print('init models')
#initlaize model and tokenizer
model_name = "facebook/mbart-large-50-many-to-many-mmt"
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
tokenizer.src_lang = "ne_NP"
tokenizer.tgt_lang = "en_XX"
print('done')

# #text that says hi my name is trevor in nepali
# nepali_text = "नमस्ते, मेरो नाम त्रेवर हो।"
# #input tokens in nepali created by tokenizer
# input_ids = tokenizer(nepali_text, return_tensors="pt").input_ids
# #find forced beginning of sentence token id
# forced_bos_token_id = tokenizer.lang_code_to_id["en_XX"]
# #generate english translation for "hi my name is trevor"
# outputs = model.generate(input_ids=input_ids, forced_bos_token_id=forced_bos_token_id, max_length=128)
# #decode generated english translation back into a sentence
# english_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

'''
Perform fine-tuning of mBART in case the flag is set appropriately
You can check out references like these to understand the code better:
    - https://colab.research.google.com/drive/1d2mSWLiv93X2hgt9WV8IJFms9dQ6arLn?usp=sharing
    - https://github.com/huggingface/transformers/issues/23185#issuecomment-1537690520
'''
if fine_tune:
    print('fine tuning')
    # Moving the model to CUDA
    model = model.cuda()

    optimizer = AdamW(model.parameters(), lr=1e-4)
    model.train()
    
    # Creating a training batch using torch dataloader
    print('prepping data')
    train_dataset = LanguageDataset(f'{DIR_PATH}/train.ne_NP.txt', f'{DIR_PATH}/train.en_XX.txt', BATCH_SIZE, tokenizer)
    print(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print('done')
    
    num_epochs = 20
    # Fine-tune for the specified number of epochs
    for i in range(num_epochs):
        print(f'epoch {i+1} of {num_epochs}')
        # TODO: See if this is working. Otherwise we can just not use batches
        # TODO: For loop not needed
        for batch in train_loader:
            model_inputs = {k: v.to('cuda') for k, v in batch.items()}
            labels = model_inputs['decoder_input_ids']
            optimizer.zero_grad()
            # are args being passed correct???
            output = model(**model_inputs, labels=labels) # forward pass
            loss = output.loss
            loss.backward() # Backward pass

# Test
#get nepali sentences and english references
dataset_np = load_dataset("text", data_files= {"train": f"{DIR_PATH}/train.ne_NP.txt", "test": f"{DIR_PATH}/test.ne_NP.txt"})
dataset_en = load_dataset("text", data_files={"train": f"{DIR_PATH}/train.en_XX.txt", "test": f"{DIR_PATH}/test.en_XX.txt"})
nepali_sentences = dataset_np["test"]
english_references = dataset_en["test"]

#init generated translations
generated_translations = []

#for each nepali sentence, generate english translation, and add to generated translations
for nepali_sentence in nepali_sentences:
    #input tokens in nepali created by tokenizer
    input_ids = tokenizer(nepali_sentence['text'], return_tensors="pt").input_ids.to('cuda')

    #find forced beginning of sentence token id
    forced_bos_token_id = tokenizer.lang_code_to_id["en_XX"]

    #generate english translation
    outputs = model.generate(input_ids=input_ids, forced_bos_token_id=forced_bos_token_id, max_length=BATCH_SIZE)

    #decode generated english translation back into a sentence
    english_translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # add english sentence to generated translations
    generated_translations.append(english_translation)

#calculate bleu score
# map each sentence to a [ sentence.split() ]
N = 0
if N:
    references = [[reference.split()] for reference in english_references["text"][:N]]
else:
    references = [[reference.split()] for reference in english_references["text"]]
hypotheses = [gen.split() for gen in generated_translations]
bleu_score = corpus_bleu(references, hypotheses)

#print bleu score
print("bleu score: ", bleu_score)
