from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model_name = "facebook/mbart-large-50-many-to-many-mmt"
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

tokenizer.src_lang = "ne_NP"
tokenizer.tgt_lang = "en_XX"

nepali_text = "नमस्ते, मेरो नाम त्रेवर हो।"

input_ids = tokenizer(nepali_text, return_tensors="pt").input_ids

forced_bos_token_id = tokenizer.lang_code_to_id["en_XX"]

outputs = model.generate(input_ids=input_ids, forced_bos_token_id=forced_bos_token_id,max_length=128)

english_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(english_text)

