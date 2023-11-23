from datasets import Dataset
import numpy as np
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq #, AdamW
import evaluate

# Load pre-trained model and tokenizer
model_name = "google/mt5-base"
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

# Prepare dataset
prefix = "translate Nepali to English: "
src_lang = "ne_NP"
tgt_lang = "en_XX"
with open("../dataset/train_raw/train.ne_NP", "r") as f:
    train_ne = [l.strip() for l in f.readlines()]
with open("../dataset/train_raw/train.en_XX", "r") as f:
    train_en = [l.strip() for l in f.readlines()]
with open("../dataset/test_raw/test.ne_NP", "r") as f:
    test_ne = [l.strip() for l in f.readlines()]
with open("../dataset/test_raw/test.en_XX", "r") as f:
    test_en = [l.strip() for l in f.readlines()]
train_dataset_dict = {
    "ne_NP": train_ne,
    "en_XX": train_en
}
test_dataset = {
    "ne_NP": test_ne,
    "en_XX": test_en
}
train_dataset = Dataset.from_dict(train_dataset_dict)
test_dataset = Dataset.from_dict(test_dataset)

def preprocess_function(examples):
    return tokenizer([prefix + example for example in examples[src_lang]], text_target=examples[tgt_lang], max_length=128, truncation=True)

tokenized_train_inputs = train_dataset.map(preprocess_function, batched=True, remove_columns=[src_lang, tgt_lang])
tokenized_test_inputs = test_dataset.map(preprocess_function, batched=True, remove_columns=[src_lang, tgt_lang])

# Set up collator and metrics
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)
metric = evaluate.load("bleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

# Train
training_args = Seq2SeqTrainingArguments(
    output_dir="mt5_ne_en",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=2,
    predict_with_generate=True,
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_inputs,
    eval_dataset=tokenized_test_inputs,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("mt5_ne_en_final")