from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import evaluate

metric = evaluate.load("accuracy")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
nouns_dataset = load_dataset("fairnlp/holistic-bias", data_files=["nouns.csv"], split="test")
sentences_dataset = load_dataset("fairnlp/holistic-bias", data_files=["sentences.csv"], split="test")
gpt2_model = AutoModelForSequenceClassification.from_pretrained("openai-community/gpt2", num_labels=13)
training_arguments = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")

# Print the column names to check the keys
print(nouns_dataset.column_names)
print(sentences_dataset.column_names)

'''
Javadocs:
'''
def tokenize_nouns(examples):
     return tokenizer(examples["noun_phrase"], padding="max_length", truncation=True)
     
def tokenize_sentences(examples):
     return tokenizer(examples["text"], padding="max_length", truncation=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

#nouns_tokenized_datasets = nouns_dataset.map(tokenize_nouns, batched=True)
sentences_tokenized_datasets = sentences_dataset.map(tokenize_sentences, batched=True)

#small_nouns_train_dataset = nouns_tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
#small_nouns_eval_dataset = nouns_tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

#small_sentences_train_dataset = sentences_tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_sentences_eval_dataset = sentences_tokenized_datasets["test"].shuffle(seed=42).select(range(1000))


def compute_metrics(eval_pred):
     logits, labels = eval_pred
     predictions = np.argmax(logits, axis=1)
     return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
     model=gpt2_model,
     args=training_arguments,
     train_dataset=small_sentences_eval_dataset,
    # eval_dataset=small_sentences_eval_dataset, 
     compute_metrics=compute_metrics
)

trainer.train()