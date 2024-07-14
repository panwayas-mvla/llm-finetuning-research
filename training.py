from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
nouns_dataset = load_dataset("fairnlp/holistic-bias", data_files=["nouns.csv"], split="train")
sentences_dataset = load_dataset("fairnlp/holistic-bias", data_files=["sentences.csv"], split="train")
gpt2_model = AutoModelForSequenceClassification.from_pretrained("openai-community/gpt2", num_labels=13)
training_arguments = TrainingArguments(output_dir="test_trainer")


'''
Javadocs:
'''
def tokenize_function(examples):
     return tokenizer(examples["text"], padding="max_length", truncation=True)

nouns_tokenized_datasets = nouns_dataset.map(tokenize_function, batched=True)
sentences_tokenized_datasets = sentences_dataset.map(tokenize_function, batched=True)