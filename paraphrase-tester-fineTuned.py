from datasets import load_dataset
from transformers import(
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

import evaluate
import numpy as np

raw_dataset = load_dataset("glue" , "mrpc")

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenizing_func(main_dict):
    return tokenizer(main_dict["sentence1"], main_dict["sentence2"] , truncation = True)

tokenized_dataset = raw_dataset.map(tokenizing_func , batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

metric = evaluate.load("glue" , "mrpc")

def compute_metrics(eval_preds):
    logits,labels = eval_preds
    predictions = np.argmax(logits , axis=-1)
    return metric.compute(predictions=predictions ,  references=labels)

fineTuned_model = AutoModelForSequenceClassification.from_pretrained("fine-tuning-trial1")

eval_args = TrainingArguments(
    output_dir="eval-temp-fineTuned",
    per_device_eval_batch_size=8,
    fp16=True,
)

trainer=Trainer(
    model = fineTuned_model,
    args=eval_args,
    eval_dataset = tokenized_dataset["validation"],
    data_collator=data_collator,
    processing_class = tokenizer,
    compute_metrics = compute_metrics,
)

results = trainer.evaluate()
print(f"Accuracy: {results['eval_accuracy']:.4f}")
print(f"F1: {results['eval_f1']:.4f}")