
from datasets import load_dataset

raw_datasets = load_dataset("glue" , "mrpc")


# train_set = raw_datasets["train"]
# validation_set = raw_datasets["validation"]


from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenizing_func(main_dict):
    return tokenizer(main_dict["sentence1"], main_dict["sentence2"],  truncation = True)

tokenized_dataset = raw_datasets.map(tokenizing_func , batched= True)

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

import evaluate
import numpy as np

metric = evaluate.load("glue" , "mrpc")# we load in the metrics for this dataset

def compute_metrics(eval_preds):
    logits,labels = eval_preds
    predictions = np.argmax(logits, axis = -1)#converts logits into understandable probabilities
    return metric.compute(predictions=predictions , references=labels)#returns the f1 score and accuracy


from transformers import TrainingArguments

training_args = TrainingArguments(
    "test-trainer", 
    eval_strategy = "epoch" ,
    save_strategy="epoch" , 
    fp16=True,
    overwrite_output_dir=True, 
    load_best_model_at_end=True , 
    metric_for_best_model= "f1")#gives you the metric after each epoch

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = 2)

from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model("fine-tuning-trial1")
tokenizer.save_pretrained("fine-tuning-trial1")