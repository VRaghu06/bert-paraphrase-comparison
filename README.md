# BERT Fine-Tuning for Paraphrase Classification

## Overview

This project aims to demonstrate the impact of fine-tuning a pretrained BERT model on the MRPC dataset from the GLUE benchmark.

Our goal is to figure out if two sentences are paraphrases of each other.

Along with the reports of the fine-tuned model's performance, this project also highlights the disparity between results of the fine-tuned model and the pre-trained model, to show that task-specific training can drastically improve a model's performance.

---

## Objective

Given the two sentences:

Sentence 1:
```
The company has launched a new product.
```

Sentence 2:
```
A new product was unveiled by the company.
```

Predict if they are semantically similar (paraphrases).

Sample Output:
```python
label:Label_1
score:0.99501
```
Note: label_1 indicates paraphrase, and label_0 indicates not paraphrase, according to the MRPC labelling scheme.

---
## Fine-Tuning Pipeline

1.Load the GLUE MRPC dataset.

2.Create a tokenizing function (tokenizing_func) in this case, that will get applied to every element of the dataset, through a .map() function.

3.Dynamically pad the tokenized dataset using DataCollatorWithPadding.

4.Create the compute_metrics function, which decides what metrics will be reported during the training and evaluation.

5.Create a trainer object and pass in the necessary parameters.

6.Call the .train() method from the Trainer API

---

## Why does the base model underperform ?

Although BERT is a pretrained model, the sequence classification head is **randomly initialized**, which essentially means that it has simply not been trained for this task, and therefore it's head is initialized with random values.

As a result, the predictions are effectively random.

```python
AutoModelForSequenceClassification(model='bert-base-uncased' , num_labels = 2)# the num_labels=2 is what indicates that the head is randomly initialized
```

---

## Results

| Model | Accuracy | F1 Score |
|--------|---------:|---------:|
| Pretrained BERT (No Fine-Tuning) | **0.3162** | **0.00** |
| Fine-Tuned BERT | **0.8676** | **0.9078** |

The large improvement indicates how fine-tuning can enable a pre-trained transformer to adapt to a downstream NLP task.

---

## Project Structure

```
.
├── paraphrase-fine-tuning.py              # Fine-tuning script
├── paraphrase-tester-fineTuned.py         # Fine-tuned model evaluation
├── paraphrase-tester-preTrained.py        # Base model evaluation
├── requirements.txt
└── README.md
```

---

## Tech Stack

- Hugging Face Transformers
- Hugging Face datasets
- Trainer API
- evaluate
- AutoTokenizer
- AutoModelForSequenceClassification
- scikit-learn
- NumPy
- PyTorch

---

## Key Concepts

- Transformer Fine-tuning
- Tokenizing a dataset
- Dynamic Padding
- Sequence Classification
- Evaluation Metrics
- Accuracy and F1 score
- Hugging Face Trainer API

---

## Installation

```bash
git clone <repository-url>
cd bert-paraphrase-comparison

pip install -r requirements.txt
```




