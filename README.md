# Question Answering Model (Fine-Tuned DistilBERT)

This repository contains a **fine-tuned DistilBERT-based Question Answering (QA) model**, trained on the **SQuAD dataset**. The model achieves **70% validation accuracy** and is designed to extract precise answers from a given context. While there is room for improvement, this serves as a strong baseline for further fine-tuning.

## Features
✅ **Transformer-based QA model** using Hugging Face's `TFAutoModelForQuestionAnswering`  
✅ **Fine-tuned on SQuAD dataset** for extractive question answering  
✅ **Achieves 70% validation accuracy**, with scope for improvement  
✅ **Efficient tokenization & batching** using `tokenizer` and TensorFlow datasets  
✅ **Deployable via API or web app** (Gradio, Streamlit, or Flask)  

## Model Overview
- **Base Model:** `distilbert-base-cased` (pretrained by Hugging Face)
- **Training Dataset:** SQuAD (Stanford Question Answering Dataset)
- **Loss Function:** Sparse Categorical Crossentropy
- **Optimizer:** AdamW
- **Evaluation Metric:** F1-score

## Installation
To use this project, install dependencies using:

```bash
pip install transformers tensorflow datasets gradio
```

## Usage
### **Inference Example**
```python
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering
import tensorflow as tf

tokenizer = AutoTokenizer.from_pretrained("path/to/saved_tokenizer")
model = TFAutoModelForQuestionAnswering.from_pretrained("path/to/saved_model")

question = "What are the challenges faced by Apollo 11?"
context = "Apollo 11 faced several challenges, including navigation errors, an overloaded computer, and a nearly failed ascent due to a broken switch."

inputs = tokenizer(question, context, return_tensors="tf")
outputs = model(**inputs)

start_scores, end_scores = outputs.start_logits, outputs.end_logits
start_idx = tf.argmax(start_scores, axis=-1).numpy()[0]
end_idx = tf.argmax(end_scores, axis=-1).numpy()[0]

answer = tokenizer.decode(inputs['input_ids'][0][start_idx:end_idx+1])
print("Answer:", answer)
```

## Training
To fine-tune the model:
```Execute the code of qa_finetune.ipynb
```

## Deployment
You can deploy the model using Gradio or Flask:

```bash
python app.py  # Flask API
```

or with Gradio:
```python
import gradio as gr

def answer_question(question, context):
    inputs = tokenizer(question, context, return_tensors="tf")
    outputs = model(**inputs)
    start_scores, end_scores = outputs.start_logits, outputs.end_logits
    start_idx = tf.argmax(start_scores, axis=-1).numpy()[0]
    end_idx = tf.argmax(end_scores, axis=-1).numpy()[0]
    answer = tokenizer.decode(inputs['input_ids'][0][start_idx:end_idx+1])
    return answer

demo = gr.Interface(fn=answer_question, inputs=["text", "text"], outputs="text")
demo.launch()
```

## Future Improvements
- **Increase accuracy** with better data augmentation and hyperparameter tuning
- **Try alternative models** (BERT, RoBERTa, or GPT-based models)
- **Deploy with a scalable API** using FastAPI or Hugging Face Spaces

## License
This project is licensed under the **Apache 2.0 License**.

## Acknowledgments
- Hugging Face's `transformers` library
- TensorFlow for model training
- SQuAD dataset

---
### 🔥 This model is a **work in progress**, and I plan to improve it incrementally. Contributions & suggestions are welcome!

