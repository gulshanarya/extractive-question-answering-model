
import gradio as gr
import tensorflow as tf
from transformers import TFAutoModelForQuestionAnswering, AutoTokenizer

model = TFAutoModelForQuestionAnswering.from_pretrained("iamgulshan/bert-qa-squad")
tokenizer = AutoTokenizer.from_pretrained("iamgulshan/bert-qa-squad")

def qa_model(question, context):
    inputs = tokenizer(question, context, return_tensors="tf")
    inputs.pop("token_type_ids", None)  # Remove token_type_ids if present
    outputs = model(**inputs)
  
    answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
    
    
    answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])
    start_prob = tf.nn.softmax(outputs.start_logits, axis=-1)[0][answer_start_index]
    end_prob = tf.nn.softmax(outputs.end_logits, axis=-1)[0][answer_end_index]

    avg_prob = (start_prob + end_prob) / 2

    if avg_prob >= 0.5:
      predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
      return tokenizer.decode(predict_answer_tokens)
    else:
      return "Sorry! I am not sure about the answer :)"


iface = gr.Interface(
    fn=qa_model,
    inputs=["text", "text"],
    outputs="text",
    title="Question Answering Chatbot",
    description="Ask a question and provide a context to get an answer.",
)

iface.launch()