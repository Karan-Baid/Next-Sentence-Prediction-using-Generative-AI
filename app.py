from transformers import pipeline, set_seed
import streamlit as st

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",            
    low_cpu_mem_usage=True      
)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
set_seed(42)
st.title("Next Sentence Prediction using Generative AI")
st.write("Enter a sentence to predict the most likely next sentence.")

input_text = st.text_input("Enter your sentence here:")

if input_text:
    st.subheader("Generated Sentences:")
    outputs = generator(input_text, max_length=50, num_return_sequences=3)
    for i, output in enumerate(outputs):
        st.write(f"{i+1}. {output['generated_text']}")
