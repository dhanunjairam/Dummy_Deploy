import streamlit as st
import torch
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM,T5Tokenizer

# Function for English summarization using Facebook BART
def english_summary(article):
    # For the pipeline, we set device=-1 to use CPU.
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
    summary = summarizer(article, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Function for Hindi summarization using the mT5 model
def hindi_summary(article):
    # Remove extra whitespace/newlines
    WHITESPACE_HANDLER = lambda k: re.sub(r'\s+', ' ', re.sub(r'\n+', ' ', k.strip()))
    model_name = "csebuetnlp/mT5_m2o_hindi_crossSum"
    tokenizer = AutoTokenizer.from_pretrained(model_name,legacy=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Tokenize and prepare the input
    input_ids = tokenizer(
        [WHITESPACE_HANDLER(article)],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )["input_ids"]
    
    # Generate the summary using beam search and control repetition
    output_ids = model.generate(
        input_ids=input_ids,
        max_length=84,
        no_repeat_ngram_size=2,
        num_beams=4
    )[0]
    
    # Decode the generated tokens
    summary = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return summary

# Streamlit UI
st.title("Multi-language Summarizer")

st.markdown("Enter an article and choose the summarization language (English or Hindi).")

# User input for article text
article_input = st.text_area("Article Text", height=250)

# Language selection
language = st.selectbox("Select summarization language", options=["English", "Hindi"])

# When the user clicks the button, process the summarization
if st.button("Summarize"):
    if article_input.strip() == "":
        st.warning("Please enter an article to summarize!")
    else:
        with st.spinner("Summarizing..."):
            if language == "English":
                summary_text = english_summary(article_input)
            else:
                summary_text = hindi_summary(article_input)
        st.subheader("Summary")
        st.write(summary_text)
