import streamlit as st
import torch
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
from langdetect import detect

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1

# Load English summarizer (Facebook BART)
@st.cache_resource
def load_english_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", device=device)

# Load Hindi summarizer (IndicBART fine-tuned for Indian languages)
@st.cache_resource
def load_hindi_summarizer():
    model_name = "ai4bharat/IndicBART"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name,use_auth_token=st.secrets["HUGGINGFACE_TOKEN"])
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading Hindi summarization model: {e}\n"
                 "Please verify your internet connection or ensure the model is available on Hugging Face.")
        return None, None

# Load translation pipeline for English-to-Hindi translation
@st.cache_resource
def load_translation_pipeline_en_to_hi():
    try:
        translator = pipeline("translation_en_to_hi", model="Helsinki-NLP/opus-mt-en-hi", device=device)
        return translator
    except Exception as e:
        st.error(f"Error loading English-to-Hindi translation model: {e}")
        return None

# Load translation pipeline for Hindi-to-English translation
@st.cache_resource
def load_translation_pipeline_hi_to_en():
    try:
        translator = pipeline("translation_hi_to_en", model="Helsinki-NLP/opus-mt-hi-en", device=device)
        return translator
    except Exception as e:
        st.error(f"Error loading Hindi-to-English translation model: {e}")
        return None

# Function to summarize English text
def english_summary(article):
    # Detect input language; if not English, translate to English first.
    try:
        lang = detect(article)
    except Exception as e:
        return f"Error detecting language: {e}"
    
    if lang != "en":
        translator_hi_en = load_translation_pipeline_hi_to_en()
        if translator_hi_en is None:
            return "Translation model for Hindi-to-English failed to load."
        try:
            translated = translator_hi_en(article, max_length=512)
            article = translated[0]['translation_text']
        except Exception as e:
            return f"Error during translation (Hindi->English): {e}"
    
    summarizer = load_english_summarizer()
    try:
        summary = summarizer(article, max_length=130, min_length=30, do_sample=False)
        if not summary or len(summary) == 0:
            return "No summary generated. Please provide a longer or more detailed article."
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error during English summarization: {e}"

# Function to summarize Hindi text
def hindi_summary(article):
    # Detect input language; if not Hindi, translate to Hindi first.
    try:
        lang = detect(article)
    except Exception as e:
        return f"Error detecting language: {e}"
    
    if lang != "hi":
        translator_en_hi = load_translation_pipeline_en_to_hi()
        if translator_en_hi is None:
            return "Translation model for English-to-Hindi failed to load."
        try:
            translated = translator_en_hi(article, max_length=512)
            article = translated[0]['translation_text']
        except Exception as e:
            return f"Error during translation (English->Hindi): {e}"
    
    tokenizer, model = load_hindi_summarizer()
    if tokenizer is None or model is None:
        return "Hindi model loading failed. Please check the error message above."
    
    # Clean input text
    cleaned_text = re.sub(r'\s+', ' ', re.sub(r'\n+', ' ', article.strip()))
    
    input_ids = tokenizer(
        [cleaned_text],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    ).input_ids
    
    try:
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=150,
            min_length=50,
            no_repeat_ngram_size=2,
            num_beams=4
        )[0]
        summary = tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        return summary
    except Exception as e:
        return f"Error during Hindi summarization: {e}"

# Load ROUGE evaluator (for evaluation metrics)
@st.cache_resource
def load_rouge():
    return evaluate.load("rouge")

# Function to compute ROUGE scores
def compute_rouge(prediction, reference):
    rouge = load_rouge()
    results = rouge.compute(predictions=[prediction], references=[reference])
    return results

# Streamlit UI
st.title("Multi-language Summarizer")
st.markdown("Enter an article and choose the summarization language (English or Hindi).")
st.markdown("Optionally, provide a reference summary to compute evaluation metrics (ROUGE scores).")

article_input = st.text_area("Article Text", height=250)

language = st.selectbox("Select summarization language", options=["English", "Hindi"])

if st.button("Summarize"):
    if article_input.strip() == "":
        st.warning("Please enter an article to summarize!")
    else:
        with st.spinner("Summarizing..."):
            if language == "English":
                generated_summary = english_summary(article_input)
            else:
                generated_summary = hindi_summary(article_input)
        st.subheader("Generated Summary")
        st.write(generated_summary)
