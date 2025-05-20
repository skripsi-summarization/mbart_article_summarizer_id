import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from newspaper import Article
import torch

model = AutoModelForSeq2SeqLM.from_pretrained("model")
tokenizer = AutoTokenizer.from_pretrained("tokenizer")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

st.title("mBART Indonesian Article Summarizer 🇮🇩")
st.write("Summarizing Indonesian news articles using mBART.")

url = st.text_input("Paste the URL of the news article:")

if url:
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text

        st.subheader("Original Article")
        st.write(text)

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="longest").to(device)
        summary_ids = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        st.subheader("Summary")
        st.write(summary)

    except Exception as e:
        st.error(f"Error: {str(e)}")
