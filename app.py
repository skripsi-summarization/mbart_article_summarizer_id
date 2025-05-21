# mBART app.py

import streamlit as st
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from newspaper import Article
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = MBartForConditionalGeneration.from_pretrained("skripsi-summarization-1234/mbart-large-50-finetuned-xlsum-summarization").to(device)
tokenizer = MBart50TokenizerFast.from_pretrained("skripsi-summarization-1234/mbart-large-50-finetuned-xlsum-summarization")

# Streamlit layout improvements
st.set_page_config(page_title="mBART News Summarizer", layout="centered")
st.title("📰 mBART News Summarizer 🇮🇩")
st.markdown("""
This app summarizes **Indonesian news articles** using a fine-tuned [mBART](https://huggingface.co/skripsi-summarization-1234/mbart-large-50-finetuned-xlsum-summarization) multilingual model.

**How it works:** Direct Indonesian summarization with mBART.
""")

url = st.text_input("📎 Paste the URL of the Indonesian news article:")

if url:
    try:
        with st.spinner("📥 Downloading and parsing article..."):
            article = Article(url)
            article.download()
            article.parse()
            text = article.text

        st.subheader("📰 Original Article")
        st.write(text)

        with st.spinner("🤖 Summarizing with mBART..."):
            tokenizer.src_lang = "id_XX"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="longest")
            summary_ids = model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True,
                forced_bos_token_id=tokenizer.lang_code_to_id["id_XX"]
            )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        st.success("✅ Summary generated successfully!")
        st.subheader("🔎 Ringkasan Berita")
        st.write(summary)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
