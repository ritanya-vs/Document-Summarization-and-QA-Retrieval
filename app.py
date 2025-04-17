import streamlit as st
from modules.summarizer_file import summarize_text_t5, summarize_text_bart
from modules import qa_retrieval

st.set_page_config(page_title="Document Summarization & QA")

st.title("📝 Text Summarizer & Q&A Assistant")

transcribed_text = st.text_area("Enter your document/text:")

summarizer_model = st.selectbox("Choose summarization model", ["T5", "BART"])

if transcribed_text:
    st.subheader("🗒 Original Text")
    st.write(transcribed_text)

    st.subheader("📄 Summary")
    if summarizer_model == "T5":
        summary = summarize_text_t5(transcribed_text)
    else:
        summary = summarize_text_bart(transcribed_text)
    st.write(summary)

    question = st.text_input("Ask a question based on the original text:")
    if question:
        answer = qa_retrieval.answer_question(question, transcribed_text)
        st.subheader("❓ Answer")
        st.write(answer)
