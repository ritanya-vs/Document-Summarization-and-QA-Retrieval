import streamlit as st
from modules.summarizer_file import summarize_text  # Import summarize function
from modules import speech_to_text, qa_retrieval, text_to_speech
from utils.audio_utils import record_audio

# Set up Streamlit page config
st.set_page_config(page_title="Voice AI Assistant")

# Title for the app
st.title("🎙 Real-time Voice-based Q&A Assistant")

# Step 1: User Input Selection (Audio or Text)
input_type = st.radio("Choose input method", ["Audio", "Text"])

if input_type == "Audio":
    # Audio-based input
    if st.button("Start Recording"):
        audio_path = record_audio("user_input.wav")
        st.success("Audio recorded!")

        # Step 2: Speech-to-Text conversion
        transcribed_text = speech_to_text.transcribe(audio_path)
        st.subheader("📝 Transcription")
        st.write(transcribed_text)

elif input_type == "Text":
    # Text-based input
    transcribed_text = st.text_area("Enter your text here:")
    if transcribed_text:
        st.subheader("📝 Text Input")
        st.write(transcribed_text)

# Step 3: Summarize the input text (optional, if you want to show summary)
if transcribed_text:
    summary = summarize_text(transcribed_text)
    st.subheader("📄 Summary (Optional)")
    st.write(summary)

    # Step 4: Ask a question related to the original text (not the summary)
    question = st.text_input("Ask a question related to the original text:")
    if question:
        # Use the original transcribed_text for QA retrieval
        answer = qa_retrieval.answer_question(question, transcribed_text)
        st.subheader("❓ Answer")
        st.write(answer)

        # Step 5: Text-to-Speech for the answer
        if st.button("🔊 Read Out Loud"):
            text_to_speech.speak(answer)
