import sounddevice as sd
from scipy.io.wavfile import write
import streamlit as st

def record_audio(filename="output.wav", duration=5, fs=44100):
    st.info("Recording... Speak now!")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)
    return filename
