import streamlit as st
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
import io

class TextToSpeech:
    def __init__(self):
        try:
            self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
            self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
            self.speaker_embeddings = self.load_speaker_embeddings()
        except Exception as e:
            print(f"Error loading models or processor: {e}")

    def load_speaker_embeddings(self):
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        return torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    def generate_speech(self, text):
        inputs = self.processor(text=text, return_tensors="pt")
        speech = self.model.generate_speech(inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder)
        return speech

    def convert_text_file_to_speech(self, uploaded_file):
        if uploaded_file is not None:
            text = uploaded_file.read().decode("utf-8")
            return self.generate_speech(text)
        return None

class StreamlitApp:
    def __init__(self):
        self.tts = TextToSpeech()

    def run(self):
        st.title("Text to Speech with SpeechT5")

        text_input = st.text_area("Enter text:", "Hello, my dog is cute.")
        if st.button("Generate Speech"):
            speech = self.tts.generate_speech(text_input)
            self.display_audio(speech)

        st.subheader("Or upload a text file:")
        uploaded_file = st.file_uploader("Choose a text file", type="txt")
        if uploaded_file is not None:
            speech = self.tts.convert_text_file_to_speech(uploaded_file)
            self.display_audio(speech)

    def display_audio(self, speech):
        if speech is not None:
            audio_bytes = io.BytesIO()
            sf.write(audio_bytes, speech.squeeze().cpu().numpy(), samplerate=16000, format='wav')
            audio_bytes.seek(0)
            st.audio(audio_bytes, format='audio/wav')

app = StreamlitApp()
app.run()
