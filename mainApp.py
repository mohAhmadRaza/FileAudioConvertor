import streamlit as st
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
import io
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

class TextToSpeech:
    def __init__(self):
        try:
            self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
            self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
            self.speaker_embeddings = self.load_speaker_embeddings()
        except Exception as e:
            logging.error(f"Error loading models or processor: {e}")

    def load_speaker_embeddings(self):
        try:
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            return torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
        except Exception as e:
            logging.error(f"Error loading speaker embeddings: {e}")
            return None

    def generate_speech(self, text):
        try:
            if not self.processor or not self.model or not self.vocoder:
                raise ValueError("Processor or model not loaded correctly.")
            
            # Process text
            inputs = self.processor(text=text, return_tensors="pt", padding=True)
            logging.debug(f"Inputs: {inputs}")
            
            # Generate speech
            speech = self.model.generate_speech(inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder)
            logging.debug(f"Speech: {speech}")
            return speech
        except Exception as e:
            logging.error(f"Error generating speech: {e}")
            return None

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
            try:
                audio_bytes = io.BytesIO()
                # Assuming speech tensor needs to be converted to numpy array
                sf.write(audio_bytes, speech.squeeze().cpu().numpy(), samplerate=16000, format='wav')
                audio_bytes.seek(0)
                st.audio(audio_bytes, format='audio/wav')
            except Exception as e:
                logging.error(f"Error displaying audio: {e}")

app = StreamlitApp()
app.run()
