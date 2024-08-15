import streamlit as st
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
import io
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

try:
    # Load the models and processor
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    
    # Load xvector embeddings dataset
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
except Exception as e:
    logging.error(f"Error loading models or datasets: {e}")
    st.error("An error occurred while loading models or datasets. Please check the logs.")
    st.stop()

# Streamlit UI
st.title("Text to Speech with SpeechT5")
text_input = st.text_area("Enter text:", "Hello, my dog is cute.")

if st.button("Generate Speech"):
    try:
        inputs = processor(text=text_input, return_tensors="pt", padding=True)
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
        
        # Save the generated speech to an in-memory file
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, speech.squeeze().cpu().numpy(), samplerate=16000, format='wav')
        audio_bytes.seek(0)

        # Display the audio file
        st.audio(audio_bytes, format='audio/wav')
    except Exception as e:
        logging.error(f"Error generating speech: {e}")
        st.error("An error occurred while generating speech. Please check the logs.")
