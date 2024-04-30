# pydantic-1.10.7, gradio 3.34 are required

import gradio as gr
import torch
import librosa
import numpy as np
from model import Model

# Initialize constants and model
CKPT = 'models/model_DF_WCE_100_16_1e-06/epoch_49.pth'
EER = 0.183  # Ensure EER is a float for comparison

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Model(None, device)
sd = torch.load(CKPT)
sd = {k.replace('module.', ''): v for k, v in sd.items()}  # Adjust for DataParallel wrap
model.load_state_dict(sd)
model.eval()

model = model.to(device)

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # Padding logic if needed
    num_repeats = (max_len // x_len) + 1
    padded_x = np.tile(x, num_repeats)[:max_len]
    return padded_x

def classify_audio(file_info) -> str:
    filepath = file_info
    print("Processing:", filepath)
    try:
        audio, sr = librosa.load(filepath, sr=16000)  # Ensure the sample rate is 16000 Hz
        audio = pad(audio)
        audio_tensor = torch.Tensor(audio).to(device).unsqueeze(0)  # Add batch dimension

        print('Running model inference...')
        with torch.no_grad():
            out = model(audio_tensor)
        out_score = out[0, 1].item()  # Assuming the second column is the target class

        result = "Real" if out_score >= EER else "Spoof"
        return result + f" (Raw CM score from model: {out_score:.4f})"
    except Exception as e:
        return "Error: " + str(e)

# Interface
demo = gr.Interface(
    fn=classify_audio,
    inputs=gr.Audio(type="filepath"),
    outputs='text'
)

demo.launch(server_port=8000)