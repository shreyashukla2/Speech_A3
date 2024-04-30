import gradio as gr
import torch
from model import Model
import librosa
import numpy as np


## define model and preproc here
CKPT = ''
EER = ''


device = 'cuda' if torch.cuda.is_available() else 'cpu'    
model = Model(None, device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()


################################

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

@torch.no_grad()
def classify_audio(filepath):
    audio = librosa.load(filepath, sr=16000)
    audio = pad(audio, 64600)
    audio = torch.Tensor(audio, device=device).unsqueeze(0)

    out = model(audio)
    out = out[:, 1].data.cpu().numpy().ravel()[0]

    outputs = {
        'spoof': out,
        'real': 1-out
    }

    return outputs


demo = gr.Interface(
    fn=classify_audio, inputs=gr.Audio(type="filepath"), outputs=gr.outputs.Label()
)
demo.launch(debug=True)