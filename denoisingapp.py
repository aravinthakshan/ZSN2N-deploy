import streamlit as st
import torch
from PIL import Image
import numpy as np
from model import DenoisingNetwork
from utils import add_noise
from training import denoise

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenoisingNetwork(n_chan=3).to(device)
model.load_state_dict(torch.load('model_weights.pth', map_location=device))

st.title("Image Denoising App")
st.write("Upload an image, and the model will remove the noise.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
noise_level = st.slider("Select Noise Level", 0, 100, 25)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)

    noisy_img = add_noise(image_tensor, noise_level=noise_level)

    denoised_img = denoise(model, noisy_img)

    st.image(noisy_img.cpu().squeeze().permute(1, 2, 0).numpy(), caption="Noisy Image", use_column_width=True)
    st.image(denoised_img.cpu().squeeze().permute(1, 2, 0).numpy(), caption="Denoised Image", use_column_width=True)
