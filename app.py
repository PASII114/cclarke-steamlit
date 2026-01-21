import streamlit as st
from PIL import Image
from transformers import pipeline

st.set_page_config(page_title="AI Summarization Tool")

@st.cache_resource
def load_summarization_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_captioning_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")


captioning_model = load_captioning_model()
st.title("Image Captioning Tool")
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    caption_generator = st.button("Generate Captions", type="primary")

with col2:
    st.markdown("Powered by Pasindu Rashmika")

if uploaded_image and caption_generator:
    with st.spinner("Generating Captions..."):
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, use_container_width=True)
        result = captioning_model(image)
        gen_text = result[0]["generated_text"]


        st.markdown(gen_text)

elif caption_generator:
    st.warning("Please upload an image first.")
