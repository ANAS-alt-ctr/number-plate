import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Set page configuration
st.set_page_config(
    page_title="License Plate Detector",
    page_icon="🚗",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
    }
    .title {
        text-align: center;
        color: #333;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .uploaded-img {
        border: 2px solid #ddd;
        border-radius: 8px;
        padding: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Application Title
st.markdown('<h1 class="title">🚗 License Plate Detector</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666;">Upload an image to detect license plates using YOLOv11</p>', unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    try:
        model = YOLO('best.pt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model:
    # File Uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        try:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_container_width=True, clamp=True)
            
            # Convert to numpy array for opencv/yolo
            img_array = np.array(image)
            
            # Run detection when button is clicked or automatically
            with st.spinner('Detecting...'):
                results = model(image)
                
                # Visualize results
                res_plotted = results[0].plot()  # Returns BGR numpy array
                res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                res_plotted_pil = Image.fromarray(res_plotted_rgb)

                st.success("Detection Complete!")
                st.image(res_plotted_pil, caption='Detected License Plates', use_container_width=True)
                
                # Optional: Display detected text (if OCR was integrated, but here just boxes)
                # st.write(results[0].boxes) -- raw boxes info if needed

        except Exception as e:
            st.error(f"Error processing image: {e}")
else:
    st.warning("Model 'best.pt' not found. Please ensure it is in the same directory.")
