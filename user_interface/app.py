import streamlit as st
from PIL import Image
import torch
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import torchvision.transforms.functional as F

# import functions
from EfficientNetClass import EfficientNetV2MOptimized
from process_image import process_uploaded_image, preprocess_for_classification
from waste_information import waste_disposal_info

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

## Load the model ##
def load_model(model_path):
    model = EfficientNetV2MOptimized(n_classes=8)
    model_state = torch.load(model_path, map_location=device)
    
    if isinstance(model_state, EfficientNetV2MOptimized):
        model = model_state
    else:
        model.load_state_dict(model_state)
    
    model.eval()  # Set to evaluation mode
    model.to(device)
    return model

# Load the model once when the app starts
@st.cache_resource
def get_model():
    return load_model("efficientnetv2-method5-RGB.pth")

# Convert image to base64 for HTML display
def image_to_base64(img):
    # Convert torch tensor to PIL Image if needed
    if not isinstance(img, Image.Image):
        img = F.to_pil_image(img)
    
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    img_bytes = buffer.getvalue()
    base64_str = base64.b64encode(img_bytes).decode()
    return base64_str

def main():
    # Set page styling
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background-color: #073832;
        }
        [data-testid="stHeader"] {
            background-color: #073832;
        }
        .custom-title {
            font-size: 30px;
            font-weight: bold;
            margin-bottom: 10px;
            color: white;
        }
        .custom-heading {
            font-size: 20px;
            color: #7cd9c0;
        }
        .custom-font {
            font-size: 16px;
            color: white;
        }
        .prediction-text {
            font-size: 30px;
            font-weight: bold;
            color: #7cd9c0;
        }
        .confidence-text {
            font-size: 18px;
            font-style: italic;
            color: #e7c4ff;
        }
        .centered-container {
            display: flex;
            justify-content: center;
            margin-bottom: 25px; /* Add margin below the image container */
        }
        .responsive-image {
            max-width: 50%;
            height: auto;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Page header
    st.markdown(
        '<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="custom-title">'
        'Waste Classification System 🚮'
        '</p>',
        unsafe_allow_html=True
    )
    
    st.markdown(
        '<p class="custom-font">'
        'Upload an image of your waste item!  '
        '</p>',
        unsafe_allow_html=True
    )

    # Get model
    classifier = get_model()

    # File uploader
    uploaded_image = st.file_uploader(label="None", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

    if uploaded_image is not None:
        # Open the uploaded image
        pil_image = Image.open(uploaded_image)
        processed_image = process_uploaded_image(pil_image)

        # Display the uploaded image
        st.markdown(
            f"""
            <div class="centered-container">
                <img src="data:image/jpeg;base64,{image_to_base64(processed_image)}" class="responsive-image" alt="Uploaded Image"/>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Add extra space using Streamlit components
        st.write("")  # This adds a small vertical space
        
        # Classification button
        if st.button('What kind of waste is this?'):
            # Get prediction and confidence
            waste_to_idx = {
                "Glass": 0,
                "Hazardous Waste": 1,
                "Metal": 2,
                "Organic Waste": 3,
                "Paper & Cardboard": 4,
                "Plastic": 5,
                "Textiles": 6,
                "Trash": 7,
            }
            
            # Debug code to show all class probabilities
            input_tensor = preprocess_for_classification(pil_image)
            input_tensor = input_tensor.to(device)
            
            with torch.no_grad():
                logits = classifier(input_tensor)
                probs = torch.softmax(logits, dim=1)[0]
                
                predicted_idx = torch.argmax(probs).item()
                predicted_waste = list(waste_to_idx.keys())[predicted_idx]
                confidence = probs[predicted_idx].item() * 100
            
                # Display results
                st.markdown(
                    f'<p class="prediction-text">{predicted_waste}</p>' +
                    f'<p class="confidence-text">Confidence: {confidence:.2f}%</p>',
                    unsafe_allow_html=True
                )
                
                ## IF CONFIDENCE IS < 50%
                # Display a warning message
                if confidence < 50:
                    st.markdown(
                        """
                        <div style="background-color: #fff3cd; border-left: 6px solid #ffa502; padding: 10px; margin-top: 10px; border-radius: 5px;">
                            <p style="color: #856404; font-size: 16px;">⚠️ <strong>We're not very confident in this result.</strong><br><p>
                            <p style="color: #856404; font-size: 16px;">
                                This might be due to background noise or multiple items in the image.<br>
                                For better results, please upload an image of <strong>a single item</strong> on a <strong>plain or neutral background</strong>.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Show alternatives with confidence > 20%, excluding top prediction
                    threshold = 0.20
                    alternative_classes = [
                        (i, prob.item())
                        for i, prob in enumerate(probs)
                        if i != predicted_idx and prob.item() > threshold
                    ]

                    # Sort by confidence
                    alternative_classes.sort(key=lambda x: x[1], reverse=True)

                    if alternative_classes:
                        alt_texts = [f"{list(waste_to_idx.keys())[i]} ({p * 100:.0f}%)" for i, p in alternative_classes[:3]]
                        st.markdown(
                            f'<p class="custom-font" style="margin-top: 10px;">🧐 This item might also be: '
                            + " or ".join(f"<strong>{text}</strong>" for text in alt_texts)
                            + ".</p>",
                            unsafe_allow_html=True
                        )

                        
                # Display category description and tip
                info = waste_disposal_info.get(predicted_waste)
                if info:
                    st.markdown(f'<h4 class="custom-heading"><i class="fas fa-info-circle" style="margin-right: 8px;"></i>About this category</h4>', unsafe_allow_html=True)
                    st.markdown(f'<p class="custom-font">{info["description"]}</p>', unsafe_allow_html=True)

                    st.markdown(f'<h4 class="custom-heading"><i class="fas fa-recycle" style="margin-right: 8px;"></i>How to dispose/recycle?</h4>', unsafe_allow_html=True)
                    st.markdown(f'<p class="custom-font">{info["disposal_tip"]}</p>', unsafe_allow_html=True)

                else:
                    st.markdown(f'<p class="custom-font">No information available for this category.</p>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()