import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import io

# Configure page
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the model to avoid reloading
@st.cache_resource
def load_model():
    """Load the trained model with error handling"""
    try:
        # Try different possible paths
        model_paths = [
            "../models/trained_plant_disease_model.keras",
            "models/trained_plant_disease_model.keras",
            "./models/trained_plant_disease_model.keras"
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                return tf.keras.models.load_model(path)
        
        st.error("❌ Model file not found. Please ensure the model is in the correct location.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Disease information dictionary
DISEASE_INFO = {
    'Apple___Apple_scab': {
        'description': 'A fungal disease causing dark, scabby lesions on leaves and fruit.',
        'treatment': 'Apply fungicide sprays and ensure good air circulation.',
        'severity': 'Medium'
    },
    'Apple___Black_rot': {
        'description': 'Causes brown rot on fruit and leaf spots.',
        'treatment': 'Remove infected parts and apply copper-based fungicides.',
        'severity': 'High'
    },
    'Tomato___Early_blight': {
        'description': 'Fungal disease causing dark spots with concentric rings.',
        'treatment': 'Use resistant varieties and apply preventive fungicides.',
        'severity': 'Medium'
    },
    'Tomato___Late_blight': {
        'description': 'Serious disease causing water-soaked lesions.',
        'treatment': 'Remove infected plants immediately and apply fungicides.',
        'severity': 'High'
    },
    # Add more disease info as needed
}

def get_disease_info(disease_name):
    """Get disease information"""
    return DISEASE_INFO.get(disease_name, {
        'description': 'Information not available for this disease.',
        'treatment': 'Consult with a plant pathologist for treatment recommendations.',
        'severity': 'Unknown'
    })

def format_disease_name(class_name):
    """Format disease name for better readability"""
    # Replace underscores with spaces and improve formatting
    formatted = class_name.replace('___', ' - ').replace('_', ' ')
    return formatted

def model_prediction(test_image):
    """Make prediction with error handling"""
    try:
        model = load_model()
        if model is None:
            return None
        
        # Process image
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr]) / 255.0  # Normalize pixel values
        
        # Make prediction
        predictions = model.predict(input_arr)
        confidence = np.max(predictions)
        result_index = np.argmax(predictions)
        
        return result_index, confidence
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        return None

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 80px;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #4CAF50, #45a049);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .main-content {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .feature-box {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
        border: 1px solid #e9ecef;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .feature-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2);
    }
    
    .welcome-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        margin: 2rem 0;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        text-align: center;
    }
    
    .welcome-box h3 {
        color: white !important;
        margin-bottom: 1rem;
    }
    
    .welcome-box p {
        color: #f8f9fa !important;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .confidence-bar {
        background: linear-gradient(90deg, #ff6b6b, #feca57, #48dbfb, #0abde3);
        height: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .quick-start {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(255, 234, 167, 0.3);
    }
    
    .quick-start h2 {
        color: #2d3436 !important;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .feature-box h4 {
        color: #2d3436 !important;
        margin-bottom: 1rem;
        font-size: 1.3rem;
    }
    
    .feature-box p {
        color: #636e72 !important;
        font-size: 1rem;
        line-height: 1.5;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #74b9ff 0%, #0984e3 100%);
    }
    
    /* Main page background */
    .main .block-container {
        background: linear-gradient(135deg, #f1f2f6 0%, #ddd5d0 100%);
        min-height: 100vh;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with improved styling
st.sidebar.markdown("### 🌿 Navigation")
app_mode = st.sidebar.selectbox(
    "Choose a page:",
    ["🏠 Home", "ℹ️ About", "🔍 Disease Recognition"],
    help="Select a page to navigate"
)

# Class names (same as original but formatted)
class_name = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Main Pages
if app_mode == "🏠 Home":
    st.markdown('<h1 class="main-header">🌿 Intelligent Plant Disease Recognition and Diagnosis System</h1>', unsafe_allow_html=True)
    
    # Create columns for better layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Try to load the home image, use placeholder if not found
        try:
            if os.path.exists("home_page.jpeg"):
                st.image("home_page.jpeg", use_column_width=True)
            else:
                st.info("🖼️ Upload a home_page.jpeg image to display here")
        except:
            st.info("🖼️ Home image not found")
    
    # # Welcome section with distinct styling
    # st.markdown("""
    # <div class="welcome-box">
    #     <h3>🎯 Welcome to our AI-Powered Plant Disease Detection System!</h3>
    #     <p>Protect your crops with cutting-edge machine learning technology. Upload a plant image and get instant disease diagnosis with treatment recommendations.</p>
    # </div>
    # """, unsafe_allow_html=True)
    
    # # Feature highlights with improved contrast
    # col1, col2, col3 = st.columns(3)
    
    # with col1:
    #     st.markdown("""
    #     <div class="feature-box">
    #         <h4>⚡ Lightning Fast</h4>
    #         <p>Get results in seconds with our optimized AI model</p>
    #     </div>
    #     """, unsafe_allow_html=True)
    
    # with col2:
    #     st.markdown("""
    #     <div class="feature-box">
    #         <h4>🎯 High Accuracy</h4>
    #         <p>95%+ accuracy with state-of-the-art deep learning</p>
    #     </div>
    #     """, unsafe_allow_html=True)
    
    # with col3:
    #     st.markdown("""
    #     <div class="feature-box">
    #         <h4>🔬 Expert Analysis</h4>
    #         <p>Detailed disease information and treatment recommendations</p>
    #     </div>
    #     """, unsafe_allow_html=True)
    
    # # Quick start guide with distinct background
    # st.markdown("""
    # <div class="quick-start">
    #     <h2>🚀 Quick Start Guide</h2>
    #     <ol style="color: #2d3436; font-size: 1.1rem; line-height: 1.8;">
    #         <li><strong>📸 Upload:</strong> Navigate to 'Disease Recognition' and upload a clear image of the affected plant</li>
    #         <li><strong>🔍 Analyze:</strong> Our AI will process the image and identify potential diseases</li>
    #         <li><strong>📋 Review:</strong> Get detailed results with confidence scores and treatment recommendations</li>
    #         <li><strong>💡 Act:</strong> Follow the suggested treatment plan for optimal plant health</li>
    #     </ol>
    # </div>
    # """, unsafe_allow_html=True)

elif app_mode == "ℹ️ About":
    st.markdown('<h1 class="main-header">📊 About This Project</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Project Overview
        This Plant Disease Recognition System uses deep learning to identify diseases in crop plants. 
        The system can detect 38 different plant conditions across multiple crop types including tomatoes, 
        apples, corn, grapes, and more.
        
        ### 📈 Dataset Information
        - **Total Images**: ~87,000 RGB images
        - **Classes**: 38 different plant disease categories
        - **Training Set**: 70,295 images (80%)
        - **Validation Set**: 17,572 images (20%)
        - **Test Set**: 33 images for evaluation
        
        ### 🔬 Technology Stack
        - **Deep Learning**: TensorFlow/Keras
        - **Frontend**: Streamlit
        - **Image Processing**: OpenCV, PIL
        - **Model Architecture**: Convolutional Neural Network (CNN)
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Model Performance
        - **Training Accuracy**: 95%+
        - **Validation Accuracy**: 95%+
        - **Image Size**: 128x128 pixels
        - **Color Mode**: RGB
        
        ### 🌱 Supported Plants
        - 🍎 Apple
        - 🫐 Blueberry  
        - 🍒 Cherry
        - 🌽 Corn (Maize)
        - 🍇 Grape
        - 🍑 Peach
        - 🫑 Pepper
        - 🥔 Potato
        - 🍓 Strawberry
        - 🍅 Tomato
        """)

elif app_mode == "🔍 Disease Recognition":
    st.markdown('<h1 class="main-header">🔍 Plant Disease Detection</h1>', unsafe_allow_html=True)
    
    # Create columns for better layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📸 Upload Plant Image")
        test_image = st.file_uploader(
            "Choose an image file:",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a clear image of the plant leaf or affected area"
        )
        
        # Image preview
        if test_image is not None:
            st.markdown("### 🖼️ Image Preview")
            image = Image.open(test_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.info(f"""
            **Image Details:**
            - Size: {image.size[0]} x {image.size[1]} pixels
            - Format: {image.format}
            - Mode: {image.mode}
            """)
    
    with col2:
        st.markdown("### 🤖 AI Analysis")
        
        if test_image is not None:
            if st.button("🔍 Analyze Plant Disease", type="primary"):
                with st.spinner("🔄 Analyzing image... Please wait"):
                    result = model_prediction(test_image)
                    
                    if result is not None:
                        result_index, confidence = result
                        predicted_class = class_name[result_index]
                        formatted_name = format_disease_name(predicted_class)
                        
                        # Display results
                        st.markdown(f"""
                        <div class="prediction-result">
                            <h3>🎯 Detection Result</h3>
                            <h2>{formatted_name}</h2>
                            <p>Confidence: {confidence:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence bar
                        confidence_color = "green" if confidence > 0.8 else "orange" if confidence > 0.6 else "red"
                        st.markdown(f"""
                        <div style="background: #f0f0f0; border-radius: 10px; padding: 5px;">
                            <div style="background: {confidence_color}; width: {confidence*100}%; height: 20px; border-radius: 10px;"></div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Disease information
                        disease_info = get_disease_info(predicted_class)
                        
                        st.markdown("### 📋 Disease Information")
                        
                        if "healthy" in predicted_class.lower():
                            st.success("🎉 Great news! Your plant appears to be healthy!")
                        else:
                            severity_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢", "Unknown": "⚪"}
                            
                            st.markdown(f"""
                            **{severity_color.get(disease_info['severity'], '⚪')} Severity:** {disease_info['severity']}
                            
                            **📝 Description:** {disease_info['description']}
                            
                            **💊 Treatment:** {disease_info['treatment']}
                            """)
                            
                            # Additional recommendations
                            st.markdown("### 💡 Additional Recommendations")
                            st.warning("""
                            - Monitor the plant closely for spread of symptoms
                            - Isolate affected plants if possible
                            - Ensure proper air circulation and drainage
                            - Consult with a local agricultural extension office for severe cases
                            """)
                        
                        # Balloons for healthy plants
                        if "healthy" in predicted_class.lower():
                            st.balloons()
        else:
            st.info("👆 Please upload an image first to start the analysis")
            
            # Example images section
            st.markdown("### 📝 Tips for Best Results")
            st.markdown("""
            - Use clear, well-lit images
            - Focus on the affected leaf or plant part
            - Avoid blurry or heavily shadowed images
            - Ensure the plant fills most of the frame
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    Made with ❤️ using Streamlit and TensorFlow | Plant Disease Recognition System v2.0
</div>
""", unsafe_allow_html=True)