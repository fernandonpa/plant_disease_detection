<div align="center">
  <img src="home_page.jpeg" alt="Plant Disease Detection System" width="100%" />
</div>

# 🌿 Plant Disease Recognition System

<div align="center">

![Python](https://img.shields.io/badge/python-v3.11+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.47.0-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**An AI-powered plant disease detection system using deep learning to identify 38 different plant diseases with 95%+ accuracy**

[Demo](#demo) • [Features](#features) • [Installation](#installation) • [Usage](#usage) • [Dataset](#dataset)

</div>

## 📋 Table of Contents

- [About the Project](#about-the-project)
- [Dataset Information](#dataset-information)
- [Model Architecture](#model-architecture)
- [Strategies and Approach](#strategies-and-approach)
- [Importance and Objectives](#importance-and-objectives)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Google Colab Setup](#google-colab-setup)
- [Model Performance](#model-performance)
- [Contributing](#contributing)
- [License](#license)

## 🎯 About the Project

This Plant Disease Recognition System leverages state-of-the-art deep learning techniques to automatically identify and classify plant diseases from leaf images. The system can detect 38 different plant conditions across multiple crop types including tomatoes, apples, corn, grapes, potatoes, and more.

### Key Highlights:

- **38 Disease Classes** across 10+ plant species
- **95%+ Accuracy** on validation dataset
- **Real-time Detection** with instant results
- **Treatment Recommendations** for identified diseases
- **User-friendly Web Interface** built with Streamlit

## 📊 Dataset Information

### New Plant Diseases Dataset (Augmented)

The project utilizes the comprehensive **New Plant Diseases Dataset** available on Kaggle:

- **Total Images**: ~87,000 RGB images
- **Classes**: 38 different plant disease categories
- **Training Set**: 70,295 images (80%)
- **Validation Set**: 17,572 images (20%)
- **Test Set**: 33 images for evaluation
- **Image Size**: 128x128 pixels
- **Color Mode**: RGB

**Dataset Source**: [Kaggle - New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

### Supported Plant Species:

- 🍎 Apple (4 classes: Scab, Black rot, Cedar apple rust, Healthy)
- 🍅 Tomato (10 classes: Various diseases + Healthy)
- 🌽 Corn/Maize (4 classes)
- 🍇 Grape (4 classes)
- 🥔 Potato (3 classes)
- 🫑 Pepper (2 classes)
- 🍑 Peach (2 classes)
- 🍓 Strawberry (2 classes)
- 🫐 Blueberry, Cherry, Orange, Raspberry, Soybean, Squash (Various classes)

## 🏗️ Model Architecture

The system employs a **Convolutional Neural Network (CNN)** with the following architecture:

```
    Input Layer (128x128x3)
                ↓
Conv2D(32) → Conv2D(32) → MaxPool2D
                ↓
Conv2D(64) → Conv2D(64) → MaxPool2D
                ↓
Conv2D(128) → Conv2D(128) → MaxPool2D
                ↓
Conv2D(256) → Conv2D(256) → MaxPool2D
                ↓
Conv2D(512) → Conv2D(512) → MaxPool2D
                ↓
    Dropout(0.25) → Flatten
                ↓
Dense(1500, ReLU) → Dropout(0.4)
                ↓
        Dense(38, Softmax)
```

## 🎯 Strategies and Approach

### 1. **Data Preprocessing**

- Image resizing to 128x128 pixels for computational efficiency
- Normalization of pixel values (0-1 range)
- Data augmentation techniques applied in the original dataset

### 2. **Model Design Philosophy**

- **Progressive Feature Learning**: Increasing filter sizes (32→512) to capture complex patterns
- **Regularization**: Dropout layers (0.25, 0.4) to prevent overfitting
- **Hierarchical Feature Extraction**: Multiple convolutional layers with pooling

### 3. **Training Strategy**

- **Optimizer**: Adam with learning rate of 0.0001
- **Loss Function**: Categorical Crossentropy for multi-class classification
- **Batch Size**: 256 (optimized for GPU memory)
- **Epochs**: 10 with early stopping potential

### 4. **Validation Approach**

- 80/20 train-validation split
- Continuous monitoring of validation accuracy
- Model checkpointing for best performance

## 🔍 Importance and Objectives

### Project Importance:

1. **Food Security**: Early disease detection can prevent crop losses and ensure food security
2. **Economic Impact**: Reduces financial losses for farmers through timely intervention
3. **Sustainable Agriculture**: Promotes precision farming and reduces unnecessary pesticide use
4. **Accessibility**: Provides AI-powered diagnostics to farmers in remote areas
5. **Scalability**: Can be deployed on mobile devices for field use

### Primary Objectives:

- ✅ **Accurate Disease Identification**: Achieve >95% accuracy in disease classification
- ✅ **Real-time Processing**: Provide instant results for quick decision making
- ✅ **User-friendly Interface**: Simple web application accessible to non-technical users
- ✅ **Treatment Guidance**: Offer actionable treatment recommendations
- ✅ **Comprehensive Coverage**: Support multiple plant species and disease types

### Impact Areas:

- **Precision Agriculture**: Enable data-driven farming decisions
- **Crop Management**: Improve overall plant health monitoring
- **Education**: Serve as a learning tool for agricultural students and professionals
- **Research**: Provide a foundation for further agricultural AI research

## ✨ Features

- 🔍 **Instant Disease Detection** with confidence scores
- 📱 **Responsive Web Interface** that works on all devices
- 🎯 **High Accuracy** detection using deep learning
- 💊 **Treatment Recommendations** for identified diseases
- 📊 **Detailed Disease Information** including severity levels
- 🖼️ **Image Preview** with metadata display
- ⚡ **Fast Processing** with optimized model architecture
- 🎨 **Beautiful UI** with intuitive navigation

## 🚀 Installation

### Prerequisites

- Python 3.11+
- pip package manager

### Local Setup

1. **Clone the repository**

```bash
git clone https://github.com/fernandonpa/plant_disease_detection.git
cd plant_disease_detection
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Download the pre-trained model**

   - Download `trained_plant_disease_model.keras` and place it in the `models/` directory

4. **Run the application**

```bash
streamlit run app/main.py
```

5. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`

## 📱 Usage

1. **Home Page**: Overview of the system and quick start guide
2. **Disease Recognition**: Upload plant images for analysis
3. **About Page**: Detailed project information and statistics

### Step-by-step Process:

1. Navigate to the "Disease Recognition" page
2. Upload a clear image of the plant leaf or affected area
3. Click "Analyze Plant Disease" button
4. View the prediction results with confidence score
5. Read the disease information and treatment recommendations

## 🔬 Google Colab Setup

### Important GPU Requirements

⚠️ **GPU Acceleration Required**: This project is optimized for Google Colab with GPU acceleration

**Recommended Configuration:**

- **GPU**: 15 GB GPU memory (T4 or better)
- **Runtime**: GPU-enabled runtime in Google Colab
- **Batch Size**: 256 (default)

**For Limited GPU Memory:**
If you don't have access to 15 GB GPU, modify the batch size:

```python
# In the notebook, change batch_size parameter
training_set = tf.keras.utils.image_dataset_from_directory(
    # ... other parameters
    batch_size=128,  # Reduce from 256 to 128 or 64
    # ... other parameters
)
```

### Colab Setup Steps:

1. **Open Google Colab**: [colab.research.google.com](https://colab.research.google.com)

2. **Enable GPU**:

   - Runtime → Change runtime type → Hardware accelerator: GPU

3. **Upload the notebook**:

   - Upload `notebooks/plant_disease_detection.ipynb`

4. **Install Kaggle API**:

   ```python
   !pip install kaggle
   ```

5. **Upload Kaggle credentials**:

   - Download `kaggle.json` from your Kaggle account
   - Upload it when prompted in the notebook

6. **Run all cells** to train the model

## 📈 Model Performance

### Training Results:

- **Training Accuracy**: 95.2%
- **Validation Accuracy**: 95.1%
- **Training Loss**: 0.142
- **Validation Loss**: 0.149
- **Training Time**: ~2 hours on GPU

### Per-Class Performance:

- **Healthy Plants**: 98%+ accuracy
- **Common Diseases**: 94-97% accuracy
- **Rare Diseases**: 90-94% accuracy

### Confusion Matrix:

The model shows excellent performance across all disease categories with minimal misclassification between similar diseases.

## 🛠️ Technology Stack

- **Backend**: Python, TensorFlow/Keras
- **Frontend**: Streamlit
- **Image Processing**: OpenCV, PIL
- **Data Visualization**: Matplotlib, Seaborn
- **Model Format**: Keras (.keras)
- **Deployment**: Streamlit Cloud (optional)

## 📁 Project Structure

```
plant_disease_detection/
├── app/
│   └── main.py                 # Streamlit web application
├── models/
│   └── trained_plant_disease_model.keras  # Pre-trained model
├── notebooks/
│   └── plant_disease_detection.ipynb      # Training notebook
├── data/
│   └── data.txt               # Dataset information
├── requirements.txt           # Python dependencies
├── pyproject.toml            # Project configuration
├── home_page.jpeg            # Homepage image
└── README.md                 # Project documentation
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### How to Contribute:

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Thanks to the creators of the New Plant Diseases Dataset on Kaggle
- **TensorFlow Team**: For providing excellent deep learning framework
- **Streamlit Team**: For the amazing web app framework
- **Agricultural Research Community**: For continuous support in plant pathology research

## 📞 Contact

**Praneeth Anjana** - anjanapraneeth7@gmail.com

Project Link: [https://github.com/fernandonpa/plant_disease_detection](https://github.com/fernandonpa/plant_disease_detection)

---

<div align="center">
  <p>⭐ Star this repository if you found it helpful!</p>
  <p>Made with ❤️ for sustainable agriculture</p>
</div>
