# Emotion Analysis Web App

A real-time emotion detection web application that uses deep learning to analyze emotions from webcam input. The application can detect 7 basic emotions: happy, sad, angry, neutral, fear, surprise, and disgust.

## Features
- Real-time webcam capture
- Emotion detection using deep learning
- Web interface for visualization
- Support for 7 basic emotions
- Real-time face detection and emotion classification
- Beautiful and responsive UI

## Project Structure
```
emotion_analysis/
├── app.py              # Flask web application
├── download_model.py   # Model training script
├── plot_training.py    # Training visualization script
├── requirements.txt    # Project dependencies
├── templates/         # HTML templates
│   └── index.html     # Main web interface
├── FER-2013/         # Dataset directory
│   ├── train/        # Training images
│   └── test/         # Test images
└── plots/            # Training visualization plots
```

## Model Architecture
The model uses a Convolutional Neural Network (CNN) with the following architecture:
- Input: 48x48 grayscale images
- Conv2D (32 filters, 3x3 kernel)
- MaxPooling2D
- Conv2D (64 filters, 3x3 kernel)
- MaxPooling2D
- Conv2D (64 filters, 3x3 kernel)
- Flatten
- Dense (64 units)
- Dense (7 units, softmax activation)

## Dataset
The model is trained on the FER2013 dataset, which contains:
- 28,709 training images
- 7,178 test images
- 7 emotion categories

## Model Performance
- Training Accuracy: [To be added after training]
- Validation Accuracy: [To be added after training]
- Test Accuracy: [To be added after training]

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python download_model.py
```

4. Generate training plots:
```bash
python plot_training.py
```

5. Run the application:
```bash
python app.py
```

6. Open your browser and navigate to `http://localhost:5000`

## Technologies Used
- Flask (Web Framework)
- OpenCV (Image Processing)
- TensorFlow (Deep Learning)
- HTML5/CSS3/JavaScript (Frontend)
- scikit-learn (Machine Learning Utilities)

## Screenshots
[To be added after running the application]

## Training Plots
[To be added after generating plots]

## License
This project is licensed under the MIT License - see the LICENSE file for details. 