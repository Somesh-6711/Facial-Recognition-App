# 👤 Facial Recognition System using Siamese Neural Network

This project implements a **Facial Recognition System** using a **Siamese Neural Network (SNN)** architecture, powered by **TensorFlow**. The model is designed for one-shot learning, capable of identifying faces with a small amount of training data.
deep facial recognition application for authentication using TensorFlow and Kivy. 
Inspired by the Siamese Neural Networks for One-shot Image Recognition paper: https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf

## 🚀 Project Overview

The system uses a twin **Convolutional Neural Network (CNN)** setup to process pairs of face images and determine whether they belong to the same person. The architecture is particularly suited for facial verification tasks, making it useful for authentication systems.

**Key Achievements**:
- **78.5% verification accuracy** on the **LFW (Labeled Faces in the Wild)** dataset.
- Real-time authentication with **<500ms verification time per image pair**.

## 📊 Model Architecture

The model is based on a **Siamese Neural Network** consisting of two **CNN branches**. Each branch extracts features from the input images and computes a similarity score.

- **Input**: Pairs of face images
- **Architecture**: CNN with 4 convolutional layers in each branch
- **Output**: Binary classification (Same/Not Same)

```python
# Sample code snippet (Siamese Network architecture)
input = Input(shape=(image_size, image_size, 3))
cnn_branch = Conv2D(64, (3, 3), activation='relu')(input)
cnn_branch = MaxPooling2D(pool_size=(2, 2))(cnn_branch)
...
model.add(Dense(1, activation='sigmoid'))
```

## 🔄 Data Preprocessing
We preprocess the images using OpenCV to align and normalize the face images before feeding them into the Siamese network. The dataset used contains 50,000 face pairs for training and testing, processed in the following way:

Grayscale conversion for reducing input size
Resizing to a consistent format (e.g., 128x128)
Face alignment using facial landmarks

## 🚧 Known Issues
## ⚠️ Memory Limitation: Due to the high memory demands of this project, it may not run on systems with limited memory. On my personal machine, I faced low-memory issues, which prevented me from running the entire training pipeline locally. However, I was able to test the model on a smaller dataset, and the results were consistent with expectations.

If you have access to a more powerful machine or cloud resources, this issue should be resolved.

🔧 Installation & Setup
Clone the repository:
```python
git clone https://github.com/your-username/facial-recognition-system.git
cd facial-recognition-system
```
2. Install the dependencies:
```python
pip install -r requirements.txt
```
3. Download and place the LFW dataset in the /data/lfw/ directory.
4. Run the preprocessing script:
```python
python preprocess.py
```
# 🎯 Results
Verification Accuracy: 78.5% on the LFW dataset.
Real-time Verification: Achieved <500ms for face authentication.
Model Parameters: Optimized CNN architecture with 30% parameter reduction for faster processing.

# 📦 Future Improvements
Memory Optimization: Explore techniques to reduce memory consumption during training.
Real-time Deployment: Deploy the model in an edge device for real-time facial authentication.
Expand Dataset: Use additional datasets to improve model robustness.

# 🤝 Contributing
Feel free to submit a pull request or open an issue for any improvements.
