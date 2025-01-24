# Face-Emotion-detection

## Image Classification with Transfer Learning

This project focuses on building an image classification pipeline using transfer learning with the VGG-19 pre-trained model. The dataset consists of grayscale images divided into 7 classes, and it is tailored to handle class imbalance and limited data samples effectively. By leveraging the power of transfer learning, the model reuses pre-trained weights and adapts them to classify images accurately. Extensive preprocessing steps, such as resizing, grayscale-to-RGB conversion, and data augmentation, ensure that the input data meets the requirements of the pre-trained model. The project demonstrates an end-to-end workflow, from exploratory data analysis to model training and evaluation.

---

## Features

- **Exploratory Data Analysis (EDA):**
  - Visualize sample images and class distributions to understand dataset characteristics.
- **Advanced Data Preprocessing:**
  - Resizing, grayscale-to-RGB conversion, and dynamic data generation.
- **Transfer Learning:**
  - Utilized VGG-19 pre-trained model with custom top layers for multi-class classification.
- **Regularization Techniques:**
  - Batch normalization and L2 regularization to improve model generalization.
- **Training Optimization:**
  - Early stopping, learning rate scheduling, and model checkpointing for efficient training.
- **Performance Visualization:**
  - Plotted training and validation metrics to evaluate model performance.

---

## Technology Stack

- **Programming Language:** Python
- **Deep Learning Frameworks:** TensorFlow, Keras, PyTorch (torchvision for data handling)
- **Data Manipulation and Analysis:** NumPy, Pandas
- **Visualization Tools:** Matplotlib, Seaborn
- **Image Processing:** OpenCV, PIL
- **Model Architecture:** VGG-19 (Transfer Learning)

---

## Dataset

The dataset used for this project is the [Face Expression Recognition Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset). It includes grayscale images of faces categorized into 7 distinct emotion classes, with a noticeable imbalance in the number of samples across classes.

---

## Outcome

- **Maximum Validation Accuracy:** 64.12%
- **No Overfitting:** Training and validation metrics show minimal differences, indicating strong generalization.
- **Challenges Addressed:**
  - Class imbalance and limited data were mitigated using data augmentation and transfer learning techniques.
  - The VGG19 model was adapted to the dataset by freezing most pre-trained layers and fine-tuning the top layers.
