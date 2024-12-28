# CNN for MNIST Classification

This repository contains a Jupyter Notebook (`cnn_mnist.ipynb`) that demonstrates how to build, train, and evaluate a Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset using TensorFlow and Keras.

## Overview

The notebook covers the following steps:

1. **Importing Libraries**: Essential libraries such as TensorFlow, Keras, NumPy, and Matplotlib are imported.
2. **Loading and Preprocessing Data**: The MNIST dataset is loaded and preprocessed. The data is reshaped, normalized, and one-hot encoded.
3. **Building the CNN Model**: A sequential CNN model is built using Keras. The model includes convolutional layers, batch normalization, flattening, and dense layers.
4. **Compiling the Model**: The model is compiled using the Adam optimizer and categorical cross-entropy loss.
5. **Training the Model**: The model is trained on the MNIST dataset for 10 epochs.
6. **Evaluating the Model**: The model's performance is evaluated on the test set, and a confusion matrix is generated to visualize the results.
7. **Visualizing Predictions**: A sample of test images is displayed along with their true and predicted labels.

## Requirements

To run this notebook, you need the following Python libraries installed:

- TensorFlow
- Keras
- NumPy
- OpenCV
- Scikit-learn
- Seaborn
- Matplotlib

You can install these libraries using pip:

```bash
pip install tensorflow keras numpy opencv-python scikit-learn seaborn matplotlib
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/omaradel73/CNN-Mnist.git
   cd cnn-mnist
   ```

2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook cnn_mnist.ipynb
   ```

3. Follow the steps in the notebook to build, train, and evaluate the CNN model.

## Results

The notebook includes visualizations of the training process, the confusion matrix, and a sample of test images with their true and predicted labels to help you understand the model's performance.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The MNIST dataset is used for training and evaluation.
- TensorFlow and Keras documentation provided valuable insights for building the CNN model.

---
