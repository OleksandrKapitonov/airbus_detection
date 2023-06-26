# airbus_detection
airbus ship segmentation chalenge
# Airbus Ship Segmentation

This project focuses on ship segmentation using the U-Net architecture with SGD optimizer and Dice loss. It includes several Python files, such as an IPython notebook, helper functions, custom datasets, and scripts for training and inference.

## Files

- `notebook.ipynb`: This IPython notebook provides a detailed explanation of the project, including the data preprocessing, model architecture, training, and evaluation. It also showcases the results and provides insights into the ship segmentation process.

- `helper_functions.py`: This file contains various utility functions used throughout the project. These functions assist in tasks such as data preprocessing, augmentation, image visualization, and evaluation metrics calculation.

- `custom_datasets.py`: Here, you will find the implementation of custom datasets for loading and preprocessing the Airbus ship segmentation dataset. These datasets are designed to be used with popular deep learning frameworks like PyTorch or TensorFlow.

- `train_u_net.py`: This script is responsible for training the U-Net model using the SGD optimizer and the Dice loss. It loads the dataset, splits it into training and validation sets, and trains the model using specified hyperparameters. The trained model weights are saved for later use.

- `inference.py`: This script demonstrates how to use the trained U-Net model for ship segmentation on unseen images. It loads the saved model weights, applies the model to the test images, and generates segmented ship masks as output.

## Usage

To run this project, follow these steps:

1. Ensure that you have the necessary dependencies installed. You can find the required packages in the `requirements.txt` file.

2. Download the Airbus ship segmentation dataset and place it in the appropriate directory.

3. Open and run the `notebook.ipynb` file in your preferred Python environment. This notebook provides a comprehensive overview of the project, from data preprocessing to model evaluation.

4. If you prefer to use the command line, you can use the provided scripts. Run `python train_u_net.py` to train the U-Net model using the SGD optimizer and Dice loss. The trained model weights will be saved to a file.

5. To perform inference on new images, run `python inference.py`. This script loads the saved model weights and applies the trained U-Net model to generate ship segmentation masks for unseen images.

## Acknowledgments

The U-Net architecture and SGD optimizer with Dice loss implementation are based on the work of Ronneberger et al. in the paper "U-Net: Convolutional Networks for Biomedical Image Segmentation."

## License

This project is licensed under the [MIT License](LICENSE).
