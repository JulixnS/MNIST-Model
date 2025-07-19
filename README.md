# MNIST Project

This project implements a simple feedforward neural network for handwritten digit classification using the MNIST dataset. The model is built with PyTorch and trained to recognize digits (0-9) from pixel data.

## Features

- Loads and preprocesses MNIST CSV data
- Splits data into training and test sets
- Defines a neural network model in PyTorch
- Trains the model and tracks loss/accuracy over epochs
- Plots training and test loss/accuracy to visualize overfitting/underfitting
- Predicts individual digits from test data

## Usage

1. Place `mnist_train.csv` and `mnist_test.csv` in the `data/` folder.
2. Run `main.py` to train the model and visualize results.
3. The script will print predictions for random test images.

## Requirements

- Python 3.x
- PyTorch
- pandas
- numpy
- scikit-learn
- matplotlib

## Large Files

If using large CSV files, track them with [git-lfs](https://git-lfs.github.com/) to avoid GitHub file size limits.
