# K-Nearest Neighbors (KNN) Classifier

## Overview
This repository contains an implementation of the K-Nearest Neighbors (KNN) algorithm using Python. The model is trained and evaluated using a dataset, achieving an accuracy of **66%**.

## Features
- Implements KNN for classification
- Preprocesses and normalizes data
- Splits data into training and testing sets
- Evaluates model performance using accuracy metric

## Dataset
The dataset used in this project consists of labeled samples for classification. It is loaded and preprocessed before applying the KNN algorithm. Ensure that the dataset is structured correctly before running the notebook.

## Prerequisites
To run this project, you need the following dependencies installed:
- Python (>=3.7)
- NumPy
- Pandas
- Scikit-learn
- Matplotlib (optional, for visualization)

You can install the required libraries using:
```sh
pip install numpy pandas scikit-learn matplotlib
```

## Usage
1. Clone this repository:
   ```sh
   git clone <https://github.com/TarunBhatia11/KNNClassification/tree/main>
   ```
2. Open the Jupyter Notebook:
   ```sh
   jupyter notebook KNN.ipynb
   ```
3. Run the notebook cells sequentially to preprocess the data, train the model, and evaluate its performance.

## Model Evaluation
The KNN classifier was evaluated using accuracy as the performance metric. The model achieved an accuracy of **63%**, indicating its effectiveness in classifying the given dataset. The accuracy can be improved by fine-tuning hyperparameters such as the number of neighbors (`k`) and distance metrics.

precision    recall  f1-score   support

           0       0.35      0.21      0.26       144
           1       0.70      0.82      0.75       317

    accuracy                           0.63       461
   macro avg       0.52      0.52      0.51       461
weighted avg       0.59      0.63      0.60       461

![knn](https://github.com/user-attachments/assets/98537895-7344-42d8-b448-6d8f66488edf)


## Conclusion
This project demonstrates the implementation of the K-Nearest Neighbors algorithm for classification tasks. It serves as a foundational model that can be improved with further optimization and hyperparameter tuning.

## Author
Tarun Bhatia 

