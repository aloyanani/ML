# ML Project

This repository contains solutions to popular machine learning tasks.

## 1. Titanic - Machine Learning from Disaster

(https://www.kaggle.com/competitions/titanic)

## Folder Structure

- `Datasets/`: Raw data files
- `preprocess_data.py`: Data cleaning and feature engineering
- `train_model.py`: Model training and evaluation

## How to Run

1. Install requirements: `pip install pandas scikit-learn seaborn`
2. Run preprocessing: `python preprocess_data.py`
3. Train model: `python train_model.py`

## Results

- Best accuracy: 0.864
- F1 Score : 0.8247
- Precision Score: 0.8695
- Kaggle submission score: 0.77033
- Model used: Random Forest

## Credits

- Titanic dataset from Kaggle


## 2. MNIST Classificaition

## Folder Structure

- Model: Convolutional Neural Network (CNN)
- Architecture :  3 convolutional layers with ReLU and MaxPooling
                    Flatten + Linear layer to output 10 classes
- Optimizer : Adam
- Loss function : CrossEntropyLoss

## Results

- F1 Score : 0.9838
- Precision Score: 9842

## Credits

- MNIST dataset 