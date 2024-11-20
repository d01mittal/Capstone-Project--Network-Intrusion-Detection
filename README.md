# Network Intrusion Detection Using AI/ML

This project focuses on detecting network intrusions using machine learning models, leveraging the NSL-KDD dataset. The primary objective is to classify network traffic as normal or attack-related and enhance detection accuracy using various AI/ML techniques.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Conclusion](#conclusion)
6. [Requirements](#requirements)
7. [How to Run](#how-to-run)
8. [References](#references)

## Introduction
In today's digital era, securing networks from malicious attacks is crucial. Traditional Network Intrusion Detection Systems (NIDS) often face challenges in keeping up with evolving threats. This project uses AI and Machine Learning to build a more adaptive and effective intrusion detection system. Various models are trained and evaluated on the NSL-KDD dataset to classify network traffic.

## Dataset
The NSL-KDD dataset is used for this project. It is a benchmark dataset widely used for evaluating network intrusion detection systems.

- **Training Dataset**: Contains labeled examples of normal and attack-related traffic.
- **Test Dataset**: Used for evaluating model performance.

## Methodology
1. **Data Preprocessing**: Data was cleaned, scaled, and categorical variables were encoded. Feature selection techniques were used to identify the most relevant features.
2. **Model Training**: Several machine learning algorithms were trained:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Gaussian Naive Bayes
   - Decision Trees
   - Random Forest
   - Support Vector Machines (SVM)
3. **Dimensionality Reduction**: Principal Component Analysis (PCA) was employed to reduce the dimensionality of the dataset.
4. **Model Evaluation**: Models were evaluated using metrics like accuracy, precision and recall.

## Results
The Random Forest Classifier showed the highest accuracy and proved to be effective in identifying different attack types. Other models also provided insights, with each model's strengths and weaknesses analyzed in detail.

## Conclusion
The project demonstrates the capability of AI/ML techniques to improve network intrusion detection. Future work could involving the deployment of the model for real-time intrusion detection using Apache Kafka.

## Requirements
To run the notebook, you need the following Python packages:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

Install them using:

```bash
pip install numpy pandas matplotlib scikit-learn
