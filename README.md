# Wine Quality Prediction Project

Hello! As part of my internship at Bharat Intern, I've been working on an interesting project that involves predicting wine quality. The dataset I've used for this project is from a CSV file named "winequality.csv."

## Introduction

In this project, I've focused on building a model that can predict the quality of wine based on various attributes. It's an exciting endeavor, as accurately predicting wine quality can have a significant impact on the wine industry.

## Libraries I Used

During this project, I harnessed the power of a few key libraries:

- **Pandas**: I used Pandas for efficient data manipulation and analysis.
- **Scikit-learn**: This versatile library provided tools for preprocessing the data, training models, and evaluating their performance.
- **Matplotlib**: I relied on Matplotlib to create visualizations for a better understanding of the data.
- **Logistic Regression**: I employed this algorithm to train a model for wine quality prediction.
- **Support Vector Machine (SVM)**: Another algorithm I used, specifically the radial basis function (RBF) kernel variant.

## My Process

Here's a step-by-step overview of how I approached this project:

### Data Exploration

I started by loading the dataset and peeking at the first few records using Pandas. This initial exploration helped me get a feel for the data.

### Data Information

To understand the dataset more deeply, I used the `info()` method to gather information about the columns and their data types.

### Descriptive Statistics

I calculated the descriptive statistics of the dataset using the `describe()` method. This gave me insights into the central tendencies, spreads, and ranges of the features.

### Handling Missing Values

I checked for missing values in the dataset using the `isnull().sum()` method. To handle these missing values, I filled them with the mean values of their respective columns.

### Feature Engineering

I dropped the 'total sulfur dioxide' column as it wasn't relevant for the prediction task. Additionally, I created a new column named 'best quality,' which indicates whether the wine is of the best quality (quality > 5) or not.

### Data Splitting

I split the data into features (attributes) and the target variable ('best quality'). Then, I further split the data into training and testing sets using the `train_test_split` function from Scikit-learn.

### Model Training and Evaluation

I trained two models: Logistic Regression and SVM with an RBF kernel. After normalizing the data, I trained the models on the training set and evaluated their performance using the area under the Receiver Operating Characteristic (ROC) curve (ROC AUC) for both training and validation data.

### Confusion Matrix

I visualized the performance of the SVM model using a confusion matrix plot.

### Classification Report

Lastly, I generated a classification report that provides a comprehensive evaluation of the model's performance, including precision, recall, and F1-score.

## Conclusion

This project delved into the intriguing realm of wine quality prediction. Through data exploration, preprocessing, model training, and evaluation, I gained valuable insights into the capabilities of Logistic Regression and SVM algorithms. Feel free to explore the detailed code and comments in the Jupyter Notebook file for a closer look at my journey during this internship at Bharat Intern.
