# Indus-Copper-Modelling

## Project Overview
  The objective of this project is to develop machine learning models and deploy them as a user-friendly web application that predicts the selling price of copper and classifies leads based on their status (WON or LOST). This project addresses the challenges in the copper industry, such as data skewness and noisy data, by utilizing advanced data preprocessing techniques and robust machine learning algorithms.

## Skills Takeaway
* Python Scripting
* Data Preprocessing
* Exploratory Data Analysis (EDA)
* Streamlit

## Domain
Manufacturing

## Problem Statement
  The copper industry deals with data related to sales and pricing, which may suffer from skewness and noise. These issues can affect the accuracy of manual predictions, making optimal pricing decisions difficult. A machine learning regression model can address these issues using techniques like data normalization, feature scaling, and outlier detection. Additionally, a lead classification model can evaluate and classify leads based on their likelihood of becoming a customer, using the STATUS variable with WON considered as Success and LOST as Failure.

## Scope of the Project
The project involves the following tasks:

1.Data Understanding: Identify variable types and their distributions. Handle rubbish values in ‘Material_Reference’ by converting values starting with '00000' to null. Treat reference columns as categorical variables.
2.Data Preprocessing:
* Handle missing values using mean/median/mode.
* Treat outliers using IQR or Isolation Forest.
* Address skewness with appropriate transformations (e.g., log transformation, boxcox transformation).
* Encode categorical variables using techniques like one-hot encoding or label encoding.
3.Exploratory Data Analysis (EDA): Visualize outliers and skewness before and after treatment using Seaborn’s boxplot, distplot, and violinplot.
4.Feature Engineering: Create new features, if applicable, and drop highly correlated columns using a heatmap.
5.Model Building and Evaluation:
* Split the dataset into training and testing/validation sets.
* Train and evaluate classification models (ExtraTreesClassifier, XGBClassifier, Logistic Regression) and regression models.
* Optimize model hyperparameters using cross-validation and grid search.
6.Model GUI: Create a Streamlit page to input values and predict Selling_Price or Status (WON/LOST).

## Deliverables
* A well-trained regression model for predicting Selling_Price.
* A well-trained classification model for predicting Status (WON/LOST).
* A user-friendly web application (built with Streamlit) to make predictions based on user inputs.
* Documentation and instructions for using the application.
* A project report summarizing data analysis, model development, and deployment processes.

## Data Description
The dataset includes the following columns:

1.id: Unique identifier for each transaction or item.
2.item_date: Date when each transaction or item was recorded.
3.quantity tons: Quantity of the item in tons.
4.customer: Name or identifier of the customer.
5.country: Country associated with each customer.
6.status: Current status of the transaction or item (e.g., Draft, Won).
7.item type: Type or category of the items.
8.application: Specific use or application of the items.
9.thickness: Thickness of the items.
10.width: Width of the items.
11.material_ref: Reference or identifier for the material used.
12.product_ref: Reference or identifier for the specific product.
13.delivery date: Expected or actual delivery date for each item.
14.selling_price: Price at which the items are sold.

## Approach
1.Data Understanding: Identify and handle rubbish values and treat reference columns as categorical variables.
2.Data Preprocessing:
* Handle missing values.
* Treat outliers.
* Address skewness.
* Encode categorical variables.
3.EDA: Visualize data distributions and outliers.
4.Feature Engineering: Create new features and drop highly correlated columns.
5.Model Building and Evaluation:
* Train and evaluate regression and classification models.
* Optimize hyperparameters.
6.Model GUI: Develop a Streamlit page for interactive predictions.
