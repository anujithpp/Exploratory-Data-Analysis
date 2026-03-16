# Exploratory Data Analysis (EDA) on JM1 Software Defect Prediction Dataset

This repository contains a Python script (`EDA.py`) demonstrating a practical and comprehensive example of **Exploratory Data Analysis (EDA)**.

## Overview
Exploratory Data Analysis is a crucial first step in any data science or machine learning project. The goal of this script is to programmatically fetch a real-world dataset, inspect its structure, clean it, and visualize its features to gain actionable insights before attempting to build any predictive models.

## Dataset
The dataset utilized in this example is the **JM1 Software Defect Prediction** dataset, fetched using the `scikit-learn` API from OpenML (Data ID: 310). 
This dataset contains various static code attributes (like McCabe's cyclomatic complexity, lines of code, etc.) and a target `class` variable indicating whether a software module contains defects.

### Key Features Explored:
*   `attr1` - `attr6`: Assorted software metrics.
*   `class`: The target boolean variable indicating whether the module is defective.

## What is Covered in this EDA Script?
1. **Data Loading & Inspection:** Downloading the dataset and viewing its initial rows, structure, and data types (`pd.DataFrame.head()`, `pd.DataFrame.info()`).
2. **Descriptive Statistics:** Analyzing the mean, standard deviation, min, and max values of features to understand the data's scale (`pd.DataFrame.describe()`).
3. **Missing Value Checks:** Identifying whether the dataset requires imputation or row-dropping (`pd.DataFrame.isnull().sum()`).
4. **Target Class Distribution:** Visualizing if the dataset is balanced or imbalanced using a Seaborn countplot.
5. **Feature Correlation:** Identifying multicollinearity (redundant features) via a correlation heatmap.
6. **Feature Distributions:** Visualizing the shape, spread, and potential outliers of numerical features using histograms.

## Requirements
To run this script locally, ensure you have Python installed along with the following libraries:
*   `pandas`
*   `matplotlib`
*   `seaborn`
*   `scikit-learn`

You can install all required dependencies using `pip`:
```bash
pip install pandas matplotlib seaborn scikit-learn
```

## How to Run
Navigate to the project directory in your terminal and execute the script:
```bash
python EDA.py
```

This will output the statistical summaries to the console and sequentially open the data visualization plots.