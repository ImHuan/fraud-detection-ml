# fraud-detection-ml

This project implements a machine learning pipeline for detecting fraudulent credit card transactions using the XGBoost algorithm.  
The model is trained on a highly imbalanced dataset and incorporates techniques such as SMOTE oversampling and class weighting to improve fraud detection performance.

## Quick Start

```bash
git clone https://github.com/ImHuan/fraud-detection-ml.git
cd fraud-detection-ml
pip install -r requirements.txt
python -m src.main
```

## Project Structure

```
fraud-detection-ml
│
├── data/                 # Dataset directory
├── src/                  # Source code
│   ├── main.py           # Entry point of the project
│   ├── config.py         # Configuration settings
│   └── ...               # Other modules
│
├── requirements.txt      # Python dependencies
└── README.md
```

## Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Running the Project

After installing the dependencies, run the project with:

```bash
python -m src.main
```

This command executes the full pipeline, including:

- Data loading and preprocessing  
- Handling class imbalance using SMOTE  
- Training the XGBoost classifier  
- Evaluating model performance  

## Dataset

The project uses the **Credit Card Fraud Detection Dataset**, which contains transactions made by European cardholders in September 2013.

Key characteristics of the dataset:

- Total transactions: **284,807**
- Fraud ratio: **0.17%**
- Number of features: **30**

Most features are anonymized using **Principal Component Analysis (PCA)**.  
Only two features remain in their original form:

- **Time** – seconds elapsed between transactions  
- **Amount** – transaction value  

Dataset source:  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Methods

The machine learning pipeline includes the following steps:

1. Data preprocessing  
2. Train-test split with stratified sampling  
3. Handling class imbalance using **SMOTE**  
4. Model training using **XGBoost**  
5. Performance evaluation using **accuracy, precision, recall, and F1-score**

## Results

Experiments show that handling class imbalance improves the model’s ability to detect fraudulent transactions.  
While recall increases after applying SMOTE and class weighting, precision slightly decreases due to more false positive predictions. This reflects the typical trade-off in fraud detection systems.

## Future Work

Possible improvements include:

- Hyperparameter tuning  
- Testing other models such as LightGBM  
- Applying anomaly detection techniques  
- Exploring deep learning approaches
