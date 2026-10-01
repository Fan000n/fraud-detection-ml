# Data

## Source

**Credit Card Transactions Fraud Detection Dataset**
- **Author:** Kartik Shenoy
- **Platform:** Kaggle
- **URL:** https://www.kaggle.com/datasets/kartik2112/fraud-detection
- **License:** CC0: Public Domain

## Files

The dataset ships as two CSV files:

| File             | Rows (approx.) | Purpose (original)                    |
|------------------|---------------:|---------------------------------------|
| `fraudTrain.csv` | ~1,296,675     | Training split provided by the author |
| `fraudTest.csv`  | ~555,719       | Test split provided by the author     |

**Note:** Although the files are named `fraudTrain` and `fraudTest`, this project **combines both files and performs its own stratified train/test split**. Relying on the pre-split would risk subtle leakage and would not demonstrate a proper ML workflow in this situation.

## Columns

| Column                           | Type       | Description                                      |
|----------------------------------|------------|--------------------------------------------------|
| `trans_date_trans_time`          | datetime   | Transaction timestamp                            |
| `cc_num`                         | int        | Credit card number (PII — will be dropped)       |
| `merchant`                       | string     | Merchant name                                    |
| `category`                       | string     | Merchant category                                |
| `amt`                            | float      | Transaction amount                               |
| `first`, `last`                  | string     | Cardholder name (PII — will be dropped)          |
| `gender`                         | string     | Cardholder gender                                |
| `street`, `city`, `state`, `zip` | string/int | Cardholder address (PII — will be dropped)       |
| `lat`, `long`                    | float      | Cardholder location                              |
| `city_pop`                       | int        | Population of cardholder's city                  |
| `job`                            | string     | Cardholder occupation                            |
| `dob`                            | date       | Cardholder date of birth (PII — will be dropped) |
| `trans_num`                      | string     | Unique transaction ID                            |
| `unix_time`                      | int        | Transaction time as Unix epoch                   |
| `merch_lat`, `merch_long`        | float      | Merchant location                                |
| **`is_fraud`**                   | int (0/1)  | **Target** — 1 if fraudulent, 0 if legitimate    |

## Setup Instructions

Raw CSV files are **not committed** to this repository (they are large and publicly redistributable from Kaggle, its a moderate sized dataset but to big to commit to GitHub ).

To reproduce the project:

1. Download the dataset from the URL above (a free Kaggle account is required).
2. Extract the archive.
3. Place `fraudTrain.csv` and `fraudTest.csv` inside `data/raw/`:
