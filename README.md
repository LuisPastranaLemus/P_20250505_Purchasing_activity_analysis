# 🧭 Purchasing Activity Analysis
Instacart is a grocery delivery platform where customers can place an order and have it delivered.

---

## 🔍 Project Overview (P-20250505)

Analyze Instacart users' shopping behavior to identify repurchase patterns, order frequency, and key products in the shopping process.

Key questions:

- Which days and time of the day have the most order activity?
- How often do users typically place orders?
- How many items do they typically purchase per order?
- Which products have the highest repurchase rate?
- Which products are typically added to the cart first, and how relevant are they?
- Is there a relationship between the order in which a product is added to the cart and the likelihood of it being reordered?
- How frequent is repurchase among different user segments?

---

## 🧮 Data Dictionary

There are five tables in the dataset, and you'll need to use all of them to perform data preprocessing and exploratory data analysis. Below is a data dictionary that lists the columns in each table and describes the data they contain.

- `instacart_orders.csv`: Each row corresponds to an order in the Instacart app.
    - `'order_id'`: An ID number that uniquely identifies each order.
    - `'user_id'`: An ID number that uniquely identifies each customer's account.
    - `'order_number'`: The number of times this customer has placed an order.
    - `'order_dow'`: The day of the week the order was placed (0 if Sunday).
    - `'order_hour_of_day'`: The hour of the day the order was placed.
    - `'days_since_prior_order'`: The number of days since this customer placed their previous order.

- `products.csv`: Each row corresponds to a unique product that customers can purchase.
    - `'product_id'`: ID number that uniquely identifies each product.
    - `'product_name'`: Name of the product.
    - `'aisle_id'`: ID number that uniquely identifies each grocery aisle category.
    - `'department_id'`: ID number that uniquely identifies each grocery department.

- `'order_products.csv`: Each row corresponds to an item ordered in an order.
    - `'order_id'`: ID number that uniquely identifies each order.
    - `'product_id'`: ID number that uniquely identifies each product.
    - `'add_to_cart_order'`: The sequential order in which each item was added to the cart.
    - `'reordered'`: 0 if the customer has never ordered this product before, 1 if they have.

- `aisles.csv`
    - `'aisle_id'`: ID number that uniquely identifies each grocery aisle category.
    - `'aisle'`: Aisle name.

- `departments.csv`
    - `'department_id'`: ID number that uniquely identifies each grocery department.
    - `'department'`: Department name.

---

## 📚 Guided Foundations (Historical Context)

The notebook `00-guided-analysis_foundations.ipynb` reflects an early stage of my data analysis learning journey, guided by TripleTen. It includes data cleaning, basic EDA, and early feature exploration, serving as a foundational block before implementing the improved structure and methodology found in the main analysis.

---

## 📂 Project Structure

```bash
├── data/
│   ├── raw/              # Original dataset(s) in CSV format
│   ├── interim/          # Intermediate cleaned versions
│   └── processed/        # Final, ready-to-analyze dataset
│
├── notebooks/
│   ├── 00-guided-analysis_foundations.ipynb     ← Initial guided project (TripleTen)
│   ├── 01_cleaning.ipynb                        ← Custom cleaning 
│   ├── 02_feature_engineering.ipynb             ← Custom feature engineering
│   ├── 03_eda_and_insights.ipynb                ← Exploratory Data Analysis & visual storytelling
│   └── 04-sda_hypotheses.ipynb                  ← Business insights and hypothesis testing
│
├── src/
│   ├── init.py              # Initialization for reusable functions
│   ├── data_cleaning.py     # Data cleaning and preprocessing functions
│   ├── data_loader.py       # Loader for raw datasets
│   ├── eda.py               # Exploratory data analysis functions
│   ├── features.py          # Creation and transformation functions for new variables to support modeling and EDA
│   └── utils.py             # General utility functions for reusable helpers
│
├── outputs/
│   └── figures/          # Generated plots and visuals
│
├── requirements/
│   └── requirements.txt      # Required Python packages
│
├── .gitignore            # Files and folders to be ignored by Git
└── README.md             # This file
```
---

🛠️ Tools & Libraries

- Python 3.11
- os, pathlib, sys, pandas, NumPy, Matplotlib, seaborn, IPython.display, scipy.stats
- Jupyter Notebook
- Git & GitHub for version control

---

## 📌 Notes

This project is part of a personal learning portfolio focused on developing strong skills in data analysis, statistical thinking, and communication of insights. Constructive feedback is welcome.

---

##👤 Author   
##### Luis Sergio Pastrana Lemus   
##### Engineer pivoting into Data Science | Passionate about insights, structure, and solving real-world problems with data.   
##### [GitHub Profile](https://github.com/LuisPastranaLemus)   
#####📍 Querétaro, México     
##### 📧 Contact: luis.pastrana.lemus [at] engineer.com   
---
