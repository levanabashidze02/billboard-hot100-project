# Project Contributions

---

## Team Members & Responsibilities

### Levani Abashidze

- Identified and sourced the historical Billboard Hot 100 dataset.
- Implemented the data cleaning pipeline, including handling missing values and filtering for the "Modern Era" (2000–Present).
- Transformed raw weekly data into a song-level dataset suitable for machine learning with feature engineering.
- Conducted the initial EDA and created visualizations to identify trends in processed data.
- Authored respective sections of the project documentation and README.

---

### Luka Sanaia

- Designed and implemented Random Forest Regressor models to predict continuous target variables.
- Developed:
  - **Peak Position Prediction** model to forecast a song's highest rank based on debut metrics.
  - **Longevity Prediction** model to estimate the total weeks a song remains on the chart.
- Analyzed feature importance to determine key drivers of chart success (e.g. Debut Rank vs. Seasonality).
- Authored respective sections of the project documentation and README.

---

### Nikoloz Qasrashvili

- Developed a classification model to distinguish between different types of chart success.
- Analysed the target variable for "Fast Burn" songs (high debut, short stay) and trained a **Decision Tree Classifier** to identify these "hype-driven" tracks.
- Evaluated model performance using Confusion Matrices, Precision and Recall to ensure accurate identification of short-term hits.
- Authored respective sections of the project documentation and README.

---

### Ana Cholokava

- Built the probabilistic framework to predict **#1 Hits (Top-1 Probability)**.
- Engineered features including `song_order`, `songs_to_top1` and `prior_chart_experience`.
- Trained and compared Logistic Regression vs. Decision Tree models, evaluating performance to analyse class imbalance.
- Generated advanced comparative plots and feature importance charts to interpret the "Number One" prediction models.
- Authored respective sections of the project documentation and README.
