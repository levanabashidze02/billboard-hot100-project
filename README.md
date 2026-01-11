# billboard-hot100-project (Predicting Song Success)

## Team Members

* **Levani Abashidze**
* **Luka Sanaia**
* **Nikoloz Qasrashvili**
* **Ana Cholokava**

---

## 1. Project Overview

The music industry is highly competitive, with thousands of songs released daily. This project analyzes **historical Billboard Hot 100 data** to identify trends, patterns and factors that contribute to a song's commercial success.

### Objectives

Our primary goal is to turn raw chart data into insights and create prediction models using **Data Science and Machine Learning** techniques. Specifically, our goals are to:

* **Analyze Trends -**
  Visualize how song longevity and chart dynamics have shifted in the *Modern Era* (2000-Present).

* **Predict Peak Performance -**
  Use regression models to forecast a song's highest chart position based on its debut metrics.

* **Forecast Longevity -**
  Predict how many weeks a song will remain on the chart.

* **Classify Success Types:**

  * Identify **"Fast Burn"** songs (high debut, quick exit) vs. **"Slow Growers"**
  * Predict if a song will reach the  **#1 spot** based on artist history

---

## 2. Dataset Description

The dataset consists of **weekly Billboard Hot 100 chart rankings**.

* **Source:**
  GitHub - utdata / rwd-billboard-data (UT-Austin School of Journalism and Media)
  [https://github.com/utdata/rwd-billboard-data](https://github.com/utdata/rwd-billboard-data)

* **Timeframe:**
  The original dataset spans **1958-2026**.
  For this project, we cleaned and filtered the data to focus on the **Modern Era (2000-2026)** to better reflect current patterns in music.

### Key Attributes

* `chart_week` - Date of the chart release
* `current_week` - Song's rank (1-100) for that week
* `title`, `performer` - Song attributes
* `wks_on_chart` - Cumulative weeks on the Hot 100
* `peak_pos` - Highest rank achieved by the song

---

## 3. Methodology & Feature Engineering

Before analysis, the data underwent some transformation.

### Data Preparation

* **Data Cleaning**

  * Handling missing values (e.g., first-week chart entries)
  * Converting date columns to datetime format
* **Aggregation**

  * Weekly rows are transformed into **song rows** (one record per unique song)

### Feature Engineering

* `no_for_performer` - Cumulative count of an artist’s previously charted songs (for future ML models based on *artist experience*)
* `songs_to_top1` - Number of songs it took an artist to reach their first #1 hit
* `debut_month` - Seasonality (holiday effects or summer hits)
* `fast_burn` - Boolean/binary label for songs that debut high but exit the chart quickly

---

## 4. Installation & Setup

Make sure **Python** is installed on your system.

### Prerequisites

Install dependencies using:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install those libraries manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### File Structure

* `hot-100-current.csv` - Raw input dataset
* `v1.0-final.py` - Primary script containing Data Cleaning, EDA and ML
* `hot_100_modern.csv` - *(Output)* Filtered dataset (2000+) (in /data/processed folder)
* `cleaned_song_stats.csv` - *(Output)* Aggregated song-level statistics (in /data/processed folder)
* Figures and processed data was saved only once & then converted to comments in the file - not to overload the directory every time the code runs (they are saved in their respective folders). It is also possible to turn the comments into code again and replicate the output.

---

## 5. Usage

1. Place `hot-100-current.csv` in the same directory as the main script.
2. Run the main analysis:

```bash
python v1.0-final.py
```

### The script will:

* Print statistical summaries and data quality checks
* Generate EDA visualizations (heatmaps, time series, box plots)
* Train machine learning models and display evaluation metrics (Accuracy, R², MAE, etc.)
* Save cleaned and processed datasets to CSV files

---

## 6. Machine Learning Models & Results Summary

We implemented **three modeling approaches**, each targeting a different definition of “success”.

### 🔹 Model A: Regression (Random Forest)

**Tasks**

* Predict **Peak Chart Position**
* Predict **Total Weeks on Chart**

**Features**

* `debut_rank`
* `debut_month`
* `no_for_performer`

**Key Insight**

* Debut rank is the strongest predictor of peak performance.
* Longevity is harder to predict due to *“sleeper hits”* that rise slowly over time.

---

### 🔹 Model B: Classification (Decision Tree)

**Task**

* Identify **Fast Burn** songs

**Definition**

* Songs that debut in the **Top 20** but stay on the chart for **less than the median duration**

**Use Case**

* Helps distinguish hype-driven releases from organically growing hits.

---

### 🔹 Model C: Top-1 Prediction

**Logistic Regression vs. Decision Tree**

**Task**

* Predict whether a song will reach **#1**

**Key Features**

* `song_order` - Is this the artist’s 1st song or 20th?
* `songs_to_top1` - Artist’s historical performance
* `has_been_in_top100_before` - New artist vs. returning artist

**Outcome**

* Logistic Regression and Decision Trees are compared to determine which better captures the rarity of a #1 hit.

---

## 7. Visualizations

The project generates multiple plots to support interpretation and storytelling:

* **Correlation Matrix -**
  Shows relationships between debut rank, longevity and success.

* **Longevity Trends -**
  Time-series analysis of how the average lifespan of hit songs has changed since 2000.

* **Feature Importance Charts -**
  Highlights which variables (e.g., debut rank vs. debut month) most influence predictions.

---

## 8. Results Overview

### A. Peak Position & Longevity (Random Forest)

#### Peak Position Prediction
- **R² Score:** 0.293 *(low predictive results)*
- **MAE:** ~20 chart positions

**Insight:**  
The model cannot predict accurately song’s exact peak position. While debut_rank dominates feature importance (88%), external factors make peak outcomes highly unpredictable from early data alone.

---

#### Weeks on Chart Prediction
- **R² Score:** 0.657 *(moderate predictive results)*
- **MAE:** ~4.3 weeks

**Insight:**  
Longevity is more predictable than peak success. peak_pos (59%) and current_week (32%) are the strongest indicators, so once a song establishes itself initially on charts, its lifespan on the chart becomes easier to estimate.

---

### B. “Fast Burn” Classification

**Task:** Identifying songs that debut strongly but decline quickly.

- **Accuracy:** 97.5%
- **Recall:** 82.5%

**Insight:**  
The model performs exceptionally well at identifying short-lived hits. Capturing 82.5% of true fast-burn tracks, it is effective at spotting songs that fail to keep early success. Although, due to unequal distribution, it might be a result of overfitting.

---

### C. Predicting #1 Hits (Model Comparison)

Predicting #1 hits is especially difficult due to class imbalance, as only 3% of songs reach the top position.

#### Model Performance Comparison

| Metric     | Logistic Regression | Decision Tree |
|-----------|---------------------|---------------|
| Accuracy  | 97.03%              | **98.47%**    |
| Precision | 0.00                | **1.00**      |
| Recall    | 0.00                | **0.49**      |

**Winner:**  Decision Tree

**Analysis:**  
Logistic Regression failed completely by predicting “No” for every song. The Decision Tree successfully identified nearly 50% of #1 hits while maintaining 100% precision (zero false positives).

**Key Indicators:**
- **songs_to_top1:** ~50% importance  
- **has_been_in_top100_before:** ~47% importance  

These results suggest that established artists with a established chart history are significantly more likely to achieve a #1 hit.

---
 
*AI tools were used to explain and understand unknown functions (similarly to how Stack Overflow would be used), as well as to implement unfamiliar functions in standard form (such as saving CSV files with different delimiters). AI tools also assisted in creating markdown files (README and CONTRIBUTIONS) with correct formatting and in establishing an appropriate code structure for the overall project report.*
