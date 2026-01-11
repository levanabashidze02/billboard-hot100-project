import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import warnings
warnings.filterwarnings('ignore')

# Loading Data
df = pd.read_csv('hot-100-current.csv')
print(f"Dataset Shape: {df.shape}")
print(f"Dataset contains {len(df)} total records before any cleaning.")
print(f"Dataset Columns: {df.columns}")

# Filtering only for modern data (after 2000)
df['chart_week'] = pd.to_datetime(df['chart_week'])
df_modern = df[df['chart_week'].dt.year >= 2000].copy()

# Song that have just entered the chart don't have last week stat, so we fill as 101 (out of top100, off-chart)
df_modern['last_week'] = df_modern['last_week'].fillna(101)

# We group by title and performer to calculate stats for each unique song, as two songs might have similar name
song_stats = df_modern.sort_values('chart_week').groupby(['title', 'performer']).agg({
    'chart_week': 'min',              # Debut date
    'current_week': 'first',          # Debut ranking
    'peak_pos': 'min',                # Best ranking
    'wks_on_chart': 'max'             # How long it stayed on chart
}).reset_index()

# Renaming columns so that they have logical names
song_stats.columns = ['title', 'performer', 'debut_date', 'debut_rank', 'true_peak', 'total_weeks']

# We create variable "Mega Hit" - if song reached top 10 ever, it is a mega hit (1), else - just a hit (0)
song_stats['is_mega_hit'] = (song_stats['true_peak'] <= 10).astype(int)

# Extracting months (seasons play a big part - for example, christmas songs will have longer stay on charts cumulatively)
song_stats['debut_month'] = song_stats['debut_date'].dt.month

# Adding the "number for performer" column - is it their first song, how many songs did they have before this? (for further ML analysis)
song_stats = song_stats.sort_values(by=['performer', 'debut_date', 'debut_rank'], ascending=[True, True, True])
song_stats['no_for_performer'] = song_stats.groupby('performer').cumcount() + 1

print(f"\nRows after initial cleaning: {len(song_stats)}")
song_stats = song_stats.dropna()
print(f"Rows after final cleaning: {len(song_stats)}")

try:
    df_modern.to_csv('hot_100_modern.csv',
                      index=False,
                      sep=';',
                      encoding='utf-8-sig',
                      quoting=csv.QUOTE_ALL) # There are some values with ' in the performer/title, so standard to_csv doesn't work properly
    print("Cleaned data saved successfully as 'hot_100_modern.csv'")
except Exception as e:
    print("Error saving cleaned data as a separate csv file.")

try:
    song_stats.to_csv('cleaned_song_stats.csv',
                      index=False,
                      sep=';',
                      encoding='utf-8-sig',
                      quoting=csv.QUOTE_ALL) # Same reasoning
    print("Cleaned data saved successfully as 'cleaned_song_stats.csv'")
except Exception as e:
    print("Error saving cleaned data as a separate csv file.")

print("\n" + "="*5 + " Descriptive Statistics for Cleaned Data " + "="*5)
print(song_stats.describe())
print(f"\nCleaned dataset contains {len(song_stats)} total records after cleaning.")
print(song_stats.columns)
print(song_stats.head())

# Finding outliers
# Focusing on chart longevity, for example
Q1 = song_stats['total_weeks'].quantile(0.25)
Q3 = song_stats['total_weeks'].quantile(0.75)
interquartile = Q3 - Q1
upper_bound = Q3 + 1.5 * interquartile
outliers = song_stats[song_stats['total_weeks'] > upper_bound].sort_values('total_weeks', ascending=False)

print(f"Upper bound for normal longevity: {upper_bound} weeks")
print(f"Number of outliers detected: {len(outliers)}")
print("\nTop 5 Longest Running Songs:")
print(outliers[['title', 'performer', 'total_weeks']].head())

# Exploratory Analysis (Charts & Graphs)

print("\n" + "="*5 + " Visuals (Charts & Graphs) " + "="*5)


# Correlation heatmap - if better debut rank correlates with being a mega hit
try:
    plt.figure(figsize=(8, 6))
    sns.heatmap(song_stats[['debut_rank', 'total_weeks', 'true_peak', 'is_mega_hit']].corr(), annot=True,
                cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()
    # plt.savefig('correlation_heatmap.png')
    print("\nHeatmap displayed successfully.")
except Exception as e:
    print(f"\nSomething went wrong when displaying the heatmap.")


# Time series chart - do song stay on chart longer now compared to in 2000?
song_stats['debut_year'] = song_stats['debut_date'].dt.year
yearly_stats = song_stats.groupby('debut_year')['total_weeks'].mean().reset_index()

try:
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=yearly_stats, x='debut_year', y='total_weeks')
    plt.title('Average Weeks on Chart by Debut Year (2000-2025)')
    plt.show()
    # plt.savefig('time_series_weeks.png')
    print("Time-series chart displayed successfully.")
except Exception as e:
    print(f"Something went wrong when displaying time-series charts.")

# Histogram - songs by peak position
try:
    plt.figure(figsize=(10, 5))
    sns.histplot(song_stats['true_peak'], bins=20, kde=True)
    ax = plt.gca()
    ax.invert_xaxis() # We need to invert X axis, as position closer to 1 is considered highest
    plt.title('Distribution of Peak Positions')
    plt.show()
    # plt.savefig('peak_pos_dist.png')
    print("Histogram displayed successfully.")
except Exception as e:
    print(f"Something went wrong when displaying histogram.")

# Box plot - trends by era (00s, 10s, 20s), longevity specifically
def get_era(year):
    if year < 2010: return '2000s'
    elif year < 2020: return '2010s'
    else: return '2020s'

try:
    song_stats['era'] = song_stats['debut_year'].apply(get_era)
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='era', y='total_weeks', data=song_stats, order=['2000s', '2010s', '2020s'])
    plt.title('Song Longevity by Era')
    plt.show()
    # plt.savefig('boxplot_era.png')
    print("Box plot displayed successfully.")
except Exception as e:
    print(f"Something went wrong when displaying boxplot.")

# Scatter plot - debut vs peak rank

try:
    plt.figure(figsize=(10, 6))
    sns.regplot(x='debut_rank', y='total_weeks', data=song_stats,
                scatter_kws={'alpha': 0.1, 'color': 'green'}, line_kws={'color': 'red'})
    ax = plt.gca()
    ax.invert_xaxis() # Same reasoning as before - better stats (#1 position) on the right
    plt.title('Scatter Plot: Debut Rank vs. Longevity')
    plt.xlabel('Debut Rank')
    plt.ylabel('Total Weeks on Chart')
    plt.show()
    # plt.savefig('regression_peak.png')
    print("Regression plot displayed successfully.")
except Exception as e:
    print(f"Something went wrong when displaying boxplot.")


# ==============================
# MACHINE LEARNING: Predict Best Position
# ==============================
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

print("\n===== MACHINE LEARNING: Peak Position Prediction =====")

# variables and target
X = song_stats[['debut_rank', 'debut_month']]
y = song_stats['true_peak']

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#  Regressor
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation 
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.3f}")

# Analyzing importance
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importance:")
print(importance)

# Prediction chart: Predicted vs Actual
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.3)
plt.plot([1, 100], [1, 100], 'r--')  # perfect prediction line
plt.xlabel("Actual Peak Position")
plt.ylabel("Predicted Peak Position")
plt.title(" Predicted vs Actual Peak Position")
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.show()
# plt.savefig('1_random_forest_peak.png')

# Variable importance chart
plt.figure(figsize=(6,4))
importance.plot(kind='bar', color='yellow')
plt.title("Feature Importance")
plt.ylabel("Importance Score")
plt.xticks(rotation=0)
plt.show()
# plt.savefig('1_feature_importance_peak.png')


# Lets try for an Example - Take an random song amd Predict vs actual peak position
example_song = pd.DataFrame({'debut_rank':[25], 'debut_month':[7]})  # Example  - July debut, rank 25
predicted_peak = model.predict(example_song)
print(f"\nPredicted Peak Position for example song: {predicted_peak[0]:.1f}")



# ==============================
# MACHINE LEARNING: Predict Weeks on Chart
# ==============================

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

print("\n===== MACHINE LEARNING: Predict Weeks on Chart =====")


#   song-level stats
song_stats_ml = df_modern.sort_values('chart_week').groupby(['title', 'performer']).agg({
    'chart_week': 'min',       # debut date
    'current_week': 'first',   # debut rank
    'peak_pos': 'min',         # best chart position
    'wks_on_chart': 'max'      # total weeks on chart
}).reset_index()

# Add debut month feature
song_stats_ml['debut_month'] = song_stats_ml['chart_week'].dt.month


#  Features and target

X = song_stats_ml[['current_week', 'debut_month', 'peak_pos']]
y = song_stats_ml['wks_on_chart']


#  Train/test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#  Regressor
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Predictions 
y_pred = model.predict(X_test)

#  evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.3f}")

# importance variables
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importance:")
print(importance)

# Variable importance chart
plt.figure(figsize=(6,4))
importance.plot(kind='bar', color='orange')
plt.title("Feature Importance")
plt.ylabel("Importance Score")
plt.xticks(rotation=0)
plt.show()
# plt.savefig('2_random_forest_weeks.png')



# Predicted vs actual chart
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.3, color='purple')
plt.plot([0, y_test.max()], [0, y_test.max()], 'r--')  # perfect prediction line
plt.xlabel("Actual Weeks on Chart")
plt.ylabel("Predicted Weeks on Chart")
plt.title("Predicted vs Actual Weeks on Chart")
plt.show()
# plt.savefig('2_feature_importance_weeks.png')


# Lets try for an Example - Take an random song amd Predicted vs Actual Weeks on Chart
example_song = pd.DataFrame({'current_week':[25], 'debut_month':[7], 'peak_pos':[3]}) #same example 
predicted_weeks = model.predict(example_song)
print(f"\nPredicted weeks on chart for example song: {predicted_weeks[0]:.1f}")

# ============================================================
# MACHINE LEARNING Predict if success is long/short-term
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

print("\n================ LONG TERM VS SHORT TERM =================")

# Create classification target
median_weeks = song_stats_ml["wks_on_chart"].median()

song_stats_ml["fast_burn"] = (
    (song_stats_ml["current_week"] <= 20) &
    (song_stats_ml["wks_on_chart"] <= median_weeks)
).astype(int)

# Features and target
X = song_stats_ml[
    [
        "current_week",
        "debut_month",
        "peak_pos"
    ]
]

y = song_stats_ml["fast_burn"]

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# classifier
model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Example prediction
example_song = pd.DataFrame({
    'current_week': [18],
    'debut_month': [6],
    'peak_pos': [5]
})

example_prediction = model.predict(example_song)

print("\nExample song prediction (1 = Fast Burn, 0 = Slow Grower):")
print(example_prediction[0])

# ML3
"""
Billboard Hot 100 Top-1 Prediction Model
Predicts whether a song will reach #1 based on artist history and song characteristics.

Features:
- song_order: Which number song this is for the artist
- songs_to_top1: How many songs artist needed to reach their first #1
- has_been_in_top100_before: Has artist been on Hot 100 before this song (0 or 1)

Target:
- reached_top1: Whether the song reached #1 position (0 or 1)
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score, classification_report)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Create a copy to avoid modifying original
data = song_stats.copy()

# Target: Did the song reach #1?
print("\n Creating target variable: reached_top1")
data['reached_top1'] = (data['true_peak'] == 1).astype(int)

print(f" Songs that reached #1: {data['reached_top1'].sum()} ({data['reached_top1'].mean()*100:.2f}%)")
print(f" Songs that didn't reach #1: {(data['reached_top1']==0).sum()} ({(1-data['reached_top1'].mean())*100:.2f}%)")

# Feature 1: song_order (already exists as 'no_for_performer')
print("\n Feature 1: song_order")
print(" Using existing 'no_for_performer' column from cleaned_song_stats.csv")
data['song_order'] = data['no_for_performer']
print(f" Range: {data['song_order'].min()} to {data['song_order'].max()}")
print(f" Mean: {data['song_order'].mean():.2f}")

# Feature 2: songs_to_top1 - How many songs artist took to reach their first #1
print("\nFeature 2: songs_to_top1")
print("  Calculating how many songs each artist took to reach their first #1")

# Get songs that reached #1
top1_songs = data[data['reached_top1'] == 1].copy()

# Find first #1 song number for each artist
first_top1 = top1_songs.groupby('performer')['no_for_performer'].min()

# Convert to dictionary
first_top1_dict = first_top1.to_dict()

print(f"  {len(first_top1_dict)} artists have reached #1")

# For each song:
# If artist reached #1 song number of first #1
# If artist never reached #1 0
data['songs_to_top1'] = (
    data['performer']
    .map(first_top1_dict)
    .fillna(0)
    .astype(int)
)

# Show statistics (only for artists who reached #1)
artists_with_top1 = (data['songs_to_top1'] > 0).sum()
songs_to_first_top1 = data[data['songs_to_top1'] > 0]['songs_to_top1']

print(f"  Songs from artists who reached #1: {artists_with_top1}")
print(f"  Songs from artists who never reached #1: {len(data) - artists_with_top1}")
print(f"  Average song number of first #1: {songs_to_first_top1.mean():.1f}")
print(f"  Median song number of first #1: {songs_to_first_top1.median():.1f}")
print(f"  Min: {songs_to_first_top1.min()}, Max: {songs_to_first_top1.max()}")

# Feature 3: has_been_in_top100_before - Binary feature
print("\nFeature 3: has_been_in_top100_before")
print("  Creating binary indicator for returning artists...")
data['has_been_in_top100_before'] = (data['song_order'] > 1).astype(int)

debut_count = (data['has_been_in_top100_before'] == 0).sum()
return_count = (data['has_been_in_top100_before'] == 1).sum()
print(f"  Debut songs (first time on chart): {debut_count} ({debut_count/len(data)*100:.1f}%)")
print(f"  Songs from returning artists: {return_count} ({return_count/len(data)*100:.1f}%)")

# Correlations
feature_cols = ['song_order', 'songs_to_top1', 'has_been_in_top100_before']
corr_matrix = data[feature_cols + ['reached_top1']].corr()
print(corr_matrix)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Exploratory Data Analysis: Features for Top-1 Prediction', fontsize=16, fontweight='bold', y=1.00)

# 1. Correlation heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0, 0], fmt='.3f', square=True, cbar_kws={'shrink': 0.8})
axes[0, 0].set_title('Feature Correlation Heatmap', fontsize=12, fontweight='bold')

# 2. Target distribution
target_counts = data['reached_top1'].value_counts()
axes[0, 1].bar(['Did Not Reach #1', 'Reached #1'], target_counts.values, color=['#3498db', '#9b59b6'])
axes[0, 1].set_title('Distribution of Target Variable', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Number of Songs')
for i, v in enumerate(target_counts.values):
    axes[0, 1].text(i, v + 100, str(v), ha='center', fontweight='bold')

# 3. Song order distribution by target
data.boxplot(column='song_order', by='reached_top1', ax=axes[0, 2])
axes[0, 2].set_title('Song Order by Top-1 Achievement', fontsize=12, fontweight='bold')
axes[0, 2].set_xlabel('Reached Top-1')
axes[0, 2].set_ylabel('Song Order')
axes[0, 2].set_xticklabels(['No', 'Yes'])
plt.sca(axes[0, 2])

# 4. Success rate by artist experience
cross_tab = pd.crosstab(data['has_been_in_top100_before'], data['reached_top1'], normalize='index') * 100
x_pos = [0, 1]
width = 0.35
axes[1, 0].bar([p - width/2 for p in x_pos], cross_tab[0], width, label='Did Not Reach #1', color='#3498db')
axes[1, 0].bar([p + width/2 for p in x_pos], cross_tab[1], width, label='Reached #1', color='#9b59b6')
axes[1, 0].set_title('Top-1 Success Rate: Debut vs Returning Artists', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Artist Type')
axes[1, 0].set_ylabel('Percentage (%)')
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(['Debut Artist', 'Returning Artist'])
axes[1, 0].legend()
axes[1, 0].set_ylim(0, 100)

# 5. Songs to Top-1 distribution
songs_to_top1_valid = data[data['songs_to_top1'] > 0]['songs_to_top1']
axes[1, 1].hist(songs_to_top1_valid, bins=30, color='#3498db', edgecolor='black', alpha=0.7)
axes[1, 1].set_title('Songs Needed to Reach First #1', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Number of Songs')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].axvline(songs_to_top1_valid.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {songs_to_top1_valid.mean():.1f}')
axes[1, 1].legend()

# 6. Song order histogram
axes[1, 2].hist(data['song_order'], bins=50, color='#9b59b6', edgecolor='black', alpha=0.7)
axes[1, 2].set_title('Distribution of Song Order', fontsize=12, fontweight='bold')
axes[1, 2].set_xlabel('Song Order (1st, 2nd, 3rd... for artist)')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].set_xlim(0, 50)

plt.tight_layout()
# plt.savefig('eda_top1_prediction.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: eda_top1_prediction.png")
plt.show()

# ============================================================================
# Preparing data for ML
# ============================================================================

# Select features and target
X = data[feature_cols].values
y = data['reached_top1'].values

print(f"\n Dataset overview:")
print(f"  Total samples: {len(X)}")
print(f"  Number of features: {len(feature_cols)}")
print(f"  Features: {feature_cols}")

# Train-test split (80/20)
print("\nSplitting data into train (80%) and test (20%) sets")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
print(f"  Training #1 rate: {y_train.mean()*100:.2f}%")
print(f"  Test #1 rate: {y_test.mean()*100:.2f}%")

# ============================================================================
# Logistic regression
# ============================================================================

# Scale features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_lr = lr_model.predict(X_test_scaled)

# Evaluation
lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_precision = precision_score(y_test, y_pred_lr, zero_division=0)
lr_recall = recall_score(y_test, y_pred_lr, zero_division=0)

print(f"  Accuracy:  {lr_accuracy:.4f} ({lr_accuracy*100:.2f}%)")
print(f"  Precision: {lr_precision:.4f}")
print(f"  Recall:    {lr_recall:.4f}")

print("\n Classification Report:")
print(classification_report(y_test, y_pred_lr, 
                            target_names=['Did Not Reach #1', 'Reached #1'],
                            zero_division=0))

print("Confusion Matrix:")
lr_cm = confusion_matrix(y_test, y_pred_lr)
print(lr_cm)
print(f"  True Negatives: {lr_cm[0,0]} | False Positives: {lr_cm[0,1]}")
print(f"  False Negatives: {lr_cm[1,0]} | True Positives: {lr_cm[1,1]}")

# Feature coefficients
print("\nFeature Coefficients:")
lr_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': lr_model.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)
print(lr_importance.to_string(index=False))

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['No #1', 'Reached #1'],
            yticklabels=['No #1', 'Reached #1'])
axes[0].set_title('Logistic Regression - Confusion Matrix', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

colors = ['#3498db' if x < 0 else '#9b59b6' for x in lr_importance['Coefficient']]
axes[1].barh(lr_importance['Feature'], lr_importance['Coefficient'], color=colors)
axes[1].set_title('Feature Coefficients', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Coefficient Value')
axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.8)

plt.tight_layout()
# plt.savefig('logistic_regression_results.png', dpi=300, bbox_inches='tight')
print("\n Saved: logistic_regression_results.png")
plt.show()

# ============================================================================
# Decision tree
# ============================================================================

# Train model
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=20, min_samples_leaf=10)
dt_model.fit(X_train, y_train)

# Predictions
y_pred_dt = dt_model.predict(X_test)

# Evaluation
dt_accuracy = accuracy_score(y_test, y_pred_dt)
dt_precision = precision_score(y_test, y_pred_dt, zero_division=0)
dt_recall = recall_score(y_test, y_pred_dt, zero_division=0)

print(f"  Accuracy:  {dt_accuracy:.4f} ({dt_accuracy*100:.2f}%)")
print(f"  Precision: {dt_precision:.4f}")
print(f"  Recall:    {dt_recall:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt, 
                            target_names=['Did Not Reach #1', 'Reached #1'],
                            zero_division=0))

print("Confusion Matrix:")
dt_cm = confusion_matrix(y_test, y_pred_dt)
print(dt_cm)
print(f"  True Negatives: {dt_cm[0,0]} | False Positives: {dt_cm[0,1]}")
print(f"  False Negatives: {dt_cm[1,0]} | True Positives: {dt_cm[1,1]}")

# Feature importance
print("\nFeature Importance:")
dt_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': dt_model.feature_importances_
}).sort_values('Importance', ascending=False)
print(dt_importance.to_string(index=False))

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['No #1', 'Reached #1'],
            yticklabels=['No #1', 'Reached #1'])
axes[0].set_title('Decision Tree - Confusion Matrix', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

axes[1].barh(dt_importance['Feature'], dt_importance['Importance'], color='#9b59b6')
axes[1].set_title('Feature Importance', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Importance Score')

plt.tight_layout()
# plt.savefig('decision_tree_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: decision_tree_results.png")
plt.show()

# ============================================================================
# Model comparison
# ============================================================================

comparison_df = pd.DataFrame({
    'Logistic Regression': [lr_accuracy, lr_precision, lr_recall],
    'Decision Tree': [dt_accuracy, dt_precision, dt_recall]
}, index=['Accuracy', 'Precision', 'Recall'])

print("\nPerformance Comparison:")
print(comparison_df.round(4))

# Determine winner
print("\nBEST MODEL:")
if lr_accuracy > dt_accuracy:
    print(f"  Logistic Regression wins!")
    print(f"  Accuracy: {lr_accuracy:.4f} vs {dt_accuracy:.4f}")
    print(f"  Difference: +{(lr_accuracy - dt_accuracy)*100:.2f}%")
elif dt_accuracy > lr_accuracy:
    print(f"  Decision Tree wins!")
    print(f"  Accuracy: {dt_accuracy:.4f} vs {lr_accuracy:.4f}")
    print(f"  Difference: +{(dt_accuracy - lr_accuracy)*100:.2f}%")
else:
    print(f"  It's a tie!")
    print(f"  Both models: {lr_accuracy:.4f}")

# Comparison chart
fig, ax = plt.subplots(figsize=(10, 6))
comparison_df.plot(kind='bar', ax=ax, color=['#3498db', '#9b59b6'], width=0.7)
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_ylabel('Score')
ax.set_xlabel('Metric')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(title='Model')
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
# plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: model_comparison.png")
plt.show()
