import pandas as pd
import numpy as np
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
    song_stats.to_csv('cleaned_song_stats.csv',
                      index=False,
                      sep=';',
                      encoding='utf-8-sig',
                      quoting=csv.QUOTE_ALL) # There are some values with ' in the performer/title, so standard to_csv doesn't work properly
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
song_stats_ml = df.sort_values('chart_week').groupby(['title', 'performer']).agg({
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

