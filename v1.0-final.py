import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Loading Data
df = pd.read_csv('hot-100-current.csv')

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

print(song_stats.head())

# Correlation heatmap - if better debut rank correlates with being a mega hit
plt.figure(figsize=(8, 6))
sns.heatmap(song_stats[['debut_rank', 'total_weeks', 'true_peak', 'is_mega_hit']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
# plt.savefig('correlation_heatmap.png')

# Time series chart - do song stay on chart longer now compared to in 2000?
song_stats['debut_year'] = song_stats['debut_date'].dt.year
yearly_stats = song_stats.groupby('debut_year')['total_weeks'].mean().reset_index()

plt.figure(figsize=(10, 5))
sns.lineplot(data=yearly_stats, x='debut_year', y='total_weeks')
plt.title('Average Weeks on Chart by Debut Year (2000-2025)')
plt.show()
# plt.savefig('time_series_weeks.png')

# Histogram - songs by peak position
plt.figure(figsize=(10, 5))
sns.histplot(song_stats['true_peak'], bins=20, kde=True)
plt.title('Distribution of Peak Positions')
plt.show()
# plt.savefig('peak_pos_dist.png')

# Box plot - trends by era (00s, 10s, 20s), longevity especially
def get_era(year):
    if year < 2010: return '2000s'
    elif year < 2020: return '2010s'
    else: return '2020s'

song_stats['era'] = song_stats['debut_year'].apply(get_era)
plt.figure(figsize=(10, 5))
sns.boxplot(x='era', y='total_weeks', data=song_stats, order=['2000s', '2010s', '2020s'])
plt.title('Song Longevity by Era')
plt.show()
# plt.savefig('boxplot_era.png')

