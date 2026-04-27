import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_clean_data(file_path):
    """Loads the preprocessed data."""
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def plot_market_trend(df):
    """Shows how the average stock price moved over time."""
    plt.figure(figsize=(12, 6))
    market_avg = df.groupby('Date')['Close'].mean()
    market_avg.plot()
    plt.title('Market Average Close Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid(True)
    plt.show()

def plot_returns_distribution(df):
    """Histogram of daily returns to check for normality/volatility."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Daily_Return'], bins=100, kde=True)
    plt.title('Distribution of Daily Returns')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.show()

def plot_top_volume_stocks(df):
    """Bar chart for the top 10 most traded stocks."""
    plt.figure(figsize=(10, 6))
    top_stocks = df.groupby('Symbol')['Volume'].mean().sort_values(ascending=False).head(10)
    top_stocks.plot(kind='bar')
    plt.title('Top 10 Stocks by Average Volume')
    plt.xlabel('Stock Symbol')
    plt.ylabel('Average Volume')
    plt.show()

def plot_correlation_heatmap(df):
    """Heatmap to see how different features (Open, High, Vol, etc.) relate."""
    plt.figure(figsize=(8, 6))
    # Select only numeric columns for correlation
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return']
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Heatmap')
    plt.show()

def plot_monthly_seasonality(df):
    """Shows average returns by month to detect seasonal trends."""
    plt.figure(figsize=(10, 6))
    monthly_avg = df.groupby('Month')['Daily_Return'].mean()
    monthly_avg.plot(kind='bar', color='skyblue')
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Average Daily Return by Month')
    plt.xlabel('Month')
    plt.ylabel('Avg Return')
    plt.xticks(range(0, 12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.show()

def run_analysis_pipeline(file_path):
    """Executes all analysis functions."""
    df = load_clean_data(file_path)
    
    plot_market_trend(df)
    
    plot_returns_distribution(df)
    
    plot_top_volume_stocks(df)
    
    plot_correlation_heatmap(df)
    
    plot_monthly_seasonality(df)

if __name__ == "__main__":
    run_analysis_pipeline('cleaned.csv')