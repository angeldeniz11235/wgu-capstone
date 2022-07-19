
#if pickle is not imported, then import it
if not 'pickle' in globals():
    import pickle
import sys
# increase the recursion limit
sys.setrecursionlimit(10000)
    
# load alpaca_bars from the pickle file 
data = open("apple_bar_data.pkl", "rb")
apple_bars_df = pickle.load(data)
data.close()


# create plot of apple_bars_df
# plotly imports
import plotly.graph_objects as go
import plotly.express as px

# SPY bar data candlestick plot
candlestick_fig = go.Figure(data=[go.Candlestick(x=apple_bars_df.index,
                open=apple_bars_df['open'],
                high=apple_bars_df['high'],
                low=apple_bars_df['low'],
                close=apple_bars_df['close'])])

# calculating 13 day SMA using pandas rolling mean
sma = apple_bars_df['close'].rolling(13).mean().dropna()

# creating a line plot for our sma
sma_fig = px.line(x=sma.index, y=sma)

# adding both plots onto one chart
fig = go.Figure(data=candlestick_fig.data + sma_fig.data)

# displaying our chart
fig.show()