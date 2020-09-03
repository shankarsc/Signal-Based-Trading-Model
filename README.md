# signal-based_trading_kit

File contains functions scatter_plot(), scatter_prediction(), assessTable(), adjustMetric(), signal_prediction() which determines if the predictor variables listed in the formula for OLS regression is useful in predicting changes in the response price of an asset.

scatter_plot() function returns the scatter matrix of the predictor and response variables and outputs the correlations.

scatter_prediction() function compares the output of the response variable against the actual response to determine if the prediction is accurate.

signal_prediction() function returns the P&L, Sharpe Ratio, and Maximum Drawdown if strategy was applied and compares it to a Buy & Hold strategy.
