import pandas as pd
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

def scatter_plot(dataset):
    """
    Generates scatter (correlation) matrix among all markets to observe the correlation coefficients.
    """
    axes = scatter_matrix(dataset, alpha=0.5, diagonal='kde', figsize=(12,6))

    # Annotates correlations of the predictor asset returns to the response asset returns
    # to the scatter plot.
    corr = dataset.corr().values
    for i, j in zip(*plt.np.triu_indices_from(axes, k=1)):
        axes[i, j].annotate("Correlation: %.3f" %corr[i,j], (0.4, 0.9), xycoords='axes fraction', ha='center', va='center')

    plt.show()

def scatter_prediction(dataset, model):
    """
    Generates the scatter plot between the the returns of the response variable
    and the actual returns of asset.
    """
    # Predicts the returns of the response asset based on the model.
    dataset['Predicted_Y'] = model.predict(dataset)

    # Returns the correlations of the returns of the response variable to 
    # the actual returns of the asset
    plt.figure(figsize=(12,5))
    plt.scatter(dataset.iloc[:,0], dataset['Predicted_Y'])
    correlation = dataset.iloc[:,0].corr(dataset['Predicted_Y'])
    plt.title("Correlation: %.3f" % correlation, fontsize=14)
    plt.xlabel(dataset.iloc[:,0].name)
    plt.ylabel(dataset['Predicted_Y'].name)

def adjustedMetric(dataset, model, num_predictors, yname):
    """
    Measure performance of model using statistical metrics - RMSE, Adjusted R^2.

    Allocating predictor variable from formula to be the constant.
    Note: Prob (F-statistic) <= 0.05 to accept alternative hypothesis.
    Alternative hypothesis - Recognises that at least one of the predictor variables are useful.
    Current model is better fitted than intercept-only model. 
    """
    # Total sum of squares.
    SST = ((dataset[yname]-dataset[yname].mean())**2).sum()

    # Sum of squared residuals.
    SSR = ((dataset['Predicted_Y']-dataset[yname].mean())**2).sum()

    # 
    SSE = ((dataset[yname]-dataset['Predicted_Y'])**2).sum()

    # Coefficient of determination is a measure that provides information about the goodness of fit of a model.
    r2 = 1 - SSR/SST
    adjust_r2 = 1 - (r2)*(dataset.shape[0]-1)/(dataset.shape[0]-num_predictors-1)

    # Root Mean Squared Error
    RMSE = (SSE/(dataset.shape[0]-num_predictors-1))**0.5
    return adjust_r2, RMSE 

def assessTable(test_dataset, train_dataset, model, num_predictors, yname):
    """
    Training Model Evaluation
    RMSE = sqrt(SSE/(n-k-1)), k = number of predictors.
    Adjusted R^2 measures percentage of variation of a response that is explained by model.
    Both RMSE and Adjusted R^2 has degrees of freedom as denominator.
    """
    
    # If Adjusted R^2 and RMSE in Train not approx. equal to Test, model is overfitted.
    r2train, RMSEtrain = adjustedMetric(train_dataset, model, num_predictors, yname)
    r2test, RMSEtest = adjustedMetric(test_dataset, model, num_predictors, yname)

    assessment = pd.DataFrame(index=['R2', 'RMSE'], columns=['Training Dataset', 'Test Dataset'])
    assessment['Training Dataset'] = [r2train, RMSEtrain]
    assessment['Test Dataset'] = [r2test, RMSEtest]

    return assessment

def signal_prediction(model, dataset):
    """
    Returns the profit/loss margin from application of trading signal.
    """
    # Evaluates the direction of the trade depending on the signal generated by model
    dataset['Direction'] = [1 if signal>0 else -1 for signal in dataset['Predicted_Y']]
    dataset['Profit (Signal)'] = dataset['Direction']*dataset.iloc[:, 0]

    # Returns the portfolio balance after applying signal strategy
    dataset['Equity (Signal)'] = dataset['Profit (Signal)'].cumsum()
    dataset['Equity (B&H)'] = dataset.iloc[:, 0].cumsum()

    # Plotting the performance of the strategy applied in dataset.
    plt.figure(figsize=(12,6))
    plt.title('Performance of Strategy from Dataset')
    plt.plot(dataset['Equity (Signal)'], color='green', label='Signal-Based Strategy')
    plt.plot(dataset['Equity (B&H)'], color='red', label='Buy and Hold Strategy')
    plt.legend()
    plt.show()

    # Prints the cumulative profit made from the lifetime of the strategy.
    print(
        'Total Profit Made From Signal Over B&H: ' 
        + str(round(dataset['Equity (Signal)'].iloc[-1]-dataset['Equity (B&H)'].iloc[-1], 4))
        )

    # Returns the daily and annualised Sharpe Ratio of the signal-based strategy.
    daily_sr = dataset['Profit (Signal)'].mean()/dataset['Profit (Signal)'].std(ddof=1)
    ann_daily_sr = (365**0.5) * dataset['Profit (Signal)'].mean()/dataset['Profit (Signal)'].std(ddof=1)

    print(
        'Daily Sharpe Ratio: ', round(daily_sr, 4), 
        '\nYearly Sharpe Ratio: ', round(ann_daily_sr, 4)
    )  

    # Returns the maximum drawdown of the signal-based strategy.
    dataset['Peak Equity (Signal)'] = dataset['Equity (Signal)'].cummax()
    dataset['Drawdown'] = (dataset['Peak Equity (Signal)'] - dataset['Equity (Signal)'])/dataset['Peak Equity (Signal)']
    print(
    'Maximum Drawdown: ', round(dataset['Drawdown'].max(), 4),
    '\nIndex of Max. Drawdown: ', dataset['Drawdown'].idxmax()
    )

    return dataset   