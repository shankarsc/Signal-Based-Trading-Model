import pandas as pd
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt

# Measure performance of model using statistical metrics - RMSE, Adjusted R^2
def adjustedMetric(data, model, num_predictors, yname):
    """
    Allocating predictor variable from formula to be the constant
    Note: Prob (F-statistic) <= 0.05 to accept alternative hypothesis
    Alternative hypothesis - Recognises that at least one of the predictor variables are useful
    Current model is better fitted than intercept-only model.
    """
    
    data['yhat'] = model.predict(data)

    SST = ((data[yname]-data[yname].mean())**2).sum()
    SSR = ((data['yhat'] - data[yname].mean())**2).sum()
    SSE = ((data[yname]-data['yhat'])**2).sum()

    r2 = SSR/SST
    adjust_r2 = 1 - (1-r2)*(data.shape[0]-1)/(data.shape[0]-num_predictors-1)

    RMSE = (SSE/(data.shape[0]-num_predictors-1))**0.5
    return adjust_r2, RMSE 

def assessTable(test_dataset, train_dataset, model, num_predictors, yname):
    """
    Training Model Evaluation
    RMSE = sqrt(SSE/(n-k-1)), k = number of predictors
    Adjusted R^2 measures percentage of variation of a response that is explained by model
    Both RMSE and Adjusted R^2 has degrees of freedom as denominator
    If Adjusted R^2 and RMSE in Train not approx. equal to Test, model is overfitted.
    """

    r2test, RMSEtest = adjustedMetric(test_dataset, model, num_predictors, yname)
    r2train, RMSEtrain = adjustedMetric(train_dataset, model, num_predictors, yname)

    assessment = pd.DataFrame(index=['R2', 'RMSE'])
    assessment['Training Dataset'] = [r2train, RMSEtrain]
    assessment['Test Dataset'] = [r2test, RMSEtest]

    return assessment

def prediction(formula, model, dataset):
    """
    Returns the profit/loss margin from application of trading signal
    """

    dataset['Predicted Y'] = model.predict(dataset)
    dataset['Trade'] = [1 if signal>0 else -1 for signal in dataset['Predicted Y']]
    dataset['Profit From Signal'] = dataset['Trade']*dataset.iloc[:, 0]

    dataset['Cum. Profit From Signal'] = dataset['Profit From Signal'].cumsum()
    dataset['Cum. Profit From B&H'] = dataset.iloc[:, 0].cumsum()

    # Plotting the performance of the strategy applied in dataset
    plt.figure(figsize=(6,6))
    plt.title('Performance of Strategy from Dataset')
    plt.plot(dataset['Cum. Profit From Signal'].values, color='green', label='Signal-Based Strategy')
    plt.plot(dataset['Cum. Profit From B&H'].values, color='red', label='Buy and Hold Strategy')
    plt.legend()
    plt.show()

    return (print('Total Profit Made From Signal Over B&H: ' + str(dataset['Cum. Profit From Signal'].iloc[-1]-dataset['Cum. Profit From B&H'].iloc[-1])))

