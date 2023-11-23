# Optimal Linear Signal Model

## Description
This repository contains the implementation of an unsupervised machine learning model designed to optimize Profit and Loss (PnL) in quantitative finance. This model utilizes linear signals constructed from exogenous variables to maximize the Sharpe ratio.

## Contents
1. `OptimalLinearSignal.py` - Unsupervised machine learning model code.
2. `Optimal Linear Signal Strategy.ipynb` - Example application of the model to a trading strategy.
3. `Optimal_Linear_Signal_explanation.pdf` - Detailed PDF document describing the model and its methodology.
4. `basic_finance_tools.py` - Basic finance functions used in quantitative finance.

## The Model
The model establishes a linear relationship between exogenous variables and the trading signal, aiming to optimize the Sharpe ratio through parameter optimization. The model has been empirically tested on an ETF representing U.S. Treasury bonds, demonstrating its effectiveness.

## Methodology
- The model assumes a linear relationship between the exogenous variables and the signal, as well as between PnL and the signal.
- Utilizes regularization techniques to mitigate overfitting.
- Parameter optimization to maximize the Sharpe ratio.

## Practical Application
The repository includes an example of applying the model in a trading strategy, demonstrating how to train the model on historical data and use it to generate trading signals.

## Getting Started
To use this model, please refer to the instructions in `example.py` and adjust the parameters as per your specific needs.
