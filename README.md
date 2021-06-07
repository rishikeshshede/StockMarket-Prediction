# Stock Price & Index trend analysis

This Project aims to provide a basic analysis/prediction of the stock and index the user wants to possibly invest in or just want to analyze. You can see the possible upward or downward trend in the prices of the stock in a certain period(say 30-60 days). 

Note: This project is currently built only for learning purpose and not meant to be used in real life trading in stock market.

## Run this project:
### Clone the repository
Use Jupiter Notebook or Google Colab (used by us) or any other viable option:

- **Open [`Google Colab`](https://colab.research.google.com/notebooks/intro.ipynb?utm_source=scs-index#scrollTo=GJBs_flRovLc)** in your brower
- Sign In using your google account
- Click on `File -> Upload notebook` and upload any one file:
  - [Index_Prediction_using_Stacked_LSTM.ipynb](Index_Prediction_using_Stacked_LSTM.ipynb) for Index analysis.
  - [Stock_Price_Prediction_using_Stacked_LSTM_GOOGL.ipynb](Stock_Price_Prediction_using_Stacked_LSTM_GOOGL.ipynb) for Stock Price analysis. 
- Run all the cells upto downloading the trained model, which will be used to get the output in the frontend.
- Now [`start the Django server`](UI/README.md) to run the UI.

## Outputs
Orange line indicates predicted values.

<img src="https://github.com/rishi4rks/StockMarket-Prediction/blob/main/GOOGL-prediction.jpeg" alt="Google Stock Analysis">&nbsp;&nbsp;

<img src="https://github.com/rishi4rks/StockMarket-Prediction/blob/main/Nifty50-prediction.jpeg" alt="Nifty50 Index Analysis">&nbsp;&nbsp;
