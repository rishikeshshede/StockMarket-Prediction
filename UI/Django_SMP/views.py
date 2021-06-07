from django.shortcuts import render
from bson.json_util import dumps
import numpy as np 
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from pmdarima import auto_arima
import csv, io
from django.contrib.staticfiles.storage import staticfiles_storage
from django.conf import settings
from statsmodels.tsa.stattools import adfuller
import os
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
## Create the Stacked LSTM model
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import LSTM
import math
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from itertools import chain

# Create your views here.
def google(request):
    # Current SMP Code

    # print (settings.STATICFILES_DIRS)
    url = os.path.join(settings.STATICFILES_DIRS[0],'datasets/GOOGL.csv')
    df = pd.read_csv(url, parse_dates=True)
    dates = df['Date']
    df = pd.read_csv(url, index_col='Date', parse_dates=True)
    url = os.path.join(settings.STATICFILES_DIRS[0],'datasets/saved_model.h5')
    model = keras.models.load_model(url)
    # df = pd.read_csv(url, index_col='Date')
    # df = df.diff()
    df = df.dropna()
    # print('Shape of data', df.shape)
    # print(df.head())
    dates = dates[:int(len(df)*.8401)]
    df1=df.reset_index()['Close']

    scaler=MinMaxScaler(feature_range=(0,1))
    df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

    training_size=int(len(df1)*0.75)
    # test_size=len(df1)-training_size
    test_size=int(len(df1)*.1)
    dataset_length = training_size+test_size
    train_data,test_data=df1[0:training_size],df1[training_size:dataset_length]

    time_step = 150
    x_train, y_train = create_dataset(train_data, time_step)
    x_test, y_test = create_dataset(test_data, time_step)

    x_train =x_train.reshape(x_train.shape[0],x_train.shape[1] , 1)
    x_test = x_test.reshape(x_test.shape[0],x_test.shape[1] , 1)

    train_predict=model.predict(x_train)
    test_predict=model.predict(x_test)

    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)

    ### Plotting 
    # shift train predictions for plotting
    look_back=150
    trainPredictPlot = np.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(df1)
    testPredictPlot[:, :] = np.nan
    # testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
    testPredictPlot[len(train_predict)+(look_back*2)+1:dataset_length-1, :] = test_predict

    x_input=test_data[len(test_data)-150:].reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    nod=60
    lst_output=[]
    n_steps=150
    i=0
    while(i<nod):
        
        if(len(temp_input)>150):
            #print(temp_input)
            x_input=np.array(temp_input[1:])
            # print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            # print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            # print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            # print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1

    day_new=np.arange(1,151)
    day_pred=np.arange(151,151+nod)

    df3=df1[100:dataset_length].tolist()
    df3.extend(lst_output)
    # plt.plot(df3[500:])
    df3=scaler.inverse_transform(df3).tolist()
    df4=df1[100:dataset_length+60]
    df4=scaler.inverse_transform(df4)

    pred_Future = df3
    test_Actual = df4

    # print(size(pred_Future))
    # print(size(test_Actual))

    all_dates = dates.array

    data_final = []
    for idx, val in enumerate(df4):
      val = val.tolist()
      val.insert(0, all_dates[idx])
      data_final.append(val)

    data_actual = []
    for idx, val in enumerate(df3):
      # val = val.tolist()
      val.insert(0, all_dates[idx])
      data_actual.append(val)

    # print("________________________________________________________________________________________________________")
    # print(type(data_final))


    data = [
      {
        'name' : 'Google Actual Data',
        'points' : data_final
      },
      {
        'name' : 'Google Predicted Data',
        'points' : data_actual
      }
    ]

    # test_Actual_Flat = list(chain.from_iterable(test_Actual))
    # data_graph = test_Actual_Flat

    # print(dates.array)

    # pred_Future_Flat = list(chain.from_iterable(pred_Future))
    # data_graph = pred_Future_Flat

    # data_graph = []
    # data_graph.append(test_Actual_Flat)
    # data_graph.append(pred_Future_Flat)

    # print(data)
    dataJSON = dumps(data)
    return render(request, 'index.html', context = {'data': dataJSON})

def home(request):
    return render(request, 'home.html')

def nifty(request):
  url = os.path.join(settings.STATICFILES_DIRS[0],'datasets/nifty50.csv')
  df = pd.read_csv(url, parse_dates=True)
  dates = df['Date']
  df = pd.read_csv(url, index_col='Date', parse_dates=True)
  url = os.path.join(settings.STATICFILES_DIRS[0],'datasets/nifty_model.h5')
  print("__________________________________________________________________________")
  print(url)
  model = keras.models.load_model(url)

  df = df.dropna()
  # CONST VALUE
  steps = 150

  df1=df.reset_index()['Close']

  scaler=MinMaxScaler(feature_range=(0,1))
  df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

  training_size=int(len(df1)*0.75)
  test_size=int(len(df1)*.15)
  dataset_length = training_size+test_size
  train_data,test_data=df1[0:training_size],df1[training_size:dataset_length]

  time_step = 150
  x_train, y_train = create_dataset(train_data, time_step)
  x_test, y_test = create_dataset(test_data, time_step)

  x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
  x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1) 

  # model = keras.models.load_model('nifty_model.h5')

  train_predict = model.predict(x_train)
  test_predict = model.predict(x_test)

  train_predict=scaler.inverse_transform(train_predict)
  test_predict=scaler.inverse_transform(test_predict)

  look_back = steps
  trainPredictPlot = np.empty_like(df1)
  trainPredictPlot[:, :] = np.nan
  trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
  # shift test predictions for plotting
  testPredictPlot = np.empty_like(df1)
  testPredictPlot[:, :] = np.nan
  # testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
  testPredictPlot[len(train_predict)+(look_back*2)+1:dataset_length-1, :] = test_predict

  x_input=test_data[len(test_data)-steps:].reshape(1,-1)

  temp_input=list(x_input)
  temp_input=temp_input[0].tolist()

  nod=21
  lst_output=[]
  n_steps=150
  i=0
  while(i<nod):
      
      if(len(temp_input)>150):
          x_input=np.array(temp_input[1:])
          x_input=x_input.reshape(1,-1)
          x_input = x_input.reshape((1, n_steps, 1))
          yhat = model.predict(x_input, verbose=0)
          temp_input.extend(yhat[0].tolist())
          temp_input=temp_input[1:]
          lst_output.extend(yhat.tolist())
          i=i+1
      else:
          x_input = x_input.reshape((1, n_steps,1))
          yhat = model.predict(x_input, verbose=0)
          temp_input.extend(yhat[0].tolist())
          lst_output.extend(yhat.tolist())
          i=i+1

  day_new=np.arange(1,151)
  day_pred=np.arange(151,151+nod)

  df3=df1[800:dataset_length].tolist()
  df3.extend(lst_output)
  df3=scaler.inverse_transform(df3).tolist()
  df4=df1[800:dataset_length+20]
  df4=scaler.inverse_transform(df4)

  all_dates = dates.array

  data_final = []
  for idx, val in enumerate(df4):
    val = val.tolist()
    val.insert(0, all_dates[idx])
    data_final.append(val)

  data_actual = []
  for idx, val in enumerate(df3):
    # val = val.tolist()
    val.insert(0, all_dates[idx])
    data_actual.append(val)

  data = [
    {
      'name' : 'Nifty Actual Data',
      'points' : data_final
    },
    {
      'name' : 'Nifty Predicted Data',
      'points' : data_actual
    }
  ]

  dataJSON = dumps(data)
  return render(request, 'nifty.html', context = {'data': dataJSON})

def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----149   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)