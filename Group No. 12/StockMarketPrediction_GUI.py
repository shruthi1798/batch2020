from tkinter import *
import os

def prediction(a, b):

  import quandl, datetime, math
  import numpy as np
  from sklearn import preprocessing, svm, model_selection
  from sklearn.linear_model import LinearRegression
  from sklearn.tree import DecisionTreeRegressor
  import matplotlib.pyplot as plt
  from matplotlib import style
  import pandas as pd

  if a=='a':
    df = quandl.get("WIKI/AAPL")
    company= "APPLE's Stockmarket"
  elif a=='g':
    df = quandl.get("WIKI/GOOGL")
    company="GOOGLE's stockmarket"
  elif a=='am':
    df = quandl.get("WIKI/AMZN")
    company="AMAZON's stockmarket"


  if b=='s':
    clf=svm.SVR(kernel='linear')
  elif b=='l':
    clf=LinearRegression(n_jobs=-1)
  elif b=='d':
    clf=DecisionTreeRegressor()

  style.use('ggplot')


  df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
  df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
  df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
  df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
  

  # print(df.head())
  # print(df.tail())

  forecast_col = 'Adj. Close'
  df.fillna(-99999, inplace=True)
  if a=='WIKI/AAPL':
    forecast_out1=int(math.ceil(0.0001*len(df)))
  elif a=='WIKI/GOOGL':
    forecast_out1=int(math.ceil(0.01*len(df)))
  else:
    forecast_out1=int(math.ceil(0.0018*len(df)))
  print(len(df))
  print(0.7*len(df))
  print(math.ceil(0.7*len(df)))
  print(int(math.ceil(0.7*len(df))))
  print(forecast_out1)
  forecast_out = forecast_out1				
  df['label'] = df[forecast_col].shift(-forecast_out)

  X = np.array(df.drop(['label'], 1))
  X = preprocessing.scale(X)
  X_lately = X[-forecast_out:]
  X = X[:-forecast_out]

  df.dropna(inplace=True)

  y = np.array(df['label'])

  X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

  clf.fit(X_train, y_train)

  accuracy = clf.score(X_test, y_test)
  accuracy=accuracy*100

  forecast_set = clf.predict(X_lately)

  for a in forecast_set:
    print(a)
  print(accuracy, forecast_out)

  msg5="The next few days "+str(company)+" price is $\n"+str(forecast_set)+"\n and its confidence is "+str(accuracy)+"%"
  test(msg5)



  df['Forecast']=np.nan
  ld=df.iloc[-1].name
  lu=ld.timestamp()
  od=86400
  nu=lu+od
  for i in forecast_set:
    nd=datetime.datetime.fromtimestamp(nu)
    nu+=od
    df.loc[nd]=[np.nan for _ in range(len(df.columns)-1)] +[i]

  df['Adj. Close'].plot()
  df['Forecast'].plot()
  plt.legend(loc=4)
  plt.xlabel('Date')
  plt.ylabel('Price')

  plt.savefig('newimg.png')
  plt.show()
  df.drop(df.index, inplace=True)

def lgooglecall():
  clear_widget(lbl)
  prediction('g','l')
def lapplecall():
  clear_widget(lbl)
  prediction('a','l')
def lamazoncall():
  clear_widget(lbl)
  prediction('am', 'l')

def sgooglecall():
  clear_widget(lbl)
  prediction('g' ,'s')
def sapplecall():
  clear_widget(lbl)
  prediction('a' ,'s')
def samazoncall():
  clear_widget(lbl)
  prediction('am' ,'s')

def dgooglecall():
  clear_widget(lbl)
  prediction('g' ,'d')
def dapplecall():
  clear_widget(lbl)
  prediction('a' ,'d')
def damazoncall():
  clear_widget(lbl)
  prediction('am' ,'d')

	
root = Tk()
root.title('STOCK MARKET PREDICTOR')
root.geometry("1300x650")

 
def test(msg5):
    left = Label(root, text=msg5, fg='BLUE')
    left.config(font=('Courier',20))
    left.pack()

lbl = Label(root, text="STOCK MARKET PRICE PREDICTOR", font=("Arial Bold",50))
lbl.place(x=80,y=160)
#lbl.pack()
def clear_widget(widget):
    widget.destroy()
	
menubar = Menu(root)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="GOOGLE", command = lgooglecall)
filemenu.add_command(label="APPLE", command=lapplecall)
filemenu.add_command(label="AMAZON", command=lamazoncall)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)
menubar.add_cascade(label="Linear Regression", menu=filemenu)


editmenu = Menu(menubar, tearoff=0)
editmenu.add_separator()
editmenu.add_command(label="GOOGLE", command=sgooglecall)
editmenu.add_command(label="APPLE", command=sapplecall)
editmenu.add_command(label="AMAZON", command=samazoncall)
menubar.add_cascade(label="Support Vector Machine", menu=editmenu)


helpmenu = Menu(menubar, tearoff=0)
helpmenu.add_command(label="GOOGLE", command=dgooglecall)
helpmenu.add_command(label="APPLE", command=dapplecall)
helpmenu.add_command(label="AMAZON", command=damazoncall)
menubar.add_cascade(label="Decision Tree Regression", menu=editmenu)

root.config(menu=menubar)

label=Label()
root.mainloop()