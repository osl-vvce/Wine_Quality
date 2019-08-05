import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

winepath='winequality-red.csv'  #file path
winedata=pd.read_csv(winepath) #read data
y=winedata.quality              #setting the prediction feature
winefeatures=['fixed acidity','volatile acidity','citric acid','residual sugar','free sulfur dioxide']    #features to predict y
x=winedata[winefeatures]
winemodel=DecisionTreeRegressor(random_state=1)   #using decision tree for prediction
winemodel.fit(x,y)
train_x, val_x, train_y, val_y = train_test_split(x, y,test_size = 0.2, random_state = 0)
train_x_scaled = preprocessing.scale(train_x)
winemodel = DecisionTreeRegressor()
winemodel.fit(train_x, train_y)
valpred = winemodel.predict(val_x)
winepred = winemodel.predict(x)                   
x=np.array(val_y).tolist()

#printing first 5 predictions
print("\nThe prediction:\n")
for i in range(0,5):
    print (x[i])
    
#printing first five expectations
print("\nThe expectation:\n")
print (val_y.head())
print('The accuracy of the model is: ',+metrics.accuracy_score(val_y, valpred))
mean_absolute_error(y, winepred)    
print('The error of the model is: ',+mean_absolute_error(val_y, valpred))
