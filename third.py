
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("carprice.csv")
print(data.head())
print(data.isnull().sum())
print(data.describe())
print(data.CarName.unique())
sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))
sns.distplot(data.price)
plt.show()
plt.figure(figsize=(20, 15))
correlations = data.corr()
sns.heatmap(correlations, cmap="coolwarm", annot=True)
plt.show()
predict = "price"
data = data[["symboling", "wheelbase", "carlength", 
             "carwidth", "carheight", "curbweight", 
             "enginesize", "boreratio", "stroke", 
             "compressionratio", "horsepower", "peakrpm", 
             "citympg", "highwaympg", "price"]]
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)


model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)


print(model.score(xtest, predictions))