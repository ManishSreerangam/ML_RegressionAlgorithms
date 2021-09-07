#importing necessary packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler



from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor



#importing datasets
# slr --> simple linear regression
# mlr --> Multiple linear regression
# pr --> polynomial regression
data_slr = pd.read_csv("C://Users//manis//Desktop//Projects//Machine Learning A-Z (Codes and Datasets)//Regression//Salary_Data_slr.csv")
data_mlr = pd.read_csv("C://Users//manis//Desktop//Projects//Machine Learning A-Z (Codes and Datasets)//Regression//50_Startups_mlr.csv")
data_pr = pd.read_csv("C://Users//manis//Desktop//Projects//Machine Learning A-Z (Codes and Datasets)//Regression//Position_Salaries_pr.csv")
data_sv = pd.read_csv("C://Users//manis//Desktop//Projects//Machine Learning A-Z (Codes and Datasets)//Regression//Position_Salaries_pr.csv")
data_dt  = pd.read_csv("C://Users//manis//Desktop//Projects//Machine Learning A-Z (Codes and Datasets)//Regression//Position_Salaries_pr.csv")
data_rf  = pd.read_csv("C://Users//manis//Desktop//Projects//Machine Learning A-Z (Codes and Datasets)//Regression//Position_Salaries_pr.csv")

#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
"""
SIMPLE-LINEAR-REGRESSION
"""
print("--Starting Simple linear regression--")
data_slr.head(2) #checking first 2 instances
data_slr.count() #counting number of instances for each feature
data_slr.plot.line(x = "YearsExperience" , y = "Salary") #visualizing features using a line plot 
data_slr.plot.scatter(x = "YearsExperience" , y = "Salary" , c= "red") #visualizing features using a scatter plot 
"""
splitting dataset into dependent and independent variables
"""
X = data_slr.iloc[:, :-1].values #indpendent variable
y = data_slr.iloc[:, -1].values #dependent variable
"""
Making a test-train splits
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
"""
Imporitng a linear regressor from Scikit learn and fitting it to our training data
"""
regressor = LinearRegression()
regressor.fit(X_train, y_train)
"""
Predicting on our test dataset
"""
y_pred = regressor.predict(X_test)
"""
Visualizing our train and test sets
"""
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

"""
Multiple linear regression : we are supposed to implement a multiple linear 
regressor on a dataset related to 50 
"""
print("--Starting Multiple  linear regression--")

data_mlr.head(2) #checking first 2 instances
print(data_mlr.corr()) #cheking if there exists any correlation between features

data_mlr["State"].unique() #checking how many unique values we have for his categorical variable

X = data_mlr.iloc[:, :-1].values
y = data_mlr.iloc[:, -1].values
print(X)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)
#multiple linear regression class automatically avoids the dummy variable trap, hence not removing 
# columns in order to dodge this. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------


"""
Polynomial Regression. consists info of a position and its respective salary 
"""
print("--Starting polynomial  linear regression--")

data_pr.head(2)

X = data_pr.iloc[:, 1:-1].values
y = data_pr.iloc[:, -1].values

lin_reg = LinearRegression()
lin_reg.fit(X, y)

poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

lin_reg.predict([[6.5]])

lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))

#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

"""
Support-Vector-Regression
"""
print("--Starting suuport vector regression --")

X = data_sv.iloc[:, 1:-1].values
y = data_sv.iloc[:, -1].values



sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))


plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid))), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

"""
Decision-Tree-Regression
"""
print("--Starting decision tree regression--")

X = data_dt.iloc[:, 1:-1].values
y = data_dt.iloc[:, -1].values

regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

regressor.predict([[6.5]])


X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

"""
Random-forest-Regression
"""
print("--Starting random forest regression--")

X = data_rf.iloc[:, 1:-1].values
y = data_rf.iloc[:, -1].values

regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

regressor.predict([[6.5]])


X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()




