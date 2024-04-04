# Import Package
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Variable value
x = np.array([1,2,3,4,5,6,7,8]).reshape((-1,1))
y = np.array([1,3,5,7,9,11,13])
plt.plot(x,y,"*")
plt.show()

# LinearRegression
model = LinearRegression()
model.fit(x,y)

# predict
res = model.predict([[9]])
print(res)

# PLOT
plt.plot(x,y,"*")
plt.plot(x,model.predict(x))
plt.show()

# DecisionTreeRegressor
model = DecisionTreeRegressor(max_depth=6)
model.fit(x,y)
res = model.predict([[10]])
print(res)
plt.plot(x,y,"*")
plt.plot(x,model.predict(x))
plt.show()