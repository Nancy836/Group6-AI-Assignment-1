# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import scipy.stats
import tensorflow as tf

#importing dataset
ds= pd.read_csv('africa-2018-population-and-internet-users.csv')

#displaying the top 5 rows in the dataset using pandas
ds.head()

#calculating the mean of the facebook\r subscribers
ds["Facebook\r subscribers 2017"].mean()

#calculating the mode of facebook subscribers in Africa in 2017
ds["Facebook\r subscribers 2017"].mode()

#displaying the column titles using pandas
ds.columns

# drawing a horizontal bar graph using matplotlib
x = ds["Facebook\r subscribers 2017"]/100000
y = ds["Country"]
figure(num=None, figsize=(18,18), dpi=256, facecolor='w', edgecolor='r')
plt.barh(y,x)
plt.ylabel("Countries in Africa")
plt.xlabel("Number of Facebook subscribers in 2017(in 100,000s)")
plt.title("Facebook subscribers in African Countries in 2017")
plt.show()

# dropping 'InternetGrowth' label from rows
x= ds.drop('InternetGrowth', axis=1)
y= ds['InternetGrowth']

# using numpy to form a matrix using data from the dataset
a = np.array(x)
b = np.array(y)

# importing sklearn
from sklearn.model_selection import train_test_split
#test to see if it works
x_train, x_test, y_train, y_test = train_test_split(a, b)

#displaying split dataset
x_train

#displaying dropped labels
y_train

tf.keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test)= mnsit.load_data()
# using matplotlib to graph tensor
plt.imshow(x_train[0])
plt.show





