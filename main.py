# Importing few modules
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

# Loading the data. Here I am using the car.data dataset, provided in UCI Machine Learning Repository
# You can download the file from my github repo for now that stands beside this file

data = pd.read_csv("car.data")

# The head function prints out the first few lines of the dataset
print(data.head())


"""We have to convert strings to Integers or decimals, as ML and DL works on Numbers only. In the dataset,
we have attribute values like acc, vgood, etc. LabelEncoder() converts the strings to int"""

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))


"""Rest of the code is simple"""

predict = "class"

X = list(zip(buying, maint, door, persons, lug_boot, safety))
Y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.2)

model = KNeighborsClassifier(n_neighbors=7)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)

print(acc)
predicted = model.predict(x_test)

print("*"*100)

name = ["unacc", "acc", "good", "vgood"]
for x in range(len(predicted)):
  print("Predicted: ", name[predicted[x]], "Data: ", x_test[x], "Actual: ", name[y_test[x]])

######################################################################################################
