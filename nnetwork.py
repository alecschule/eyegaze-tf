import tensorflow as tf
import numpy as np
import os
import csv
import random

# seed uses system time by default
random.seed()

fileDir = 'annotated_files'

data = []
labels = []

for filename in os.listdir(fileDir):
    with open(fileDir + "/" + filename, newline='') as f:
        # read csv file into array
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            rowArray = [float(item) for item in row]
            data.append(rowArray[:-1])
            labels.append(rowArray[-1])

# fit all columns to range (-1, 1)
npData = np.array(data)
absData = np.absolute(npData)
normalData = npData / absData.max(axis=0)

npLabels = np.array(labels)

# Split annotated frames randomly into test data and evaluation data
trainingData = []
trainingLabels = []
testData = []
testLabels = []
for i in range(len(normalData)):
    if (random.random() > 0.25):
        trainingData.append(normalData[i])
        trainingLabels.append(npLabels[i])
    else:
        testData.append(normalData[i])
        testLabels.append(npLabels[i])

# Numpy arrays are necessary for tensorflow
npTrainingData = np.array(trainingData)
npTrainingLabels = np.array(trainingLabels)
npTestData = np.array(testData)
npTestLabels = np.array(testLabels)

# Describes the neural network model
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Assign optimization function, loss function, and success metric
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fit model to training data
model.fit(npTrainingData, npTrainingLabels, epochs=5)
# Evaluate model on test frames
loss, acc = model.evaluate(npTestData, npTestLabels)
print("Test loss: " , loss)
print("Test accuracy: " , acc)
