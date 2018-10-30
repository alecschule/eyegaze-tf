import tensorflow as tf
import numpy as np
import os
import csv
import random

# seed uses system time by default
random.seed()

fileDir = 'annotated_files'
testDir = 'testing_files'

trainingData = []
trainingLabels = []
testData = []
testLabels = []

for filename in os.listdir(fileDir):
    with open(fileDir + "/" + filename, newline='') as f:
        # read csv file into array
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            rowArray = [float(item) for item in row]
            trainingData.append(rowArray[:-1])
            trainingLabels.append(rowArray[-1])

for filename in os.listdir(testDir):
    with open(testDir + "/" + filename, newline='') as f:
        # read csv file into array
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            rowArray = [float(item) for item in row]
            testData.append(rowArray[:-1])
            testLabels.append(rowArray[-1])

# randomize data
indexes = [i for i in range(len(trainingData))]
random.shuffle(indexes)
randomData = []
randomLabels = []

for i in indexes:
    randomData.append(trainingData[i])
    randomLabels.append(trainingLabels[i])

# fit all columns to range (-1, 1)
npData = np.array(randomData)
absData = np.absolute(npData)
absMax = absData.max(axis=0)
for i in range(len(absMax)):
    if absMax[i] == 0:
        absMax[i] = 1
normalData = npData / absMax

npTestData = np.array(testData)
absTestData = np.absolute(npTestData)
absTestMax = absTestData.max(axis=0)
for i in range(len(absTestMax)):
    if absTestMax[i] == 0:
        absTestMax[i] = 1
normalTestData = npTestData / absTestMax
# Split annotated frames randomly into test data and evaluation data
# trainingData = []
# trainingLabels = []
# testData = []
# testLabels = []
# for i in range(len(normalData)):
    # if (random.random() > 0.25):
        # trainingData.append(normalData[i])
        # trainingLabels.append(npLabels[i])
    # else:
        # testData.append(normalData[i])
        # testLabels.append(npLabels[i])

# Numpy arrays are necessary for tensorflow
npTrainingData = np.array(normalData)
npTrainingLabels = np.array(randomLabels)
npTestData = np.array(normalTestData)
npTestLabels = np.array(testLabels)

# Describes the neural network model
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
  # tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])

# Assign optimization function, loss function, and success metric
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fit model to training data
model.fit(npTrainingData, npTrainingLabels, epochs=5)
# Evaluate model on test frames

predictions = model.predict(npTestData)
lookingSuccess = 0
lookingFail = 0
notLookingSuccess = 0
notLookingFail = 0
for i in range(len(npTestLabels)):
# for prediction,label in predictions,npTestLabels:
    prediction = predictions[i]
    label = npTestLabels[i]
    if label == 0:
        # not looking
        if np.argmax(prediction) == 0:
            # success
            notLookingSuccess+= 1
        else:
            # fail
            notLookingFail += 1
    else:
        # looking
        if np.argmax(prediction) == 0:
            # fail
            lookingFail += 1
        else:
            # success
            lookingSuccess += 1

totalLooking = lookingSuccess+lookingFail
totalNotLooking = notLookingSuccess+notLookingFail
print ("Looking frame success: ", lookingSuccess, "/", totalLooking, "| ", lookingSuccess/totalLooking*100, "% accuracy")
print ("Not looking frame success: ", notLookingSuccess, "/", totalNotLooking, "| ", notLookingSuccess/totalNotLooking*100, "% accuracy")

# loss, acc = model.evaluate(npTestData, npTestLabels)
# print("Test loss: " , loss)
# print("Test accuracy: " , acc)
