import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import random
import tensorflow as tf
from keras.regularizers import l2

# Learning rate scheduler function
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.99

lr_scheduler = LearningRateScheduler(scheduler)

seed = 61
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Load dataset
set_a = r"badjusted\adjusted_a.csv"
dataset = pd.read_csv(set_a, delimiter=';', header=None)

x = dataset.iloc[:, :5].values
y = dataset.iloc[:, 5].values

# Normalize input data
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

# Reshape data into sequences of length 300
t = 300
samples = x.shape[0] // t
x = np.reshape(x, (samples, t, x.shape[1]))

# Extract corresponding labels for each sequence
indices = [ind for ind in range(len(y)) if ind % t == 0]
y = y[indices]

# Assume the label for "A" is in your dataset as a string 'A'
label_for_a = 'A'

# Update labels: 1 for 'A', 0 for others
y_binary = np.where(y == label_for_a, 1, 0)

# Convert to categorical (you could skip this if it's a binary classification)
y_one_hot = to_categorical(y_binary, num_classes=2)

print("Unique labels after encoding:", np.unique(y_binary))

# Shuffle data and split into training and test sets
indices = np.arange(len(x))
np.random.shuffle(indices)

x_shuffled = x[indices]
y_shuffled = y_one_hot[indices]

X_train, X_test, y_train, y_test = train_test_split(x_shuffled, y_shuffled, test_size=0.2, shuffle=False)

# Define your model (as mentioned previously)
model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(x.shape[1], x.shape[2]), kernel_regularizer=l2(0.01)))
model.add(Dropout(0.2))
model.add(LSTM(units=64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=2, activation='softmax'))  # Use 'softmax' since we have two classes now

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with the learning rate scheduler
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), callbacks=[lr_scheduler])

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")
