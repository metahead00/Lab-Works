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

# Load dataset
group1 = r"C:\Users\melik\Desktop\softsensorslab\half1.csv"
dataset = pd.read_csv(group1, delimiter=';', header=None)

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

# Encode labels and one-hot encode
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_one_hot = to_categorical(y_encoded)

# Ensure sequences are not shuffled within blocks
X_train, X_test, y_train, y_test = train_test_split(x, y_one_hot, test_size=0.2, shuffle=False)

# Print the first 5 samples from each dataset
print("X_train sample:")
print(X_train[:5])
print("\nY_train sample:")
print(y_train[:5])

print("\nX_test sample:")
print(X_test[:5])
print("\nY_test sample:")
print(y_test[:5])

# Learning rate scheduler function
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.99

lr_scheduler = LearningRateScheduler(scheduler)

# Define the LSTM model
n_neurons = 100
model = Sequential()
model.add(LSTM(n_neurons, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(LSTM(n_neurons,3))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(LSTM(n_neurons))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(units=12, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[lr_scheduler])

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Evaluate the model on the test set
scores = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {scores[1] * 100:.2f}%")

# Predict the labels for the test set and plot confusion matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Compute and plot confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
