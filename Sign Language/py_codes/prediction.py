import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

model = load_model("model_a.h5")

file = r"badjusted\adjusted_b.csv"

label_to_letter = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'K',
    10: 'L',
    11: 'M'
}

df = pd.read_csv(file, delimiter=";")
x = df.iloc[:300, :5]
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

x = x.reshape((1, 300, 5))

print(x)
y_pred = model.predict(x)
print("Prediction: ", y_pred)

pred = np.argmax(y_pred)
print(pred)
print("Letter: ", label_to_letter[pred])

