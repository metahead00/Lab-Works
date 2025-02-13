import time
import pandas as pd
import numpy as np
import serial
from keras.models import load_model 

# Load trained model
model = load_model("model_half1_seed61_second.h5")

# Function to normalize data
def normalize_data(df):
    # Normalization between 0 and 1
    # Adjust based on training data normalization
    normalized_df = (df - df.min()) / (df.max() - df.min())
    return normalized_df

def read_data(ser, duration=4, max_lines=300):
    start_time = time.time()
    finger_data = []

    while time.time() - start_time < duration:
        raw = ser.readline()
        print("Raw line: ", raw)

        line = raw.decode('utf-8', errors='ignore').strip()
        print("Line: ", line)

        if line.startswith('!') and '#' in line:
            features = line[1:line.index('#')].split(';')
            print("Filtered: ", features)

            values = [int(features[i]) for i in range(5)]
            finger_data.append(values)

    # Truncate data to the last `max_lines` lines
    if len(finger_data) > max_lines:
        finger_data = finger_data[-max_lines:]
    
    # Ensure the data has the required shape (max_lines, 5)
    if len(finger_data) < max_lines:
        print("Not enough data collected.")
        return None

    return pd.DataFrame(finger_data)

def calibrate(ser):
    print("Calibration")
    accel_bias = []
    gyro_bias = []
    calibration_complete = False
    
    while not calibration_complete:
        raw = ser.readline()
        print("Raw line: ", raw)

        line = raw.decode('utf-8', errors='ignore').strip()
        print("Line: ", line)
        
        if "Accel biases X/Y/Z:" in line:
            accel_bias = line.split(",")
        elif "Gyro biases X/Y/Z:" in line:
            gyro_bias = line.split(",")
        elif "Calibration done!" in line:
            print("Calibration complete")
            calibration_complete = True
    
    return accel_bias, gyro_bias

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

def predict_gesture(ser, prediction_duration=5, max_lines=300):
    df = read_data(ser, prediction_duration, max_lines)
    if df is not None and not df.empty:
        print("Data Captured:")
        print(df)
        
        # Normalize the data
        df_normalized = normalize_data(df)
        print("Normalized Data:")
        print(df_normalized)
        
        # Reshape the data for the model
        data_array = df_normalized.values
        if data_array.shape[0] == 300 and data_array.shape[1] == 5:
            data_array = data_array.reshape((1, 300, 5))
            print("Data reshaped for model:", data_array.shape)

            # Predict using the loaded model
            y_pred = model.predict(data_array)

            # Debugging: print raw output probabilities
            print("Model output probabilities:", y_pred)

            # Get the predicted label (index of the highest probability)
            pred = np.argmax(y_pred)

            # Print the predicted label and corresponding letter
            print("Predicted label:", pred)
            print("Predicted letter:", label_to_letter[pred])

        else:
            print("Data shape mismatch. Expected (300, 5), got", data_array.shape)
    else:
        print("Failed to capture data.")  

def main():
    ser = serial.Serial('COM5', 115200)
    acc, gyro = calibrate(ser)
    while True:
        input("Press Enter to start gesture prediction...")
        predict_gesture(ser)

if __name__ == "__main__":
    main()
