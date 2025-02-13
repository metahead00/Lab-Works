import time
import pandas as pd
import serial
import numpy as np
from keras.models import load_model

# Load trained model
model = load_model('model_test1.h5')

def read_data(ser, duration, max_lines):
    start_time = time.time()
    data = []

    while time.time() - start_time < duration:  # Wait for the specified duration
        try:
            raw_line = ser.readline()
            print("Raw byte sequence:", raw_line)  # Debug print

            line = raw_line.decode('utf-8', errors='ignore').strip()
            print("Decoded line:", line)  # Debug print

            # Filter important data
            if line.startswith('!') and '#' in line:
                features = line[1:line.index('#')].split(';')  # Extract data between '!' and '#'
                print("Filtered features:", features)  # Debug print
                
                # Convert first five features to integers and append to data
                if len(features) >= 5:
                    values = [int(features[i]) for i in range(5)]
                    data.append(values)
                    print("Parsed values:", values)  # Debug print
        except Exception as e:
            print(f"Error reading line: {e}")

    # Truncate data to the last `max_lines` lines
    if len(data) > max_lines:
        data = data[-max_lines:]
    
    # Ensure the data has the required shape (max_lines, 5)
    if len(data) < max_lines:
        print("Not enough data collected.")
        return None

    return pd.DataFrame(data)

def calibrate(ser):
    print("Calibration phase started.")
    calibration_complete = False

    while not calibration_complete:
        try:
            raw_line = ser.readline()
            print("Raw byte sequence:", raw_line)  # Debug print

            line = raw_line.decode('utf-8', errors='ignore').strip()
            print("Decoded line:", line)  # Debug print

            # Check for calibration completion lines
            if "Keep IMU level." in line:
                print("IMU level instruction received.")
            elif "Calibration done!" in line:
                print("Calibration completed.")
            elif "Accel biases X/Y/Z:" in line:
                print("Accel biases received.")
            elif "Gyro biases X/Y/Z:" in line:
                print("Gyro biases received.")
            elif line.startswith("-") or line.startswith("0") or line.startswith("1") or line.startswith("2") or line.startswith("3") or line.startswith("4") or line.startswith("5") or line.startswith("6") or line.startswith("7") or line.startswith("8") or line.startswith("9"):
                print("Calibration data: ", line)
                # Check if this is the last calibration line
                if "Gyro biases X/Y/Z:" in line:
                    calibration_complete = True
        except Exception as e:
            print(f"Error reading line: {e}")

    print("Calibration phase completed.")

def predict_gesture(ser, prediction_duration=4, max_lines=290):
    df = read_data(ser, prediction_duration, max_lines)
    if df is not None and not df.empty:
        print("Data Captured:")
        print(df)
        # Reshape the data for the model
        data_array = df.values
        if data_array.shape[0] == 290 and data_array.shape[1] == 5:
            data_array = data_array.reshape((1, 290, 5))
            print("Data reshaped for model:", data_array.shape)

            # Predict using the loaded model
            y_pred = model.predict(data_array)
            pred = np.argmax(y_pred)
            print("Predicted label:", pred)
        else:
            print("Data shape mismatch. Expected (290, 5), got", data_array.shape)
    else:
        print("Failed to capture data.")

def main():
    ser = serial.Serial('COM5', 115200)  # Adjust the port and baud rate as needed

    # Calibration phase
    calibrate(ser)

    # Prediction phase
    while True:
        input("Press Enter to start gesture prediction...")
        predict_gesture(ser)

if __name__ == "__main__":
    main()
