import time
import pandas as pd
import numpy as np
import serial

def read_data(ser, max_lines=500):
    counter = 0
    data = []

    while counter < max_lines:
        counter += 1
        raw = ser.readline()
        print("Raw line: ", raw)

        line = raw.decode('utf-8', errors='ignore').strip()
        print("Line: ", line)

        if line.startswith('!') and '#' in line:
            features = line[1:line.index('#')].split(';')
            print("Filtered: ", features)
            data.append(features)

    df = pd.DataFrame(data)

    return df

def save_data_to_csv(df, filename="real_victory.csv"):
    try:
        # Append to existing CSV file or create a new one if it doesn't exist
        with open(filename, 'a') as f:
            df.to_csv(f, header=f.tell() == 0, index=False)
        print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error saving data: {e}")

def main():
    ser = serial.Serial('COM5', 115200)
    num_repeats = int(input("Enter the number of repetitions: "))

    for i in range(num_repeats):
        input(f"Press Enter to start data collection for gesture {i + 1}...")
        df = read_data(ser)
        if df is not None and not df.empty:
            print("Data Captured:")
            print(df)
            save_data_to_csv(df)
        else:
            print("Failed to capture data for gesture", i + 1)

if __name__ == "__main__":
    main()
