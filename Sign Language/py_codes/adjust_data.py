import pandas as pd

def meandata(files):
    for file_key, file_path in files.items():
        data = pd.read_csv(file_path, delimiter=';') 
        data.head()
        column = data['cycle']
        freq = {}
        for seq in column:
            if seq in freq:
                freq[seq] += 1
            else:
                freq[seq] = 1
        print(freq)
        values = freq.values()
        mean = sum(values) / len(values)
        print(mean)    
    
def preprocess(files):
    for file_key, file_path in files.items():
        df = pd.read_csv(file_path, delimiter=';')
        df = df[df['cycle'] < 30]
        df = df[df['cycle']  != 0]
        df.to_csv(file_path, sep=';', index=False)

def adjust_cycle_length(file_dict, target_length=290):
    for file_key, file_path in file_dict.items():
        data = pd.read_csv(file_path, delimiter=';')

        column = data['cycle']
        freq = {}
        for seq in column:
            if seq in freq:
                freq[seq] += 1
            else:
                freq[seq] = 1

        print(f"Frequency map for {file_key}:")
        print(freq)
        
        values = freq.values()
        if len(values) > 0:
            mean_value = sum(values) / len(values)
        else:
            mean_value = 0
        
        print(f"Mean frequency for {file_key}: {mean_value}")
        print('-' * 50)

        adjusted_data = []
        for seq, count in freq.items():
            cycle_data = data[data['cycle'] == seq]
            if count > target_length:
                # Truncate to the target length
                cycle_data = cycle_data.iloc[:target_length]
            elif count < target_length:
                # Pad to the target length
                padding_needed = target_length - count
                padding_values = cycle_data.iloc[-1:]  # Using the last row for padding
                padding = pd.concat([padding_values] * padding_needed, ignore_index=True)
                cycle_data = pd.concat([cycle_data, padding], ignore_index=True)
            
            adjusted_data.append(cycle_data)

        # Concatenate all adjusted cycles
        adjusted_data = pd.concat(adjusted_data, ignore_index=True)
        
        # Save the adjusted DataFrame to a new CSV file or overwrite the existing one
        output_file_path = file_path.replace('.csv', '_adjusted.csv')
        adjusted_data.to_csv(output_file_path, index=False, sep=';')
        print(f"Adjusted data saved to {output_file_path}")


file_dict = {
    'file1': r"C:\Users\melik\Desktop\softsensorslab\TestSubject01\2024_05_23___15_46_23_TestSubject01_1_Single_Thumb.csv",
    'file2' : r"C:\Users\melik\Desktop\softsensorslab\TestSubject01\2024_05_23___15_49_46_TestSubject01_2_Single_Index.csv",
    'file3' : r"C:\Users\melik\Desktop\softsensorslab\TestSubject01\2024_05_23___15_53_38_TestSubject01_3_Single_Middle.csv",
    'file4' : r"C:\Users\melik\Desktop\softsensorslab\TestSubject01\2024_05_23___15_57_14_TestSubject01_4_Single_Ring.csv",
    'file5' : r"C:\Users\melik\Desktop\softsensorslab\TestSubject01\2024_05_23___16_00_48_TestSubject01_5_Single_Pinkie.csv",
    'file6' : r"C:\Users\melik\Desktop\softsensorslab\TestSubject01\2024_05_23___16_04_02_TestSubject01_6_Grasp.csv",
    'file7' : r"C:\Users\melik\Desktop\softsensorslab\TestSubject01\2024_05_23___16_07_28_TestSubject01_7_FourFinger_Grasp.csv",
    'file8' : r"C:\Users\melik\Desktop\softsensorslab\TestSubject01\2024_05_23___16_11_00_TestSubject01_8_Thumb2Index.csv",
    'file9' : r"C:\Users\melik\Desktop\softsensorslab\TestSubject01\2024_05_23___16_14_29_TestSubject01_9_Thumb2Middle.csv",
    'file10' : r"C:\Users\melik\Desktop\softsensorslab\TestSubject01\2024_05_23___16_17_59_TestSubject01_10_Thumb2Ring.csv",
    'file11' : r"C:\Users\melik\Desktop\softsensorslab\TestSubject01\2024_05_23___16_22_08_TestSubject01_11_Thumb2Pinkie.csv"
}

preprocess(file_dict)
meandata(file_dict)
adjust_cycle_length(file_dict)

meandata(file_dict)