import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib.widgets import SpanSelector
from scipy.signal import savgol_filter

def plot_data(name, path):
    df = pd.read_csv(path, delimiter=',')
    #df.dropna()
    #fingers = df.iloc[:, :5]
    #for col in fingers.columns:
    #   fingers[col] = pd.to_numeric(fingers[col], errors='coerce')
    #accel = df.iloc[:, 5:8]
    #gyro = df.iloc[:, 8:]
    #print(fingers.head())
    #print(accel.head())
    #print(gyro.head())
    colors = ['b', 'orange', 'g', 'r', 'purple']  
    plt.figure(figsize=(12, 8))
    plt.plot(df)
    plt.title(name)
    plt.xlabel('Timestamp')
    plt.ylabel('Value')
    plt.show()

def plot_pyplot(name, path):
    df = pd.read_csv(path, delimiter=',', usecols=[0,1,2,3,4])
    df.dropna()
    fingers = df.iloc[:, :5]
    for col in fingers.columns:
        fingers[col] = pd.to_numeric(fingers[col], errors='coerce')
    #accel = df.iloc[:, 5:8]
    #gyro = df.iloc[:, 8:]
    print(fingers.head())
    #print(accel.head())
    #print(gyro.head())
    fig = px.line(fingers, title=name)
    fig.show()

def plot_multiple(files):
    for name, path in files.items():
        plot_data(name, path)    

def range_multiple(files):
    for name, path in files.items():
        range_selector(name, path)

def get_clean_rows(name, path):
    df = pd.read_csv(path, delimiter=',', usecols=[0,1,2,3,4])
    df.dropna()
    fingers = df.iloc[:, :5]
    for col in fingers.columns:
        fingers[col] = pd.to_numeric(fingers[col], errors='coerce')
    #accel = df.iloc[:, 5:8]
    #gyro = df.iloc[:, 8:]
    print(fingers.head())
    #print(accel.head())
    #print(gyro.head())
    colors = ['b', 'orange', 'g', 'r', 'purple']  
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(fingers.columns):
        plt.plot(fingers[col], color=colors[i], label=col) 
    selected_range = []

def range_selector(name, path):
    df = pd.read_csv(path, delimiter=',', usecols=[0,1,2,3,4])
    df.dropna()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    selected_ranges = []

    def on_select(xmin, xmax):
        selected_range = df.iloc[int(xmin):int(xmax)]
        selected_range['label'] = len(selected_ranges)
        selected_ranges.append(selected_range)

    selected_ranges=[]
    fig, ax = plt.subplots(figsize=(10, 5)) 
    ax.plot(df.index, df.iloc[:, :5])
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Value')
    ax.set_title(name)
    ax.legend()

    span = SpanSelector(ax, on_select, 'horizontal', useblit=True)
    plt.show()

    selected_data = pd.concat(selected_ranges)
    selected_data.to_csv(f'selected_{name}.csv', index=False)
    print("Saved")

def adjust_sequence(df, target_length, name):
    fixed = []
    grouped = df.groupby('label')
    for label, group in grouped:
        seq = group.iloc[:, :-1].values  # Extract the feature columns only
        seq_len = len(seq)
        
        if seq_len < target_length:
            # Padding: Calculate how much padding is needed
            padding_needed = target_length - seq_len
            
            # Pad with preceding data (you can also pad with succeeding data if needed)
            padded_sequence = np.pad(seq, ((padding_needed//2, padding_needed - padding_needed//2), (0, 0)), 
                                     mode='edge')
            padded_df = pd.DataFrame(padded_sequence, columns=group.columns[:-1])
            padded_df['label'] = label  # Re-assign the label
            fixed.append(padded_df)
        
        elif seq_len > target_length:
            # Truncate: Decide where to truncate
            start_index = (seq_len - target_length) // 2
            truncated_sequence = seq[start_index:start_index + target_length]
            truncated_df = pd.DataFrame(truncated_sequence, columns=group.columns[:-1])
            truncated_df['label'] = label  # Re-assign the label
            fixed.append(truncated_df)
        
        else:
            # Sequence is already the correct length
            fixed.append(group)
    
    # Concatenate all adjusted sequences back into a single DataFrame
    adjusted_df = pd.concat(fixed)
    adjusted_df = pd.DataFrame(adjusted_df)
    adjusted_df.to_csv(f'adjusted_{name}.csv', index=False)

def adjust_multiple(files):
    for name, path in files.items():
        df = pd.read_csv(path, delimiter=',')
        print(df.head())
        adjust_sequence(df, 300, name)

file_dict = {
    'a': r"letters\gesture_a.csv",
    'b': r"letters\gesture_b.csv",
    'c': r"letters\gesture_c.csv",
    'd': r"letters\gesture_d.csv",
    'e': r"letters\gesture_e.csv",
    'f': r"letters\gesture_f.csv",
    'g': r"letters\gesture_g.csv",
    'h': r"letters\gesture_h.csv",
    'i': r"letters\gesture_i.csv",
    #'j': r"letters\gesture_j.csv",
    'k': r"letters\gesture_k.csv",
    'l': r"letters\gesture_l.csv",
    'm': r"letters\gesture_m.csv",
    'n': r"letters\gesture_n.csv",
    'o': r"letters\gesture_o.csv",
    'p': r"letters\gesture_p.csv",
    'q': r"letters\gesture_q.csv",
    'r': r"letters\gesture_r.csv",
    's': r"letters\gesture_s.csv",
    't': r"letters\gesture_t.csv",
    'u': r"letters\gesture_u.csv",
    'v': r"letters\gesture_v.csv",
    'w': r"letters\gesture_w.csv",
    'x': r"letters\gesture_x.csv",
    'y': r"letters\gesture_y.csv",
    #'z': r"letters\gesture_z.csv"    
}

selected_dict = {
    'a': r"selected\selected_a.csv",
    'b': r"selected\selected_b.csv",
    'c': r"selected\selected_c.csv",
    'd': r"selected\selected_d.csv",
    'e': r"selected\selected_e.csv",
    'f': r"selected\selected_f.csv",
    'g': r"selected\selected_g.csv",
    'h': r"selected\selected_h.csv",
    'i': r"selected\selected_i.csv",
    #'j'
    'k': r"selected\selected_k.csv",
    'l': r"selected\selected_l.csv",
    'm': r"selected\selected_m.csv",
    'n': r"selected\selected_n.csv",
    'o': r"selected\selected_o.csv",
    'p': r"selected\selected_p.csv",
    'q': r"selected\selected_q.csv",
    'r': r"selected\selected_r.csv",
    's': r"selected\selected_s.csv",
    't': r"selected\selected_t.csv",
    'u': r"selected\selected_u.csv",
    'v': r"selected\selected_v.csv",
    'w': r"selected\selected_w.csv",
    'x': r"selected\selected_x.csv",
    'y': r"selected\selected_y.csv"
    #'z'
}

def meandata(files):
    for _, file_path in files.items():      
        data = pd.read_csv(file_path, delimiter=',') 
        data.head()
        column = data['label']
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

adjusted_dict = {
    'a': r"adjusted\adjusted_a.csv",
    'b': r"adjusted\adjusted_b.csv",
    'c': r"adjusted\adjusted_c.csv",
    'd': r"adjusted\adjusted_d.csv",
    'e': r"adjusted\adjusted_e.csv",
    'f': r"adjusted\adjusted_f.csv",
    'g': r"adjusted\adjusted_g.csv",
    'h': r"adjusted\adjusted_h.csv",
    'i': r"adjusted\adjusted_i.csv",
    #'j'
    'k': r"adjusted\adjusted_k.csv",
    'l': r"adjusted\adjusted_l.csv",
    'm': r"adjusted\adjusted_m.csv",
    'n': r"adjusted\adjusted_n.csv",
    'o': r"adjusted\adjusted_o.csv",
    'p': r"adjusted\adjusted_p.csv",
    'q': r"adjusted\adjusted_q.csv",
    'r': r"adjusted\adjusted_r.csv",
    's': r"adjusted\adjusted_s.csv",
    't': r"adjusted\adjusted_t.csv",
    'u': r"adjusted\adjusted_u.csv",
    'v': r"adjusted\adjusted_v.csv",
    'w': r"adjusted\adjusted_w.csv",
    'x': r"adjusted\adjusted_x.csv",
    'y': r"adjusted\adjusted_y.csv"
    #'z'
}

def change_sep(files):
    for name, path in files.items():
        df = pd.read_csv(path, delimiter=',')
        df.to_csv(f'b{path}', sep=';')


plot_multiple(selected_dict)
#plot_data("a", r"letters\gesture_a.csv")
#plot_pyplot("b", r"letters\gesture_b.csv")
#get_clean_rows("a", r"letters\gesture_a.csv")
#range_multiple(file_dict)
#meandata(selected_dict)
#df = pd.read_csv(r"selected\selected_a.csv", delimiter=',')
#adjust_sequence(df, 300, "a")
#adjust_multiple(selected_dict)
#plot_multiple(adjusted_dict)
#change_sep(adjusted_dict)






