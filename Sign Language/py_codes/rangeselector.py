import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector

# Load CSV file
df = pd.read_csv(r"letters\gesture_a.csv", usecols=[0, 1, 2, 3, 4], on_bad_lines='skip')
df = df.iloc[:, :5]
# Convert columns to numeric (forcing errors to NaN)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle NaN values (drop rows with NaN values)
df = df.dropna()

# Initialize selected data container
selected_ranges = []

# Define a callback function to capture the range selection
def onselect(xmin, xmax):
    # Convert selected range to integer indices
    selected_range = df.iloc[int(xmin):int(xmax)]
    
    # Label the selected range
    selected_range['Label'] = f'Sequence {len(selected_ranges) + 1}'  # Add your label
    
    # Append the selected range to the list
    selected_ranges.append(selected_range)
    
    print(f"Selected range from index {int(xmin)} to {int(xmax)}")

# Plotting the first column as an example
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df.index, df.iloc[:, :5])
ax.set_xlabel('Index')
ax.set_ylabel('Value')
ax.legend()
ax.set_title("Select a range by clicking and dragging")

# Create the SpanSelector
span = SpanSelector(ax, onselect, 'horizontal', useblit=True)

plt.show()

# After closing the plot, concatenate all selected ranges into a single DataFrame
if selected_ranges:
    selected_data = pd.concat(selected_ranges)
    selected_data.to_csv('selected_test.csv', index=False)
    print("Selected ranges saved to 'selected_ranges.csv'")
else:
    print("No range selected.")
