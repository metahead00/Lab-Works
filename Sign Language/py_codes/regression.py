import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

merged = r"C:\Users\melik\Desktop\softsensorslab\TestSubject01\merged.csv"

df = pd.read_csv(merged, delimiter=';')
df = df[df['thumb_y'] <= 3] 
print(df.head())
df.describe()

# Define features and labels
features = df[['thumb', 'index', 'middle', 'ring', 'pinkie']]
labels = df[['thumb_y', 'index_y', 'middle_y', 'ring_y', 'pinkie_y']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize models for each finger
thumb_model = LinearRegression()
index_model = LinearRegression()
middle_model = LinearRegression()
ring_model = LinearRegression()
pinkie_model = LinearRegression()

# Train models on the training data
thumb_model.fit(X_train, y_train['thumb_y'])
index_model.fit(X_train, y_train['index_y'])
middle_model.fit(X_train, y_train['middle_y'])
ring_model.fit(X_train, y_train['ring_y'])
pinkie_model.fit(X_train, y_train['pinkie_y'])

# Predict states on the test set
thumb_predictions = thumb_model.predict(X_test)
index_predictions = index_model.predict(X_test)
middle_predictions = middle_model.predict(X_test)
ring_predictions = ring_model.predict(X_test)
pinkie_predictions = pinkie_model.predict(X_test)

# Convert predictions to integers (round to nearest integer)
thumb_predictions = thumb_predictions.round().astype(int)
index_predictions = index_predictions.round().astype(int)
middle_predictions = middle_predictions.round().astype(int)
ring_predictions = ring_predictions.round().astype(int)
pinkie_predictions = pinkie_predictions.round().astype(int)

# Create DataFrame for actual states from the test set
actual_states = y_test.copy()
actual_states.columns = ['thumb_actual', 'index_actual', 'middle_actual', 'ring_actual', 'pinkie_actual']

# Create DataFrame for predicted states
predicted_states = pd.DataFrame({
    'thumb_pred': thumb_predictions,
    'index_pred': index_predictions,
    'middle_pred': middle_predictions,
    'ring_pred': ring_predictions,
    'pinkie_pred': pinkie_predictions
})

# Combine actual and predicted states into one DataFrame
results = pd.concat([actual_states.reset_index(drop=True), predicted_states], axis=1)

# Define the dictionary that maps finger states to meanings
state_meanings = {
    (3, 1, 3, 3, 3): 'Number 1',
    (3, 1, 1, 3, 3): 'Number 2',
    (1, 1, 1, 3, 3): 'Number 3',
    (3, 1, 1, 1, 1): 'Number 4',
    (1, 1, 1, 1, 1): 'Number 5',
    (3, 1, 1, 1, 3): 'Number 6',
    (2, 1, 1, 1, 2): 'Number 6',
    (3, 1, 1, 3, 1): 'Number 7',
    (2, 1, 1, 2, 1): 'Number 7',
    (3, 1, 3, 1, 1): 'Number 8',
    (2, 1, 2, 1, 1): 'Number 8',
    (3, 3, 1, 1, 1): 'Number 9',
    (2, 2, 1, 1, 1): 'Number 9',
    (1, 3, 3, 3, 3): 'Number 10'
}

# Function to determine the meaning based on finger states
def determine_meaning(row):
    finger_states = (row['thumb_pred'], row['index_pred'], row['middle_pred'], row['ring_pred'], row['pinkie_pred'])
    return state_meanings.get(finger_states, 'Unknown')

def determine_actual_meaning(row):
    finger_states = (row['thumb_actual'], row['index_actual'], row['middle_actual'], row['ring_actual'], row['pinkie_actual'])
    return state_meanings.get(finger_states, 'Unknown')

# Apply the function to the DataFrame
results['meaning'] = results.apply(determine_meaning, axis=1)
# Assuming 'results' DataFrame already has actual states
results['actual_meaning'] = results.apply(determine_actual_meaning, axis=1)

# Save the combined DataFrame to a CSV file
results.to_csv('actual_vs_predicted_states_with_meaning.csv', sep=';', index=False)

from sklearn.metrics import accuracy_score, precision_score

# Calculate accuracy
accuracy = accuracy_score(results['actual_meaning'], results['meaning'])
print(f'Accuracy: {accuracy:.2f}')

# Calculate precision
# For precision, you need to specify the average method if you're dealing with multiple classes.
# Here, we'll calculate precision for each class separately.
precision = precision_score(results['actual_meaning'], results['meaning'], average=None, labels=list(state_meanings.values()))
print('Precision for each class:')
for meaning, precision_value in zip(state_meanings.values(), precision):
    print(f'{meaning}: {precision_value:.2f}')
