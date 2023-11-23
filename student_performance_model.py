
import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import  pyplot as plt
import seaborn as sns

df = pd.read_csv('student-por.csv', sep=';')

df.head()

missing_values = df.isnull().sum()

print("Missing Values Count per Column:")
print(missing_values)

df.describe()

df.columns

columns = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
       'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
       'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
       'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
       'Walc', 'health', 'absences']

# Loop through the list of columns and display the value counts for each one
for column in columns:
    counts = df[column].value_counts()
    print(column + ':')
    print(counts)
    print()

pivot_table = df.pivot_table(index='school', columns='G3', aggfunc='size', fill_value=0)

pivot_table_percentage = pivot_table.div(pivot_table.sum(axis=1), axis=0) * 100

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table_percentage, annot=True, fmt='.1f', cmap='YlGnBu', cbar_kws={'label': 'Percentage of Students'})
plt.title('Heatmap of School vs Final Grade (Percentage)')
plt.show()

pivot_table = df.pivot_table(index='sex', columns='G3', aggfunc='size', fill_value=0)


pivot_table_percentage = pivot_table.div(pivot_table.sum(axis=1), axis=0) * 100

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table_percentage, annot=True, fmt='.1f', cmap='YlGnBu', cbar_kws={'label': 'Percentage of Students'})
plt.title('Heatmap of Gender vs Final Grade (Percentage)')
plt.show()

pivot_table = df.pivot_table(index='reason', columns='G3', aggfunc='size', fill_value=0)

pivot_table_percentage = pivot_table.div(pivot_table.sum(axis=1), axis=0) * 100

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table_percentage, annot=True, fmt='.1f', cmap='YlGnBu', cbar_kws={'label': 'Percentage of Students'})
plt.title('Heatmap of Reason why they go to the school vs Final Grade')
plt.show()

pivot_table = df.pivot_table(index='higher', columns='G3', aggfunc='size', fill_value=0)

pivot_table_percentage = pivot_table.div(pivot_table.sum(axis=1), axis=0) * 100

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table_percentage, annot=True, fmt='.1f', cmap='YlGnBu', cbar_kws={'label': 'Percentage of Students'})
plt.title('Heatmap of Pursuit if further education vs Final Grade')
plt.show()

pivot_table = df.pivot_table(index='studytime', columns='G3', aggfunc='size', fill_value=0)

pivot_table_percentage = pivot_table.div(pivot_table.sum(axis=1), axis=0) * 100

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table_percentage, annot=True, fmt='.1f', cmap='YlGnBu', cbar_kws={'label': 'Percentage of Students'})
plt.title('Heatmap of weekly study time vs Final Grade')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# List of columns to label encode
columns_to_encode = ['school', 'sex', 'address', 'famsize', 'Pstatus',
                     'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup',
                     'famsup', 'paid', 'activities', 'nursery', 'higher',
                     'internet', 'romantic']

# Initialize a LabelEncoder
label_encoder = LabelEncoder()

for column in columns_to_encode:
    df[column] = label_encoder.fit_transform(df[column])
print(df)

# Split the data into features and target
X = df.drop('G3', axis=1)
y = df['G3']

# Standardize numerical features
scaler = StandardScaler()
X[X.select_dtypes(['float64', 'int64']).columns] = scaler.fit_transform(X.select_dtypes(['float64', 'int64']))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

X_train.shape

# The Model

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1)
])

model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.007)

model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Plot the training history
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.show()

# Test set
predictions = model.predict(X_test)
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions.flatten()})

print(predictions_df)

# Calculate the absolute differences between actual and predicted values
absolute_differences = np.abs(predictions_df['Actual'] - predictions_df['Predicted'])

average_difference = absolute_differences.mean()

print(f'Average Difference between Actual and Predicted: {average_difference}')