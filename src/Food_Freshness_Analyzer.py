'''' Goal: Predict whether stored food is Fresh, Stale, or 
    Spoiled using environmental data 
    (temperature, humidity, gas levels, storage time).'''

#Simulate or collect a dataset representing food storage conditions over time. 
# Each record should have environmental readings (temperature, humidity, time, gas concentration) and a freshness label.

import pandas as pd
import random

# Step 1 - Generate/creating the dataset
data = []

# We use a for loop to repeat this code 200 times
# We use the random.uniform function to generate a random floating point number

for i in range(200):
    temp = random.uniform(2, 35)
    humidity = random.uniform(30, 90)
    time_stored = random.uniform(1, 72)
    gas = random.uniform(100, 500)

# Labeling the logic
    if time_stored < 24 and gas < 200:
        label = "Fresh"
    elif time_stored < 48:
        label = "Stale"
    else:
        label = "Spoiled"

data.append([temp, humidity, time_stored, gas, label])

df = pd.DataFrame(data, columns = ["Temperature", "Humidity", "Time", "Gas", "Label"])
df.to_csv("food_freshness.csv", index=False)

print(df.head())

# Step 2 - Data Analysis
import matplotlib.pyplot as plt
import seaborn as sns

print(df.describe()) # Summary stats

sns.scatterplot(data=df, x="Temperature", y="Gas", hue="Label")
plt.title("Storage Time Distribution by Freshness Level")
plt.show()

sns.boxplot(data=df, x="Label", y="Time")
plt.title("Storage Time Distribution by Freshness Level")
plt.show()

# Step 3 Data Preparation 

from sklearn.preprocessing import LabelEncoder

# Features
X = df[["Temperature", "Humidity", "Time", "Gas"]]

# This converts fresh/stale/spoiled to numbers
y = LabelEncoder().fit_transform(df["Label"])

# If the dataset is large enough, split; otherwise use all data

if len(df) > 1:
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
        )
else:
    print("Dataset is too small to split. Using all data for training and testing.")
    X_train, X_test, y_train, y_test = X, X, y, y
# Scaling and model training because out features are on different scales

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Scale the features

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions

y_pred = model.predict(X_test_scaled)

# Evaluate

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report: \n", classification_report(y_test, y_pred))

# Visulization

importances = model.feature_importances_
features = X.columns

sns.barplot(x=importances, y=features)
plt.title("Feature Importance for Food Freshness")
plt.show()