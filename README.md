 Food Freshness Analyzer (Synthetic Data Project) 
 Project Overview

The Food Freshness Analyzer is a machine learning project that predicts whether stored food is Fresh, Stale, or Spoiled using synthetically generated environmental data. No Juypter Notebooks.

The project simulates real-world food storage conditions and demonstrates an end-to-end machine learning pipeline, including data generation, analysis, visualization, model training, and evaluation.


  Project Goal 

Predict food freshness based on environmental storage conditions:

Temperature

Humidity

Storage time

Gas concentration (proxy for spoilage/microbial activity)


 Why Synthetic Data? 

Real food freshness datasets are often limited, inconsistent, or proprietary.
To address this, this project uses synthetic data generated with controlled logic to realistically model food degradation over time.

Benefits of using synthetic data:

Reproducible experiments

Controlled feature distributions

Balanced freshness classes

Ethical and privacy-safe data usage

 Dataset Description (Synthetic) 

Each data point represents a stored food item under simulated conditions.

Feature	Description
Temperature	Storage temperature (°C)
Humidity	Storage humidity (%)
Time_Stored	Hours since storage
Gas	Gas concentration indicating spoilage
Label	Fresh, Stale, or Spoiled
Labeling Logic

Freshness labels are assigned using domain-inspired rules:

Fresh: Low storage time and low gas levels

Stale: Moderate storage time

Spoiled: Extended storage time

Labels are synthetic and used to validate the ML pipeline rather than real-world deployment accuracy.

 Tech Stack 

Language: Python

Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn

Model: Random Forest Classifier

Environment: Jupyter Notebook

 Project Workflow 

Synthetic data generation

Data exploration & summary statistics

Exploratory Data Analysis (EDA)

Feature scaling

Train/test split

Model training

Model evaluation

Feature importance visualization

 Machine Learning Model 

Algorithm: Random Forest Classifier

Evaluation Metrics:

Accuracy

Confusion Matrix

Classification Report

Feature importance is analyzed to understand which storage conditions most influence freshness predictions.

 Visualizations 

Temperature vs Gas concentration by freshness label

Storage time distribution across freshness levels

Feature importance bar chart

 Project Structure 

food-freshness-analyzer/
│
├── data/
│   └── food_freshness.csv
│
├── src/
│   └── food_freshness_analyzer.py
│
├── README.md
├── requirements.txt
└── .gitignore

 How to Run 

git clone https://github.com/katelynn3/food-freshness-analyzer.git
cd food-freshness-analyzer
pip install -r requirements.txt
python src/food_freshness_analyzer.py

 Future Improvements 

Incorporate real sensor-based datasets

Add deep learning models

Deploy as a web application (Streamlit or Flask)

Integrate IoT sensor simulations

 Educational Purpose 

This project was developed as part of a Computer Science & Machine Learning portfolio to demonstrate practical ML skills when real-world datasets are unavailable.

 Author 

Katelynn Comlan-Cataria
B.S. Computer Science — WGU
Aspiring Machine Learning Engineer / Data Scientist

 License 

This project is open-source and intended for educational use.
