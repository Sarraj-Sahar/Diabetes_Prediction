# Importation des bibliothèques
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st
# bib pour visualiser et manipuler les outils statistiques
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# Adding multiple databases to select from
# Create a title and a sub-title
st.write("""
# Diabetes Detection
## Detect if someone has diabetes using machine learning and python 
""")
st.subheader('')
st.subheader('')


def file_selector(folder_path='./datasets/'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox(" 1.Select a diabetes dataset", filenames)
    return os.path.join(folder_path, selected_filename)


# # load dataset
filename = file_selector()


# # Read Data
dataset = pd.read_csv(filename)


# ########## DATA PREPROCESSING
# ########## change 0s to median values

# Les Outlayers et les valeurs Manquantes
# On remplace les 0 par NaN

cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols:
    dataset[col].replace(0, np.NaN, inplace=True)


# On peut remplir les valeurs avec la médiane (depend si le patient est diabétique ou non)
for col in dataset.columns:
    dataset.loc[(dataset["Outcome"] == 0) & (dataset[col].isnull()),
                col] = dataset[dataset["Outcome"] == 0][col].median()
    dataset.loc[(dataset["Outcome"] == 1) & (dataset[col].isnull()),
                col] = dataset[dataset["Outcome"] == 1][col].median()


# WEB APP


# Show the data as a table
st.subheader('')
st.subheader('')
st.subheader('Data Information: ')
st.dataframe(dataset)

# Show statistics on the data
st.subheader('Data Description : mean , max,...')
st.write(dataset.describe())


# Show Columns
st.subheader('Show All columns in our database')
if st.button("Column Names"):
    st.write(dataset.columns)

# # heatmap to show correlation
#  set the size of figure to 12 by 10.
st.subheader(' Show between correlation our Variables using heatmap :')
fig = plt.figure(figsize=(12, 10))
p = sns.heatmap(dataset.corr(), annot=True, cmap='YlGnBu')
st.write(fig)


# Split the data into independent 'X' and dependent 'Y' variables
X = dataset.iloc[:, 0:8].values
Y = dataset.iloc[:, -1].values

# Split the data set into 75% Training and 25% Testing
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0)


# Get the feature input from the user
def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('glucose', 40, 199, 117)
    blood_pressure = st.sidebar.slider('blood_pressure', 20, 122, 72)
    skin_thickness = st.sidebar.slider('skin_thickness', 5, 99, 23)
    insulin = st.sidebar.slider('insulin', 10.0, 850.0, 110.0)
    BMI = st.sidebar.slider('BMI', 15.0, 67.1, 32.0)
    DPF = st.sidebar.slider('DPF', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('age', 15, 81, 29)

    # Store a dictionary into a variable
    user_data = {'pregnancies': pregnancies,
                 'glucose': glucose,
                 'blood_pressure': blood_pressure,
                 'skin_thickness': skin_thickness,
                 'insulin': insulin,
                 'BMI': BMI,
                 'DPF': DPF,
                 'age': age
                 }

    # Transform the data into a data frame
    features = pd. DataFrame(user_data, index=[0])
    return features


# Store the user input into a variable
user_input = get_user_input()

# Set a subheader and display the users input
st.subheader(' User Input:')
st.write(user_input)  # anytime user changes the slider , we can see the value


# Create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

# Show the models metrics
st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100) + '%')

# Store the models predictions in a variable
prediction = RandomForestClassifier.predict(user_input)

# Set a subheader and display the classification
st.subheader('Classification: ')
st.write(prediction)


# st.subheader('Logistic Regression :')

# features = dataset[['Pregnancies', 'Glucose', 'BloodPressure',
#                     'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
# Y = dataset['Outcome']

# train_features, test_features, train_labels, test_labels = train_test_split(
#     features, Y)

# scaler = StandardScaler()

# train_features = scaler.fit_transform(train_features)
# test_features = scaler.transform(test_features)

# model = LogisticRegression()
# model.fit(X_train, train_labels)
# LogisticRegression_prediction = (model.score(train_features, train_labels))
# st.write(LogisticRegression_prediction)
