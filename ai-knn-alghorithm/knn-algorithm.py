import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

data = pd.read_csv('data-base/smartphone_dataset_pt_br.csv')

#Remove "modelo" and "Resolução" columns to future tests with new smartphones
data = data.drop(columns=['Modelo', 'Resolução'])

#Impute missing values with the mean for numeric columns
numeric_data = data.select_dtypes(include=['float64', 'int64'])
for column in numeric_data.columns:
    mean = int(data[column].mean(skipna=True))
    data[column] = data[column].replace(np.nan, mean)

#Impute missing values with the mode for string columns
string_data = data.select_dtypes(include=['object'])
for column in string_data:
    mode = data[column].mode()[0]
    data[column] = data[column].replace(np.nan, mode)

#Encode string data to analysis
label_encoders = {}
for column in string_data:
    label_encoder = LabelEncoder()
    data[column] = label_encoder.fit_transform(data[column])
    label_encoders[column] = label_encoder # Save label encoder for later use

#Transform "Avaliação" values into categories
ranges = [0, 70, 85, 100]  # Define ranges, Low (0-70), Medium (70-85), High (85-100)
labels = ['Low', 'Medium', 'High']
data['Avaliação'] = pd.cut(data['Avaliação'], bins=ranges, labels=labels)

#Drop "Avaliação" column from data and save in X, save dropped "Avaliação" column in y
X = data.drop(columns=['Avaliação'])
y = data['Avaliação']

#Normalizing scales of data
scaler = StandardScaler()
X = scaler.fit_transform(X)

#Separate 20% of data to test\
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Train KNN model with the value of K = 3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print('Model accuracy:', (accuracy_score(y_test, y_pred) * 100).__round__(1), '%')

def preprocess_new_data(new_data):
    #Encode data with saved encoders
    for column, encoder in label_encoders.items():
        new_data[column] = encoder.transform(new_data[column])

    new_data = scaler.transform(new_data)
    return new_data

new_data = pd.DataFrame({
    'Marca': ['samsung'],
    'Preço': [4229.00],
    '5G': [0],
    'NFC': [1],
    'IR_Blaster': [0],
    'Marca_Processador': ['exynos'],
    'Qtd_Cores': [8],
    'Veloc_Processador': [2.73],
    'Capac_Bateria': [4500],
    'Carreg_Rápido_Disp': [1],
    'Carreg_Rápido': [25],
    'Capacidade_Ram': [8],
    'Memória_Interna': [128],
    'Tamanho_Tela': [6.7],
    'Taxa_Atualização': [120],
    'Qtd_Câm_Tras': [4],
    'Qtd_Câm_Front': [1],
    'Sistema_Operacional': ['android'],
    'Câm_Tras_Principal': [64],
    'Câm_Front_Principal': [10],
    'Memória_Esten_Disp': [1],
    'Expansível_Até': [1024]
})

new_data_processed = preprocess_new_data(new_data)
new_prediction = knn.predict(new_data_processed)

print('Predicted "Avaliação" for new data:', new_prediction[0])
