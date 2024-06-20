"https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud adresinden dataseti indirebilirsiniz."

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Veri setini yükleme
df = pd.read_csv('C:/Users/kaana/creditcard.csv')

# Veriyi inceleme
print(df.head())
print(df.info())

# Özellikleri ve hedef değişkeni ayırma
X = df.drop('Class', axis=1)
y = df['Class']

# Veriyi eğitim ve test setlerine bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi standartlaştırma
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# k-NN modelini oluşturma ve eğitme
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Test veri seti üzerinde tahmin yapma
y_pred = knn.predict(X_test)

# Modelin performansını değerlendirme
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy * 100:.2f}%")
