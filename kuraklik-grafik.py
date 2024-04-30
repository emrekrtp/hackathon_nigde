import tkinter as tk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Baraj doluluğu verilerini yükle
file_path_dams = "C:/Users/emrek/Desktop/hackathon/genel_baraj_doluluk.csv"
data_dams = pd.read_csv(file_path_dams)
data_dams['Tarih'] = pd.to_datetime(data_dams['Tarih'], format='%d/%m/%Y')  # Tarih sütununu datetime formatına dönüştür

# NaN değerleri kontrol et ve temizle
data_dams.dropna(inplace=True)

# Yıllık ortalama doluluk oranlarını hesapla
average_yearly_data = data_dams.groupby(data_dams['Tarih'].dt.year)['doluluk_orani'].mean().reset_index()

# Özellikleri ve hedef değişkeni seç
X = average_yearly_data[['Tarih']].values  # Tarih
y = average_yearly_data['doluluk_orani'].values  # Yıllık ortalama Doluluk Oranı

# Modeli oluştur ve eğit
model = LinearRegression()
model.fit(X, y)

# Nüfus verilerini yükle
file_path_population = "C:/Users/emrek/Desktop/hackathon/50yil_nufus.csv"
data_population = pd.read_csv(file_path_population)

# "Nüfus" sütunundaki virgülleri kaldır ve float'a dönüştür
data_population["nufus"] = data_population["nufus"].replace(",", "", regex=True).astype(float)

# Tekrarlanan yılları kontrol et ve temizle
if not data_population['yil'].is_unique:
    data_population = data_population.drop_duplicates(subset=['yil'])

# Lineer regresyon modelini oluştur
model_population = LinearRegression()

# Verileri X ve y olarak ayır
X_population = data_population["yil"].values.reshape(-1, 1)  # Yıl
y_population = data_population["nufus"].values

# Modeli eğit
model_population.fit(X_population, y_population)

# Gelecek yılları oluştur (2024-2200 arası)
future_years = np.arange(2024, 2034).reshape(-1, 1)

# Baraj doluluğu tahminlerini yap
dam_predictions = model.predict(future_years)

# Nüfus tahminlerini yap
population_predictions = model_population.predict(future_years)

# Kuraklık tahmini yap (örneğin, baraj doluluğu eşik değeri 30 olarak varsayalım)
threshold = 30
drought_probabilities = [1 * max(1, min(100, dam - threshold) / threshold) for dam in dam_predictions]

# Grafik oluşturma
plt.figure(figsize=(7, 4))
plt.bar(future_years.flatten(), drought_probabilities, color='green', alpha=0.7)
plt.title("2024-2034 Yılları Arasında Kuraklık Tahmini")
plt.xlabel("Yıl")
plt.ylabel("Kuraklık Olasılığı (%)")
plt.grid(True)
plt.xticks(range(2024, 2034, 10))
plt.yticks(range(0, 2, 1), ['100%',"0"])
plt.show()
