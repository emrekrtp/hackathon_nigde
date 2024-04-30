import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Verileri yükle
file_path = "C:/Users/emrek/Desktop/hackathon/water-consumption.csv"
data = pd.read_csv(file_path)

# Şehirlere göre verileri grupla
grouped_data = data.groupby('District').sum().reset_index()

# Sadece kullanım miktarları sütunlarını seç
years = ['2015 (Consumption-m3)', '2016 (Consumption-m3)', '2017 (Consumption-m3)', '2018 (Consumption-m3)', '2019 (Consumption-m3)']

# Tkinter penceresi oluştur
root = tk.Tk()
root.title("Şehir Seçim ve Tahmini Kullanım Miktarı")

# Şehirleri içeren bir liste
cities = grouped_data['District'].tolist()

# Combobox oluştur
selected_city = tk.StringVar()
city_combobox = ttk.Combobox(root, textvariable=selected_city, values=cities)
city_combobox.pack(pady=20)

# Tahminleri gösteren fonksiyon
def show_predictions():
    city_name = selected_city.get()
    
    # Seçilen şehrin verilerini al
    city_data = grouped_data[grouped_data['District'] == city_name][years].iloc[0]
    
    # Geçmiş yılların kullanım verilerini kullanarak modeli eğit
    X = [[year] for year in range(2015, 2020)]  # X: Yıllar
    y = city_data.values.reshape(-1, 1)          # y: Kullanım miktarları
    
    # Doğrusal regresyon modeli oluştur ve eğit
    model = LinearRegression()
    model.fit(X, y)
    
    # Gelecek 10 yılın tahmini kullanım miktarını hesapla
    future_years = [[year] for year in range(2020, 2030)]
    future_predictions = model.predict(future_years)
    
    # Grafik oluşturma
    plt.figure(figsize=(10, 6))
    plt.bar(range(2020, 2030), future_predictions.flatten(), color='skyblue')
    plt.title(f"Gelecek 10 Yılın Tahmini Kullanım Miktarı - {city_name} Şehri")
    plt.xlabel("Yıl")
    plt.ylabel("Tahmini Kullanım Miktarı (milyon m3)")  # Kullanım miktarını milyon metreküp cinsinden göster
    plt.grid(axis='y')
    
    # Yatay ekseni (x ekseni) düzenle
    plt.xticks(range(2020, 2030), [str(year) for year in range(2020, 2030)])  # Yılları sayı olarak göster
    
    # Sütunların üzerine sayıları yazdır
    for i, value in enumerate(future_predictions.flatten()):
        plt.text(2020 + i, value + 0.2, f"{value:.1f}", ha='center')
    
    plt.show()

# Tahminleri göster butonu
show_button = tk.Button(root, text="Tahminleri Göster", command=show_predictions)
show_button.pack(pady=10)

# Pencereyi çalıştır
root.mainloop()
