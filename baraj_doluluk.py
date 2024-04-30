import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import numpy as np

# Verileri yükle
file_path_dams = "C:/Users/emrek/Desktop/hackathon/genel_baraj_doluluk.csv"
data_dams = pd.read_csv(file_path_dams)
data_dams['Tarih'] = pd.to_datetime(data_dams['Tarih'], format='%d/%m/%Y')  # Tarih sütununu datetime formatına dönüştür

# NaN değerleri kontrol et ve temizle
data_dams.dropna(inplace=True)

# Yıllık ortalama doluluk oranlarını hesapla
data_dams['Year'] = data_dams['Tarih'].dt.year
average_yearly_data = data_dams.groupby('Year')['doluluk_orani'].mean().reset_index()

# Özellikleri ve hedef değişkeni seç
X = average_yearly_data[['Year']].values  # Yıl
y = average_yearly_data['doluluk_orani'].values  # Yıllık ortalama Doluluk Oranı

# Modeli oluştur ve eğit
model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(X, y)

# Tahmin fonksiyonu
def predict_dam_occupancy():
    # Tahminleri yapmak için gelecek yılların tarihlerini oluştur
    future_years = np.arange(2023, 2101).reshape(-1, 1)

    # Tahminleri yap
    future_predictions = model.predict(future_years)

    # Tahminleri görselleştir
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(average_yearly_data['Year'], average_yearly_data['doluluk_orani'], label='Gerçek Doluluk Oranı', color='blue')
    ax.plot(future_years, future_predictions, label='Tahmini Doluluk Oranı', color='red', linestyle='--')
    ax.set_title("Baraj Doluluk Oranları Tahmini")
    ax.set_xlabel("Yıl")
    ax.set_ylabel("Yıllık Ortalama Doluluk Oranı (%)")
    ax.legend()
    ax.grid()

    # Matplotlib figürünü Tkinter penceresine göm
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Tkinter penceresi oluştur
root = tk.Tk()
root.title("Baraj Doluluk Oranı Tahmini")

# Baraj Doluluk Oranlarını Göster butonu
show_button_dams = tk.Button(root, text="Baraj Doluluk Oranlarını Tahmin Et", command=predict_dam_occupancy)
show_button_dams.pack(pady=10)

# Pencereyi çalıştır
root.mainloop()
