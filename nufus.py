import tkinter as tk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LinearRegression

# Verileri yükle
file_path = "C:/Users/emrek/Desktop/hackathon/50yil_nufus.csv"
# Veri yolunu belirtin
data = pd.read_csv(file_path)

# "Nüfus" sütunundaki virgülleri kaldır ve float'a dönüştür
data["nufus"] = data["nufus"].replace(",", "", regex=True).astype(float)

# Lineer regresyon modelini oluştur
model = LinearRegression()

# Verileri X ve y olarak ayır
X = data["yil"].values.reshape(-1, 1) # Sütun adını veri setinizdeki adla değiştirin
y = data["nufus"].values

# Modeli eğit
model.fit(X, y)

# Tahmin yapılacak yılları oluştur (2024-2034 arası)
tahmin_yillari = np.arange(2024, 2060).reshape(-1, 1)

# Tahmin yap
nufus_tahminleri = model.predict(tahmin_yillari)

# Tahmin edilen nüfus verilerini DataFrame'e dönüştür
tahmin_verisi = pd.DataFrame({"Yıl": tahmin_yillari.flatten(), "Tahmini Nüfus": nufus_tahminleri})

# Tkinter penceresi oluştur
root = tk.Tk()
root.title("Nüfus Projeksiyonu")

# Grafik için boş bir figür oluştur
fig, ax = plt.subplots(figsize=(15, 9))
plot_canvas = FigureCanvasTkAgg(fig, master=root)
plot_canvas.get_tk_widget().pack()

# Nüfus Grafiği Gösterme Fonksiyonu
def show_population_graph():
    # Grafiği temizle
    ax.clear()
    # Mevcut nüfus verisini çiz
    ax.bar(data['yil'], data['nufus'], color='skyblue', label='Mevcut Veri (Milyon)') # Sütun adlarını veri setinizdekilerle değiştirin
    # Tahmini nüfus verisini çiz
    ax.plot(tahmin_verisi['Yıl'], tahmin_verisi['Tahmini Nüfus'], marker='o', linestyle='-', color='orange', label='Tahmini Veri (Milyon)')
    ax.set_title("Nüfus Projeksiyonu")
    ax.set_xlabel("Yıl")
    ax.set_ylabel("Nüfus")
    ax.grid(True)
    ax.legend()

    # Nüfus değerlerini virgüllü şekilde göster
    def format_func(x, pos):
        return f'{x:,.2f}'

    ax.yaxis.set_major_formatter(FuncFormatter(format_func))

    # Grafiği güncelleyerek Tkinter penceresine göm
    plot_canvas.draw()

# Grafik penceresini doğrudan aç
show_population_graph()

# Pencereyi çalıştır
root.mainloop()
