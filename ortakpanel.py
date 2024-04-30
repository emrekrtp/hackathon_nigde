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
from matplotlib.ticker import FuncFormatter

# Verileri yükle
file_path = "C:/Users/emrek/Desktop/hackathon/water-consumption.csv"
data = pd.read_csv(file_path)

# Şehirlere göre verileri grupla
grouped_data = data.groupby('District').sum().reset_index()

# Sadece kullanım miktarları sütunlarını seç
years = ['2015 (Consumption-m3)', '2016 (Consumption-m3)', '2017 (Consumption-m3)', '2018 (Consumption-m3)', '2019 (Consumption-m3)']

# Tkinter penceresi oluştur
root = tk.Tk()
root.title("Ana Menü")

# Şehir Seçim ve Tahmini Kullanım Miktarı Arayüzü
def open_water_consumption_interface():
    water_consumption_window = tk.Toplevel(root)
    water_consumption_window.title("Şehir Seçim ve Tahmini Kullanım Miktarı")

    # Şehirleri içeren bir liste
    cities = grouped_data['District'].tolist()

    # Combobox oluştur
    selected_city = tk.StringVar()
    city_combobox = ttk.Combobox(water_consumption_window, textvariable=selected_city, values=cities)
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

        # Gelecek 10 yılın tahmini kullanım miktarını hesapla (milyar metreküp)
        future_years = [[year] for year in range(2020, 2030)]
        future_predictions = model.predict(future_years) / 1e6  # m3 -> milyar m3

        # Grafik oluşturma
        plt.figure(figsize=(8, 6))
        plt.bar(range(2020, 2030), future_predictions.flatten(), color='skyblue')
        plt.title(f"Gelecek 10 Yılın Tahmini Su Kullanım Miktarı - {city_name} İlçesi")
        plt.xlabel("Yıl")
        plt.ylabel("Tahmini Kullanım Miktarı (milyar m3)")  # Kullanım miktarını milyar metreküp cinsinden göster
        plt.grid(axis='y')

        # Yatay ekseni (x ekseni) düzenle
        plt.xticks(range(2020, 2030), [str(year) for year in range(2020, 2030)])  # Yılları sayı olarak göster

        # Sütunların üzerine sayıları yazdır
        for i, value in enumerate(future_predictions.flatten()):
            plt.text(2020 + i, value + 0.2, f"{value:.2f}", ha='center')

        plt.show()

        # Su kullanımının artışını yorumla
        if future_predictions[-1] > future_predictions[0]:
            print(f"{city_name} ilçesinde su kullanımı artış eğiliminde.")
            # Çözüm önerileri
            print("Su tasarrufu için bilinçlendirme kampanyaları ve altyapı yatırımları artırılabilir.")
        else:
            print(f"{city_name} ilçesinde su kullanımında durağan veya azalma eğilimi var.")
            # Çözüm önerileri
            print("Mevcut su kullanım alışkanlıkları sürdürülebilirlik odaklı gözden geçirilebilir.")

    # Tahminleri göster butonu
    show_button = tk.Button(water_consumption_window, text="Tahminleri Göster", command=show_predictions)
    show_button.pack(pady=10)

# Baraj Doluluk Oranı Tahmini Arayüzü
def open_dam_occupancy_interface():
    dam_occupancy_window = tk.Toplevel(root)
    dam_occupancy_window.title("Baraj Doluluk Oranı Tahmini")

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
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(average_yearly_data['Year'], average_yearly_data['doluluk_orani'], label='Gerçek Doluluk Oranı', color='blue')
        ax.plot(future_years, future_predictions, label='Tahmini Doluluk Oranı', color='red', linestyle='--')
        ax.set_title("Baraj Doluluk Oranları Tahmini")
        ax.set_xlabel("Yıl")
        ax.set_ylabel("Yıllık Ortalama Doluluk Oranı (%)")
        ax.legend()
        ax.grid()

        # Matplotlib figürünü Tkinter penceresine göm
        canvas = FigureCanvasTkAgg(fig, master=dam_occupancy_window)
        canvas.draw()
        canvas.get_tk_widget().pack()

    # Baraj Doluluk Oranlarını Göster butonu
    show_button_dams = tk.Button(dam_occupancy_window, text="Baraj Doluluk Oranlarını Tahmin Et", command=predict_dam_occupancy)
    show_button_dams.pack(pady=10)

# "Şehir Seçim ve Tahmini Kullanım Miktarı" arayüzüne erişim düğmesi
water_consumption_button = tk.Button(root, text="Şehir Seçim ve Tahmini Kullanım Miktarı", command=open_water_consumption_interface)
water_consumption_button.pack(pady=10)

# "Baraj Doluluk Oranı Tahmini" arayüzüne erişim düğmesi
dam_occupancy_button = tk.Button(root, text="Baraj Doluluk Oranı Tahmini", command=open_dam_occupancy_interface)
dam_occupancy_button.pack(pady=10)

# Nüfus Projeksiyonu Arayüzüne erişim düğmesi
def show_population_graph():
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

    # Tahmin yapılacak yılları oluştur (2024-2060 arası)
    tahmin_yillari = np.arange(2024, 2060).reshape(-1, 1)

    # Tahmin yap
    nufus_tahminleri = model.predict(tahmin_yillari)

    # Tahmin edilen nüfus verilerini DataFrame'e dönüştür
    tahmin_verisi = pd.DataFrame({"Yıl": tahmin_yillari.flatten(), "Tahmini Nüfus": nufus_tahminleri})

    # Tkinter penceresi oluştur
    population_window = tk.Toplevel(root)
    population_window.title("Nüfus Projeksiyonu")

    # Grafik için boş bir figür oluştur
    fig, ax = plt.subplots(figsize=(9, 4))
    plot_canvas = FigureCanvasTkAgg(fig, master=population_window)
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

# Nüfus Projeksiyonu Arayüzüne erişim düğmesi
population_projection_button = tk.Button(root, text="Nüfus Projeksiyonu", command=show_population_graph)
population_projection_button.pack(pady=10)

# Pencereyi çalıştır
root.mainloop()
