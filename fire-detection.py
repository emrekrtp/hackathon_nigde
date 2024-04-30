from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Veri setini hazırla
veri = [
    {"başlık": "Yoğun trafik", "açıklama": "Dünya Bankası, yoksulluğu azaltmak ve dünya genelinde ekonomik eşitsizlikleri gidermek amacıyla yeni bir program başlattı.", "kategori": "teknoloji"},
    {"başlık": "Trafik kazaları arttı", "açıklama": "Gelişmiş ülkelerde işsizlik oranı son çeyrekte beklenenden daha düşük seviyeye geriledi.", "kategori": "ekonomi"},
    {"başlık": "Yeni Teknolojik Keşif, Endüstride Devrim Yarattı", "açıklama": "Bilim adamları, endüstride devrim yaratacak yeni bir teknolojik keşif yaptı.", "kategori": "pos"}, # "pos" kategorisine ait bir örnek ekledik
    {"başlık": "Yeni Film Festivali, Sinema Tutkunlarını Buluşturdu", "açıklama": "bilim ışında ilerleyen çalışanları bir araya getirdi ve büyük ilgi gördü.", "kategori": "koş"}
]

# Metin verilerini ve kategorileri ayır
metinler = [haber["başlık"] + " " + haber["açıklama"] for haber in veri]
kategori = [haber["kategori"] for haber in veri]

# TF-IDF vektörlerine dönüştür
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(metinler)

# Eğitim ve test setlerini ayır
X_train, X_test, y_train, y_test = train_test_split(X, kategori, test_size=0.2, random_state=42)

# Destek Vektör Makinesi (SVM) sınıflandırıcı modelini eğit
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Modelin doğruluğunu değerlendir
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model doğruluğu:", accuracy)

def yapay_zeka_tahmini(yeni_baslik, yeni_aciklama):
    metin = yeni_baslik + " " + yeni_aciklama
    metin_vector = vectorizer.transform([metin])
    tahmin = model.predict(metin_vector)
    
    return tahmin[0]

# Yapay zekadan gelecek tahmini alma
yeni_baslik = "Yeni İş İlanları, İstihdam Piyasasını Canlandırdı"
yeni_aciklama = "Şirketlerin yayınladığı yeni iş ilanları, işsizlik oranının düşmesine katkı sağladı."
tahmin = yapay_zeka_tahmini(yeni_baslik, yeni_aciklama)
print("Yapay zeka tahmini:", tahmin)
