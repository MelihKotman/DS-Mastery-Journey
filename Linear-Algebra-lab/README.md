# Linear Algebra Lab — Tam Rehber
## Yapı · Taslak · Dataset · Kaynaklar

---

## 📁 Klasör Yapısı

```
linear-algebra-lab/
├── README.md
├── 01_vectors_matrices_basics/
│   ├── vectors_matrices_basics.ipynb
│   └── data/
├── 02_eigenvalues_eigenvectors/
│   ├── eigenvalues_eigenvectors.ipynb
│   └── data/
├── 03_svd_pca/
│   ├── svd_pca.ipynb
│   └── data/
├── 04_norms_regularization/
│   ├── norms_regularization.ipynb
│   └── data/
└── 05_special_matrices/
    ├── special_matrices.ipynb
    └── data/
```

> **Hücre kuralı**: Her notebook Python tek dilde ilerliyor

---

## 📓 Notebook Taslakları & Dataset Önerileri

---

### 01 · Vektör ve Matris Temelleri (`vectors_matrices_basics.ipynb`)

```
# 1. Vektörler, Matrisler ve Lineer Dönüşümler

## 1.1 Teorik Özet (Markdown)
   - Vektör: yön + büyüklük, matris: vektörleri dönüştüren bir "makine"
   - Temel işlemler: toplama, skaler çarpım, iç çarpım (dot product)
   - KRİTİK SEZGİ: Matris çarpımı = LİNEER DÖNÜŞÜM (rotasyon, ölçekleme, shear'in hepsi
     birer matris)
   - Determinant: bir dönüşümün alanı/hacmi ne kadar "büyüttüğü/küçülttüğü"
   - Matris tersi (inverse): dönüşümü GERİ ALAN matris; determinant=0 ise tersi YOK

## 1.2 [PYTHON] NumPy ile Temel İşlemler
   - Vektör/matris oluşturma, toplama, çarpma, transpoze

## 1.3 [PYTHON] Matris Çarpımını Geometrik Dönüşüm Olarak Görselleştirme
   - Bir kareyi (birim kare) farklı matrislerle çarpıp nasıl döndüğünü/eğrildiğini/
     büyüdüğünü adım adım gösterme
   - Rotasyon matrisi, ölçekleme matrisi, shear matrisi — üçünü yan yana

## 1.4 [PYTHON] Determinant ve Tersini Alma
   - Determinantın "alan ölçekleme faktörü" olduğunu görsel kanıtlama (dönüşüm öncesi/sonrası alan)
   - Determinant=0 olan (tekil/singular) bir matrisin NEDEN tersinin olmadığını görsel gösterme

## 1.5 Egzersiz: Kendi 2D şeklini (basit bir "ev" çizimi gibi) 3 farklı matrisle dönüştür
```

**📦 Dataset:**
| Dataset | Kaynak | Kullanım |
|---|---|---|
| Sentetik 2D noktalar/şekiller | `numpy` (elle üretilen) | Geometrik dönüşüm görselleştirmesi — kontrollü şekiller (kare/ev çizimi) matris çarpımının etkisini en net gösteren yöntem, bu yüzden bilerek sentetik tutuldu |

---

### 02 · Özdeğerler ve Özvektörler (`eigenvalues_eigenvectors.ipynb`)

```
# 2. Eigenvalue / Eigenvector — PCA'nın Matematiksel Temeli

## 2.1 Teorik Özet (Markdown)
   - Tanım: Av = λv — dönüşüm sonrası YÖNÜ değişmeyen (sadece uzunluğu λ katına çıkan) vektörler
   - Karakteristik denklem: det(A - λI) = 0
   - Geometrik yorum: özvektörler, bir dönüşümün "doğal eksenleri"
   - Neden önemli: PCA, Google'ın PageRank'i, titreşim analizi (mühendislik) hep bu kavrama dayanıyor

## 2.2 [PYTHON] Karakteristik Denklemi Elle Çözme
   - 2x2'lik somut bir matrisin özdeğerlerini elle (karakteristik polinom) bul
   - np.linalg.eig() ile doğrula

## 2.3 [PYTHON] Geometrik Görselleştirme
   - Bir dönüşüm matrisinin özvektörlerini, dönüşümden ÖNCE ve SONRA çizip
     "yönlerinin değişmediğini" görsel kanıtlama

## 2.4 [PYTHON] Kovaryans Matrisinin Özdeğerleri — Finansal Risk Faktörü Analizi
   - Gerçek hisse senedi getirilerinden (5-6 teknoloji hissesi) kovaryans matrisi hesapla
   - Özdeğer/özvektörlerini bul — en büyük özdeğere karşılık gelen özvektör, portföyün
     "en büyük ortak risk faktörünü" (genelde "piyasa geneliyle birlikte hareket etme"
     eğilimini) temsil ediyor — bu, gerçek portföy risk yönetiminde kullanılan bir teknik

## 2.5 Egzersiz: Farklı bir 3x3 matrisin özdeğerlerini elle ve kodla bul, karşılaştır
```

**📦 Dataset:**
| Dataset | Kaynak | Kullanım |
|---|---|---|
| Hisse Senedi Getirileri (5-6 teknoloji hissesi) | `yfinance` (gerçek zamanlı) | Kovaryans matrisi → özdeğer/özvektör → risk faktörü analizi |
| Sentetik 2x2/3x3 matrisler | `numpy` | Elle çözüm alıştırması |

---

### 03 · SVD ve PCA (`svd_pca.ipynb`)

```
# 3. Singular Value Decomposition (SVD) ve PCA'yı Sıfırdan Kurma

## 3.1 Teorik Özet (Markdown)
   - SVD: HERHANGİ bir matris A = U Σ V^T şeklinde ayrıştırılabilir (eigenvalue'nün
     kare-olmayan matrislere genellemesi)
   - PCA ile ilişki: PCA aslında verinin kovaryans matrisi üzerinde SVD/eigendecomposition uygulamaktır
   - Boyut indirgeme mantığı: en büyük tekil değerleri (singular values) tutup küçükleri atmak

## 3.2 [PYTHON] PCA'yı Sıfırdan Kurma (Kütüphanesiz)
   - Veriyi merkezle → kovaryans matrisi → özdeğer/özvektör → verileri yeni eksenlere
     projekte et
   - sklearn.decomposition.PCA ile karşılaştırıp SONUÇLARIN ÖRTÜŞTÜĞÜNÜ doğrula

## 3.3 [PYTHON] Görüntü Sıkıştırma: SVD'nin En Görsel Uygulaması
   - Gerçek bir görüntüyü SVD ile ayrıştır, sadece ilk k tekil değeri kullanarak
     "sıkıştırılmış" halini geri kur
   - k arttıkça görüntü kalitesinin nasıl iyileştiğini göster (klasik ve çarpıcı bir demo)

## 3.4 [PYTHON] Gerçek Veride PCA: Boyut İndirgeme ve Görselleştirme
   - Çok boyutlu bir çalışan verisini (HR Analytics, ~10+ sayısal özellik) 2 boyuta indirip görselleştirme
   - Açıklanan varyans oranı (explained variance ratio) — kaç bileşen "yeterli"
   - İşten ayrılan/ayrılmayan çalışanların 2 boyutlu PCA uzayında nasıl ayrıştığını (veya ayrışmadığını) gözlemleme

## 3.5 Egzersiz: Kendi seçtiğin bir görüntüyü farklı k değerleriyle sıkıştır, dosya boyutu/kalite dengesini tart
```

**📦 Dataset:**
| Dataset | Kaynak | Kullanım |
|---|---|---|
| `skimage.data` örnek görüntüleri | `scikit-image` (built-in, indirme gerekmez) | SVD ile görüntü sıkıştırma demosu |
| [IBM HR Analytics Employee Attrition](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) | Kaggle | Çok boyutlu çalışan verisini PCA ile 2 boyuta indirme |

---

### 04 · Normlar ve Regularizasyon (`norms_regularization.ipynb`)

```
# 4. L1/L2 Normları ve Regularizasyonun Geometrisi

## 4.1 Teorik Özet (Markdown)
   - L1 norm (Manhattan): Σ|x_i| — birim "çemberi" bir ELMAS (köşeli) şeklinde
   - L2 norm (Öklid): √(Σx_i²) — birim çemberi GERÇEK bir daire
   - HATIRLATMA: `statistical-inference-lab` 03'te (Ridge=MAP) Gauss prior'un L2'ye,
     Laplace prior'un L1'e karşılık geldiğini görmüştük — burada GEOMETRİK tarafını görüyoruz

## 4.2 [PYTHON] Normları Görselleştirme
   - L1 ve L2 birim "çemberlerini" (elmas vs daire) aynı grafikte çizme

## 4.3 [PYTHON] Ridge (L2) Regresyonun Geometrik Yorumu
   - Neden L2 regularizasyon katsayıları KÜÇÜLTÜYOR ama NADİREN tam sıfıra indiriyor
     (elips + daire kesişiminin geometrik gösterimi)

## 4.4 [PYTHON] Lasso (L1) Regresyonun "Seyrek" (Sparse) Özelliği
   - Neden L1 regularizasyon katsayıları TAM SIFIRA itiyor (elips + elmas kesişiminin
     KÖŞEDE olma eğilimi) — Ridge'den bu yönüyle temelden farklı
   - Gerçek veride (İkinci El Araba Fiyatları) Ridge vs Lasso katsayı karşılaştırması —
     kilometre, motor hacmi, yaş gibi birbiriyle doğal olarak ilişkili özellikler var,
     Lasso'nun hangilerini "gereksiz" bulup sıfırladığını gözlemleme

## 4.5 Egzersiz: Kendi sentetik regresyon verinde, çok sayıda gereksiz (gürültülü) özellik
   ekleyip Lasso'nun bunları nasıl "elediğini" (sıfırladığını) göster
```

**📦 Dataset:**
| Dataset | Kaynak | Kullanım |
|---|---|---|
| [Used Car Price Prediction](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset) | Kaggle | Ridge vs Lasso gerçek katsayı karşılaştırması |
| Sentetik regresyon verisi | `sklearn.datasets.make_regression()` | Lasso'nun seyreklik özelliğini net göstermek için |

---

### 05 · Özel Matrisler (`special_matrices.ipynb`)

```
# 5. Pozitif Tanımlı Matrisler ve Konvekslik

## 5.1 Teorik Özet (Markdown)
   - Simetrik matris: A = A^T
   - Pozitif tanımlı (Positive Definite): TÜM özdeğerleri pozitif — VEYA eşdeğer olarak,
     her v≠0 için v^T A v > 0
   - Neden önemli: kovaryans matrisleri HER ZAMAN pozitif YARI-tanımlı (PSD) olmak zorunda
   - Konvekslik bağlantısı: bir fonksiyonun Hessian'ı pozitif tanımlıysa, fonksiyon
     KONVEKSTİR — yani optimizasyon problemi "tek bir global minimuma" sahiptir
     (`calculus-lab`'a doğrudan köprü)

## 5.2 [PYTHON] Pozitif Tanımlılık Testi
   - Bir matrisin özdeğerlerini hesaplayıp hepsinin pozitif olup olmadığını kontrol etme

## 5.3 [PYTHON] Kovaryans Matrisi HER ZAMAN Neden PSD'dir
   - `probability-lab` 04b'deki kovaryans formülünden başlayıp, HERHANGİ bir veri
     kümesinin kovaryans matrisinin neden asla negatif özdeğere sahip olamayacağını
     cebirsel + görsel kanıtlama
   - Gerçek hava kalitesi sensör verisinde (birden fazla kirletici ölçümü) kovaryans
     matrisini hesaplayıp özdeğerlerin gerçekten hep negatif-olmadığını doğrulama

## 5.4 [PYTHON] Konvekslik Testi: Hessian ile
   - Basit bir fonksiyonun (örn. f(x,y)=x²+y²) Hessian'ını hesaplayıp pozitif tanımlı
     olduğunu doğrulama — `calculus-lab`'daki Jacobian/Hessian notebook'una doğrudan önizleme

## 5.5 Egzersiz: Kasıtlı olarak pozitif tanımlı OLMAYAN bir matris üret, optimizasyon
   algoritmasının (basit bir gradient descent) bu durumda nasıl "kararsızlaştığını" göster
```

**📦 Dataset:**
| Dataset | Kaynak | Kullanım |
|---|---|---|
| Sentetik matrisler | `numpy` | Pozitif tanımlılık/konvekslik testleri |
| [Air Quality Dataset (UCI)](https://archive.ics.uci.edu/dataset/360/air+quality) | UCI | Gerçek sensör verisinde kovaryans matrisinin PSD olduğunu kanıtlama |

---

## 📚 Kaynak Haritası

### Kitaplar

| # | Kitap | Yazar | Ücretsiz? | Bağlantı |
|---|---|---|---|---|
| 🥇 | Linear Algebra Done Right | Sheldon Axler | ❌ (bazı baskıları ✅) | Springer |
| 🥇 | Introduction to Linear Algebra | Gilbert Strang | ❌ | Wellesley-Cambridge |
| 🥈 | Mathematics for Machine Learning | Deisenroth, Faisal, Ong | ✅ | [mml-book.github.io](https://mml-book.github.io) |
| 🥉 | No Bullshit Guide to Linear Algebra | Ivan Savov | ❌ | Minireference |

### Kurslar & Görsel Referans

| Kaynak | Kullanım |
|---|---|
| [3Blue1Brown — Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) | Bu lab'ın TEK EN ÖNEMLİ referansı — görsel sezgi için birebir |
| [MIT 18.06 — Gilbert Strang](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) | Klasik, kapsamlı üniversite dersi |
| [Seeing Theory](https://seeing-theory.brown.edu) | Olasılık ağırlıklı ama kovaryans matrisi görselleştirmeleri var |
| [setosa.io — Eigenvectors](https://setosa.io/ev/eigenvectors-and-eigenvalues/) | Özdeğer/özvektör için interaktif görselleştirme |
| [NumPy `linalg` Dokümantasyonu](https://numpy.org/doc/stable/reference/routines.linalg.html) | Bu lab'ın ana Python aracı |

