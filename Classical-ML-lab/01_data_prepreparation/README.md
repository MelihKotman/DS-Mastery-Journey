# 01 · Veri Hazırlığı (Data Preparation)

`Classical-ML-lab` serisinin ilk kategorisi (23 notebook'luk serinin A
bölümü). Bu klasör, her makine öğrenmesi projesinin temelini oluşturan iki
konuyu kapsıyor: verinin görselleştirilerek anlaşılması (EDA) ve modele
girmeden önce temizlenip dönüştürülmesi (preprocessing).

## 📁 Klasör Yapısı

```
01_data_prepreparation/
├── README.md
├── 01_eda_visualization.ipynb
├── 02_data_preprocessing.ipynb
└── data/
    ├── ames_housing.csv
    ├── pima_diabetes.csv
    └── winequality_red.csv
```

> **Hücre kuralı**: Her notebook'ta sırayla Python → R → SQL (DuckDB) uygulaması.
> R hücreleri `%%R` (rpy2) sihirbazıyla çalışıyor; her notebook çözümsüz
> egzersizlerle bitiyor.

---

## 📓 01 · EDA Görselleştirme (`01_eda_visualization.ipynb`)

Hangi grafik türünün ne zaman kullanılacağına dair karar haritasından başlayıp
tek değişkenli, iki değişkenli ve çok değişkenli görselleştirme tekniklerinin
tamamını işliyor.

```
1. Neden Görselleştirme? Sayılar Yalan Söylemez Ama Yanıltabilir
2. Genişletilmiş Grafik Seçim Haritası (Chart Selection Decision Map)
3. Tek Değişkenli Sayısal Dağılım Grafikleri
   3.1 Histogram  3.2 KDE  3.3 Rug Plot  3.4 ECDF
   3.5 Boxplot  3.6 Violin Plot  3.7 Strip/Swarm Plot  3.8 Çarpıklık (Skewness)
4. Tek Değişkenli Kategorik Grafikler — Bar/Count Plot
5. İki Değişkenli (Bivariate) Grafikler
   5.1 Scatter  5.2 Regresyon/Trend Çizgisi  5.3 Joint Plot
   5.4 Hexbin/2D Histogram  5.5 2D KDE (Contour)  5.6 Correlation Heatmap
6. Çok Değişkenli (Multivariate) Grafikler
   6.1 Pairplot  6.2 FacetGrid/Small Multiples  6.3 Clustermap
   6.4 Bubble Chart  6.5 Parallel Coordinates

🐍 Python: seaborn/matplotlib ile 18 uygulama adımı (tam veri, örneklem
   sadece hesaplama açısından zorunlu olduğunda — ör. büyük pairplot)
📊 R: ggplot2 ailesi (ggplot2, corrplot, GGally, ggbeeswarm, ggExtra) ile
   Python adımlarının birebir karşılığı
🗄️ SQL (DuckDB): Grafik çizmeden "grafik okumak" — histogram/boxplot/
   korelasyon istatistiklerini SQL agregasyonlarıyla üretme
✏️ Egzersizler: E1–E6 (Python, R, SQL ve sentez)
```

**📦 Veri Setleri** (üçü de `seaborn.load_dataset()` ile, tam veri kullanıldı):

| Dataset         | Kaynak                                                                                          | Kullanım                                                                   |
| --------------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Titanic         | [Kaggle](https://www.kaggle.com/competitions/titanic)                                            | Kategorik grafikler, hayatta kalma oranı (n=891)                           |
| Diamonds        | [Kaggle](https://www.kaggle.com/datasets/shivam2503/diamonds)                                    | Büyük N, çoğunlukla sayısal — overplotting/hexbin örneği (n=53.940) |
| Palmer Penguins | [Kaggle](https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data) | Küçük, temiz, çok değişkenli EDA (n=333)                              |

---

## 📓 02 · Veri Ön İşleme (`02_data_preprocessing.ipynb`)

Eksik veri, aykırı değer, kategorik encoding, ölçekleme ve temel özellik
mühendisliğini kapsayan, karar haritalarıyla desteklenen bir seri.

```
1. Neden Veri Ön İşleme? "Garbage In, Garbage Out"
2. Veri Ön İşleme Karar Haritası (eksik veri / aykırı değer / encoding / ölçekleme
   için 4 ayrı akış + hızlı referans tablosu)
3. Eksik Veri (Missing Data)
   3.1 Eksiklik Mekanizmaları (Rubin, 1976) — MCAR/MAR/MNAR
   3.2 Listwise/Pairwise Deletion  3.3 Mean/Median/Mode Imputation
   3.4 KNN Imputation  3.5 Iterative Imputer / MICE
   3.6 Eksiklik Göstergesi ve Yapısal Eksiklik
4. Aykırı Değer Tespiti
   4.1 IQR Kuralı (Tukey, 1977)  4.2 Z-Score ve Modified Z-Score
   4.3 Aykırı Değerlerle Ne Yapmalı?
5. Kategorik Değişken Kodlama (Encoding)
   5.1 One-Hot  5.2 Label Encoding ve Tuzağı  5.3 Ordinal
   5.4 Target (Mean) Encoding  5.5 Frequency/Count Encoding
6. Ölçekleme (Feature Scaling)
   6.1 Standardization  6.2 Min-Max  6.3 Robust Scaling  6.4 Log/Power Transform
7. Temel Özellik Mühendisliği — türetilmiş/oran özellikler, binning,
   etkileşim/polinom terimler, tarih/zaman ayrıştırma

🐍 Python: 21 uygulama adımı, son adımda Pipeline + ColumnTransformer ile
   tüm adımların tek bir sentezde birleştirilmesi
📊 R: tidyverse, mice, VIM, naniar ile 8 uygulama adımı
🗄️ SQL (DuckDB): SUMMARIZE, COALESCE+MEDIAN, PERCENTILE_CONT, pencere
   fonksiyonlarıyla (OVER) tek sorguda imputation/outlier/encoding
✏️ Egzersizler: E1–E6 (Python, R, SQL ve sentez)
```

**📦 Veri Setleri** (üçü de gerçek eksik/aykırı veri problemleri taşıyor,
`data/` klasöründe):

| Dataset               | Kaynak                                                                                                                                                                     | Kullanım                                                                                   |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Ames Housing          | [De Cock (2011), JSE](https://jse.amstat.org/v19n3/decock/DataDocumentation.txt) / [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) | Gerçek eksik veri, yapısal eksiklik (MNAR), yüksek kardinaliteli kategorikler            |
| Pima Indians Diabetes | [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) / [UCI](https://archive.ics.uci.edu/dataset/34/diabetes)                                      | Gizlenmiş eksik veri (imkansız sıfır kodlu değerler) — KNN/MICE imputation            |
| Red Wine Quality      | [Kaggle, Cortez et al. 2009](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)                                                                      | Tamamen sayısal, eksik değer yok — aykırı değer/ölçekleme/binning odaklı (n=1.599) |

---

## 📚 Kaynak Haritası

### Kitaplar

| #  | Kitap                                   | Yazar          | Ücretsiz? | Bağlantı                                               |
| -- | --------------------------------------- | -------------- | ---------- | -------------------------------------------------------- |
| 🥇 | Fundamentals of Data Visualization      | Claus O. Wilke | ✅         | [clauswilke.com/dataviz](https://clauswilke.com/dataviz/) |
| 🥇 | Feature Engineering and Selection       | Kuhn & Johnson | ✅         | [feat.engineering](http://www.feat.engineering/)          |
| 🥈 | Statistical Analysis with Missing Data  | Little & Rubin | ❌         | Wiley                                                    |
| 🥉 | An Introduction to Statistical Learning | James vd.      | ✅         | [statlearning.com](https://www.statlearning.com/)         |

### Resmi Dokümantasyon & Kurslar

| Kaynak                                                                                                                                | Kullanım                                               |
| ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| [scikit-learn — Imputation of Missing Values](https://scikit-learn.org/stable/modules/impute.html)                                    | `SimpleImputer`, `KNNImputer`, `IterativeImputer` |
| [scikit-learn — Preprocessing Data](https://scikit-learn.org/stable/modules/preprocessing.html)                                       | Scaler'lar ve encoder'lar                               |
| [scikit-learn — ColumnTransformer / Pipeline](https://scikit-learn.org/stable/modules/compose.html)                                   | Notebook 02'nin sentez adımının API referansı       |
| [Seaborn — Distributions / Relational Tutorials](https://seaborn.pydata.org/tutorial/distributions.html)                              | Notebook 01'in Python görselleştirme API'si           |
| [Van Buuren &amp; Groothuis-Oudshoorn (2011), JSS](https://www.jstatsoft.org/article/view/v045i03)                                     | MICE yönteminin orijinal makalesi                      |
| [DuckDB — Window Functions](https://duckdb.org/docs/sql/window_functions) & [SUMMARIZE](https://duckdb.org/docs/guides/meta/summarize) | Her iki notebook'un SQL bölümlerinin temeli           |
| [Kaggle Learn — Data Cleaning](https://www.kaggle.com/learn/data-cleaning)                                                            | Kısa, pratik mikro-kurs                                |

Notebook içi `Kaynaklar` bölümlerinde her iki dosya için de tam kaynak listesi
mevcut (Kaggle sayfaları, R paket dokümantasyonları, projedeki önceki
notebook'lara çapraz referanslar dahil).

---
