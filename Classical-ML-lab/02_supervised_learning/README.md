# 02 · Denetimli Öğrenme — Temel Aileler (Supervised Learning — Core Families)

`Classical-ML-lab` serisinin ikinci kategorisi (23 notebook'luk serinin B
bölümü, roadmap'teki notebook'lar 02-06). Bu klasör, denetimli öğrenmenin beş
temel model ailesini kapsıyor: doğrusal regresyon, doğrusal sınıflandırma,
örnek-tabanlı öğrenme (k-NN), olasılıksal (Naive Bayes) ve marj-tabanlı (SVM)
yaklaşımlar. Her notebook, bir öncekinin üzerine inşa edilecek şekilde
tasarlandı — özellikle `svm_kernels`, `linear_classification_family`'deki
çok-sınıflı stratejilere (OvO/OvR) ve `knn_instance_based`'teki
ölçekleme/boyut laneti derslerine sürekli atıf yapıyor.

## 📁 Klasör Yapısı

```
02_supervised_learning/
├── README.md
├── 03_linear_classification_family.ipynb
├── 04_linear_regression_family.ipynb
├── 05_knn_instance_based.ipynb
├── 06_naive_bayes_family.ipynb
├── 07_svm_kernels.ipynb
└── data/
    ├── breast_cancer.csv
    ├── diabetes.csv
    ├── digits.csv
    ├── insurance.csv
    ├── ionosphere.csv
    ├── iris.csv
    ├── ships.csv
    └── wine.csv
```

> **Hücre kuralı**: Her notebook'ta sırayla Python → R → SQL (DuckDB) uygulaması.
> R hücreleri `%%R` (rpy2) sihirbazıyla çalışıyor; her notebook çözümsüz
> egzersizlerle bitiyor.
>

---

## 📓 03 · Doğrusal Sınıflandırma Ailesi (`03_linear_classification_family.ipynb`)

Lojistik Regresyon'u derinlemesine işleyip (sigmoid, düzenlileştirme, çok
sınıflı Softmax/OvR), üretici (generative) LDA/QDA ile ayırt edici
(discriminative) yaklaşımı karşılaştırıyor; Perceptron ve SGDClassifier ile
klasik/çevrimiçi öğrenme çerçevesini tamamlıyor.

```
1. Genel Çerçeve: Sınıflandırma Problemi ve Dört Felsefe
2. Lojistik Regresyon — Derinlemesine
   2.1 İkili LR: Sigmoid ve Olabilirlik  2.2 Karar Sınırı ve Odds Ratio
   2.3 Düzenlileştirme: L1/L2/Elastic Net  2.4 Softmax vs One-vs-Rest
3. Diskriminant Analizi: LDA ve QDA — Üretici Yaklaşım
   3.1 Önce Veriyi Sonra Sınıfı Modelle  3.2 LDA (ortak kovaryans)
   3.3 QDA (sınıf-bazlı kovaryans)  3.4 LDA vs QDA: Ne Zaman Hangisi?
4. Perceptron — En Eski Doğrusal Sınıflandırıcı
   4.1 Biyolojik Motivasyon  4.2 Yakınsama Teoremi  4.3 Perceptron vs LR
5. SGDClassifier: Stokastik Gradyan İnişiyle Genel Çerçeve
   5.1 Batch/Stokastik/Mini-Batch  5.2 Kayıp Fonksiyonu Seçimi
   5.3 Öğrenme Oranı Zamanlamaları  5.4 `partial_fit` ile Çevrimiçi Öğrenme

🐍 Python: 15 uygulama adımı — ikili/çok sınıflı LR, L1/L2/ElasticNet katsayı
   yolları, karar eşiği (precision/recall ödünleşimi), LDA/QDA karar sınırı
   karşılaştırması, Perceptron yakınsama/yakınsamama, SGD kayıp fonksiyonu +
   öğrenme oranı + `partial_fit`, sentez karşılaştırması
📊 R: `glm(family=binomial)`, `nnet::multinom` (Softmax), `MASS::lda`/`qda`
🗄️ SQL (DuckDB): sınıf-bazlı ortalama/varyans/kovaryans (LDA/QDA'nın arkasındaki
   hesap), confusion matrix, ampirik log-odds ile doğrusallık varsayımı sınama
✏️ Egzersizler: E1–E6 (Python, R, SQL ve sentez)
```

**📦 Veri Setleri** (üçü de `sklearn.datasets` built-in, ağ erişimi gerektirmez):

| Dataset                                   | Kaynak                                                                                                                    | Kullanım                                  |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| Breast Cancer Wisconsin Diagnostic        | [UCI](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) / `sklearn.datasets.load_breast_cancer` | İkili Lojistik Regresyon, SGDClassifier   |
| Wine Recognition                          | [UCI](https://archive.ics.uci.edu/dataset/109/wine) / `sklearn.datasets.load_wine`                                       | Çok sınıflı Softmax/OvR, LDA/QDA       |
| Optical Recognition of Handwritten Digits | [UCI](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits) / `sklearn.datasets.load_digits` | Perceptron, SGDClassifier +`partial_fit` |

---

## 📓 04 · Doğrusal Regresyon Ailesi (`04_linear_regression_family.ipynb`)

OLS'i çoklu bağlantı (multicollinearity) sorunuyla hatırlatıp Ridge/Lasso/
Elastic Net düzenlileştirmesine, ardından aykırı-değer-dayanıklı (robust)
yöntemlere ve Genelleştirilmiş Doğrusal Modellere (GLM) uzanıyor.

```
1. Genel Çerçeve: "Doğrusal Regresyon Ailesi" Ne Demek?
2. OLS Hatırlatma ve Çoklu Bağlantı (Multicollinearity) — VIF
3. Düzenlileştirme I: Ridge Regression (L2)
4. Düzenlileştirme II: Lasso Regression (L1)
5. Elastic Net: Ridge + Lasso Sentezi
6. Dayanıklı (Robust) Regresyon
   6.1 Huber  6.2 RANSAC  6.3 Theil-Sen  6.4 Quantile Regression
7. Genelleştirilmiş Doğrusal Modeller (GLM)
   7.1 Poisson  7.2 Gamma  7.3 Tweedie
8. Karar Haritası: Hangi Yöntemi Ne Zaman Kullanmalı?

🐍 Python: 19 uygulama adımı — OLS/VIF, Ridge/Lasso/ElasticNet katsayı yolları
   ve model karşılaştırması, Huber/RANSAC/Theil-Sen'in bozuk veride
   karşılaştırması, Quantile Regression, Poisson/Gamma/Tweedie GLM, sentez
📊 R: `glmnet` (Ridge/Lasso/ElasticNet), `MASS::rlm` (Huber) + `mblm`
   (Theil-Sen), `quantreg::rq`, GLM (Poisson offset'li, Gamma, Tweedie)
🗄️ SQL (DuckDB): grup ortalamaları, saf SQL'de OLS (`regr_slope`/`regr_r2`),
   Poisson GLM'in SQL karşılığı (kaza oranı), pencere fonksiyonuyla IQR
   aykırı değer tespiti, persentillerle Quantile Regression karşılığı
✏️ Egzersizler: E1–E6 (Python, R, SQL ve sentez)
```

**📦 Veri Setleri:**

| Dataset                           | Kaynak                                                                                                                                    | Kullanım                                          |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| Medical Cost Personal (Insurance) | [Kaggle (mirichoi0218)](https://www.kaggle.com/datasets/mirichoi0218/insurance) / GitHub mirror `stedy/Machine-Learning-with-R-datasets` | OLS/Ridge/Lasso/ElasticNet/Quantile/Gamma/Tweedie  |
| Ships Damage Data                 | McCullagh & Nelder (1989) /[Rdatasets](https://vincentarelbundock.github.io/Rdatasets/) `MASS::ships` mirror                             | Poisson GLM + offset                               |
| Auto MPG                          | `seaborn.load_dataset('mpg')` ([UCI](https://archive.ics.uci.edu/dataset/9/auto+mpg) kaynaklı)                                          | Huber/RANSAC/Theil-Sen (sensör hatası senaryosu) |

---

## 📓 05 · k-En Yakın Komşu ve Örnek Tabanlı Öğrenme (`05_knn_instance_based.ipynb`)

"Tembel" (lazy) öğrenmenin parametrik modellerden farkını işleyip mesafe
metriklerini, k-NN sınıflandırma/regresyonu, Nearest Centroid'i ve boyut
lanetini kapsıyor — ölçeklemenin kritikliği burada ilk kez derinlemesine
işleniyor ve `svm_kernels`'te tekrar referans alınıyor.

```
1. Genel Çerçeve: Parametrik vs Örnek Tabanlı (Lazy) Öğrenme
2. Mesafe Metrikleri: "Yakınlık" Nasıl Tanımlanır?
3. k-NN Sınıflandırma
4. k-NN Regresyon
5. Nearest Centroid Classifier
6. Boyut Laneti (Curse of Dimensionality)
7. k Seçimi: Bias-Variance Ödünleşimi
8. Ölçeklemenin Kritik Rolü
9. Hesaplama Verimliliği: Brute-Force vs KD-Tree vs Ball-Tree

🐍 Python: 11 uygulama adımı — Iris'te karar sınırları (k=1,3,5,15),
   mesafe-ağırlıklı oylama, ölçekleme etkisi (sentetik farklı-ölçekli
   öznitelik), CV ile optimal k, mesafe metrikleri karşılaştırması,
   Nearest Centroid, k-NN regresyon, boyut lanetinin ampirik gösterimi,
   Brute-Force/KD-Tree/Ball-Tree hız karşılaştırması, sentez
📊 R: `class::knn` (sınıflandırma), `FNN::knn.reg` (regresyon), Nearest
   Centroid'in manuel uygulaması
🗄️ SQL (DuckDB): saf SQL'de brute-force k-NN (self-join + window function),
   Nearest Centroid (`GROUP BY`), mesafe-ağırlıklı k-NN regresyon, boyut
   lanetinin SQL'deki ampirik izi
✏️ Egzersizler: E1–E6 (Python, R, SQL ve sentez)
```

**📦 Veri Setleri:**

| Dataset    | Kaynak                                                                                         | Kullanım                               |
| ---------- | ---------------------------------------------------------------------------------------------- | --------------------------------------- |
| Iris       | [UCI](https://archive.ics.uci.edu/dataset/53/iris) / `sklearn.datasets.load_iris`             | k-NN sınıflandırma, Nearest Centroid |
| Diabetes   | `sklearn.datasets.load_diabetes`                                                             | k-NN regresyon                          |
| Ionosphere | [UCI](https://archive.ics.uci.edu/dataset/52/ionosphere) / GitHub mirror `jbrownlee/Datasets` | Ölçekleme etkisi, boyut laneti        |

---

## 📓 06 · Naive Bayes Ailesi (`06_naive_bayes_family.ipynb`)

Bayes teoremini ve MAP kuralını temel alıp dört Naive Bayes varyantını
(Gaussian/Multinomial/Bernoulli/Complement) ve metin sınıflandırma için
Bag-of-Words/TF-IDF pipeline'ını işliyor — LDA/QDA'nın (Notebook 03) üretici
yaklaşımıyla doğrudan karşılaştırma yapıyor.

```
1. Bayes Teoremi ve "Naif" Bağımsızlık Varsayımı
2. Genel Sınıflandırma Çerçevesi: MAP Kuralı ve Log-Olasılıklar
3. Gaussian Naive Bayes
4. Multinomial Naive Bayes
5. Bernoulli Naive Bayes
6. Complement Naive Bayes (Rennie et al., 2003)
7. Metin Sınıflandırma Pipeline'ı: Bag-of-Words ve TF-IDF
8. Naive Bayes: Güçlü/Zayıf Yönler ve Ne Zaman Kullanılır

🐍 Python: 9 uygulama adımı — sıfırdan Gaussian NB (sklearn ile doğrulama),
   GaussianNB tam değerlendirme + PCA karar sınırı, metin ön işleme +
   MultinomialNB, alpha (Laplace/Lidstone) hiperparametre taraması,
   BernoulliNB karşılaştırması, yapay dengesizlik + ComplementNB, TF-IDF,
   en bilgilendirici kelimeler, dört varyantın özet karşılaştırması
📊 R: `e1071::naiveBayes` (Sonar için Gaussian, SMS için 200-kelimelik ikili
   matrisle Bernoulli karşılığı)
🗄️ SQL (DuckDB): GaussianNB parametrelerinin `GROUP BY` ile hesabı ve
   log-olasılık toplamıyla sınıflandırma (sklearn ile %100 eşleşme
   doğrulandı), MultinomialNB'nin P(kelime|sınıf) çekirdeğinin SQL'de
   yeniden üretilip en bilgilendirici kelimelerin sorgulanması
✏️ Egzersizler: E1–E6 (Python, R, SQL ve sentez) — **çözülmeden** teslim edildi
```

**📦 Veri Setleri:**

| Dataset                | Kaynak                                                                                                                                          | Kullanım                                                           |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Sonar, Mines vs. Rocks | [UCI](https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks) / GitHub mirror `jbrownlee/Datasets`                   | GaussianNB (208 örnek, 60 sürekli öznitelik, dengeli sınıflar) |
| SMS Spam Collection    | [UCI](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), Almeida & Hidalgo (2011) / GitHub mirror `justmarkham/pycon-2016-tutorial` | Multinomial/Bernoulli/ComplementNB (5.572 SMS, doğal dengesizlik)  |

---

## 📓 07 · SVM ve Kernel Trick (`07_svm_kernels.ipynb`)

Maksimum marj sınıflandırıcısından soft margin'e, oradan kernel trick'in
matematiğine (Mercer teoremi, linear/poly/RBF/sigmoid) uzanıp SVC (çok
sınıflı) ve SVR (regresyon) varyantlarını işliyor — ölçeklemenin kritikliğini
(Notebook 05'in devamı olarak) ampirik olarak gösteriyor.

```
1. Maksimum Marj Sınıflandırıcısı (Maximal Margin Classifier)
2. Soft Margin ve C Hiperparametresi (hinge loss, dual problem)
3. Kernel Trick: Doğrusal Olmayan Sınırlar İçin Boyut Artırma
   Mercer Teoremi, XOR örneği, linear/polynomial/RBF/sigmoid formülleri
4. SVC: Çok Sınıflı Sınıflandırma (OvO/OvR) ve Karar Fonksiyonu
5. SVR: Destek Vektörü Regresyonu (ε-duyarsız kayıp)
6. Hiperparametre Etkileri (C, γ, derece) ve Ölçeklemenin Kritik Önemi
7. SVM: Güçlü/Zayıf Yönler ve Ne Zaman Kullanılır

🐍 Python: 11 uygulama adımı — sentetik veriyle marj/kernel trick
   görselleştirmesi (XOR, make_circles, make_moons), Wheat Seeds'te çok
   sınıflı kernel karşılaştırması + PCA karar sınırı + GridSearchCV (C×γ ısı
   haritası), Banknote'ta soft margin C taraması + ölçekli/ölçeksiz
   karşılaştırma + `SVC` vs `LinearSVC` hız testi + tam değerlendirme,
   Abalone'da SVR kernel karşılaştırması + ε taraması + OLS baseline, sentez
📊 R: `e1071::svm()` (Seeds kernel karşılaştırması + `tune()` ile C/γ
   araması, Abalone SVR)
🗄️ SQL (DuckDB): fit edilmiş linear ve RBF kernel SVC'nin karar fonksiyonunun
   (destek vektörleri + `dual_coef_` + intercept'ten `CROSS JOIN`/`GROUP BY`
   ile) yeniden üretilmesi, sklearn ile %100 eşleşme doğrulandı
✏️ Egzersizler: E1–E6 (Python, R, SQL ve sentez) — **çözülmeden** teslim edildi
```

**📦 Veri Setleri** (üçü de GitHub mirror üzerinden, orijinal kaynak UCI):

| Dataset                 | Kaynak                                                                                                         | Kullanım                                                                              |
| ----------------------- | -------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Wheat Seeds             | [UCI](https://archive.ics.uci.edu/dataset/236/seeds) / GitHub mirror `jbrownlee/Datasets`                     | Çok sınıflı SVC (210 örnek, 7 geometrik öznitelik, 3 dengeli sınıf)            |
| Banknote Authentication | [UCI](https://archive.ics.uci.edu/dataset/267/banknote+authentication) / GitHub mirror `jbrownlee/Datasets`   | Soft margin, ölçekleme etkisi (1.372 örnek, 4 dalgacık-dönüşümü özniteliği) |
| Abalone                 | [UCI](https://archive.ics.uci.edu/dataset/1/abalone), Nash et al. (1994) / GitHub mirror `jbrownlee/Datasets` | SVR ile yaş tahmini (4.177 örnek, fiziksel ölçümler)                              |

---

## 📚 Kaynak Haritası

### Kitaplar

| #  | Kitap                                    | Yazar                             | Ücretsiz? | Bağlantı                                                                                                                                      |
| -- | ---------------------------------------- | --------------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| 🥇 | An Introduction to Statistical Learning  | James, Witten, Hastie, Tibshirani | ✅         | [statlearning.com](https://www.statlearning.com/)                                                                                                |
| 🥇 | The Elements of Statistical Learning     | Hastie, Tibshirani, Friedman      | ✅         | [hastie.su.domains/ElemStatLearn](https://hastie.su.domains/ElemStatLearn/)                                                                      |
| 🥈 | Pattern Recognition and Machine Learning | Bishop                            | ✅         | [Microsoft Research PDF](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) |
| 🥉 | Statistical Learning Theory              | Vapnik                            | ❌         | Wiley                                                                                                                                           |

### Anahtar Makaleler

| Makale                                                                                                 | Notebook | Konu                                  |
| ------------------------------------------------------------------------------------------------------ | -------- | ------------------------------------- |
| Cortes & Vapnik (1995),*Support-Vector Networks*                                                     | 07       | SVM'i tanıtan orijinal makale        |
| Boser, Guyon & Vapnik (1992),*A Training Algorithm for Optimal Margin Classifiers*                   | 07       | Kernel trick'in SVM'e ilk uygulaması |
| Rennie, Shih, Teevan & Karger (2003),*Tackling the Poor Assumptions of Naive Bayes Text Classifiers* | 06       | ComplementNB'yi tanıtan makale       |

### Resmi Dokümantasyon

| Kaynak                                                                                                                                         | Kullanım                                                          |
| ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| [scikit-learn — Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)                                                      | Notebook 03-04: LR, Ridge/Lasso/ElasticNet, robust regression, GLM |
| [scikit-learn — LDA/QDA](https://scikit-learn.org/stable/modules/lda_qda.html)                                                                 | Notebook 03                                                        |
| [scikit-learn — Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html)                                                     | Notebook 05                                                        |
| [scikit-learn — Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)                                                         | Notebook 06                                                        |
| [scikit-learn — Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)                                                     | Notebook 07                                                        |
| [`glmnet` R paketi (CRAN)](https://cran.r-project.org/web/packages/glmnet/index.html)                                                         | Notebook 04 R uygulaması                                          |
| [`e1071` R paketi (CRAN)](https://cran.r-project.org/web/packages/e1071/e1071.pdf)                                                            | Notebook 06-07 R uygulaması                                       |
| [DuckDB — SQL Fonksiyonları](https://duckdb.org/docs/sql/functions/numeric) & [Window Functions](https://duckdb.org/docs/sql/window_functions) | Tüm notebook'ların SQL bölümlerinin temeli                     |

Notebook içi `Kaynaklar` bölümlerinde her dosya için tam kaynak listesi
mevcut (UCI/Kaggle sayfaları, R paket dokümantasyonları, projedeki önceki
notebook'lara çapraz referanslar dahil).

---
