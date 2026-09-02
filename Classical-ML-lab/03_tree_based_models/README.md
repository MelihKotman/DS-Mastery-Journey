# 03 · Ağaç Tabanlı Yöntemler (Tree-Based Models)

`Classical-ML-lab` serisinin üçüncü kategorisi (23 notebook'luk serinin C
bölümü, roadmap'teki notebook'lar 08-10). Bu klasör, tek bir kararsız CART
ağacından başlayıp iki ana ensemble stratejisine uzanıyor: **paralel**
(bagging → Random Forest → Extra Trees) ve **sıralı** (AdaBoost → GBM →
XGBoost/LightGBM/CatBoost). Her notebook bir öncekinin üzerine inşa edildi —
`09_bagging_random_forest`, `08_decision_trees`'teki MDI kavramını
permütasyon önemiyle karşılaştırarak yeniden ziyaret ediyor; `10_gradient_boosting_family`
ise hem MDI/permütasyon tartışmasına hem de `09`'un paralel ensemble
yaklaşımına doğrudan atıf yapıp sıralı ensemble ile karşılaştırıyor.

## 📁 Klasör Yapısı

```
03_tree_based_models/
├── README.md
├── 08_decision_trees.ipynb
├── 09_bagging_random_forest.ipynb
├── 10_gradient_boosting_family.ipynb
└── data/
    ├── gini_entropi_olcek_karsilastirmasi.svg
    └── gini_saflik_karsilastirmasi.svg
```

> **Hücre kuralı**: Her notebook'ta sırayla Python → R → SQL (DuckDB) uygulaması.
> R hücreleri `%%R` (rpy2) sihirbazıyla çalışıyor; her notebook çözümsüz
> egzersizlerle bitiyor.
>

---

## 📓 08 · Karar Ağaçları (`08_decision_trees.ipynb`)

CART algoritmasını Gini/Entropi kriterleriyle sıfırdan inşa edip, aşırı
öğrenme ile ön-budama/son-budama (maliyet-karmaşıklık) stratejilerini
işliyor; özellik önemi (MDI) kavramını burada tanıtıp `09` ve `10`'da tekrar
ele alıyor — serideki ilk, kategorik değişkenlerle native çalışabilen ve
doğrudan yorumlanabilir model ailesi.

```
1. Karar Ağaçlarına Giriş: Sezgi ve Terminoloji
2. Saflık (Impurity) Ölçütleri: Gini Index, Entropi ve Bilgi Kazancı
3. CART Algoritması: Sınıflandırma Ağaçları (Recursive Binary Splitting)
4. Regresyon Ağaçları: Varyans / SSE Azaltma
5. Aşırı Öğrenme ve Budama (Overfitting & Pruning)
6. Özellik Önemi (Feature Importance): Mean Decrease in Impurity (MDI)
7. Kategorik Değişkenler ve CART'ın Sınırlamaları

🐍 Python: 14 uygulama adımı — sıfırdan Gini/Entropi fonksiyonları + kök
   düğüm en iyi bölünme araması, basit recursive CART sınıflandırıcısı
   (`scikit-learn` ile doğrulama), `plot_tree` görselleştirme, Gini vs
   Entropi karşılaştırması, Adult Income'da aşırı öğrenme demosu (`max_depth`
   taraması), `GridSearchCV` ön-budama, maliyet-karmaşıklık son-budama yolu,
   MDI feature importance, Bike Sharing'de regresyon ağacı (basamak
   fonksiyonu), üç veri seti sentezi
📊 R: `rpart` + `rpart.plot` (Heart Disease sınıflandırma), `printcp`/
   `plotcp`/`prune.rpart` (Adult Income maliyet-karmaşıklık budaması),
   `rpart(method="anova")` (Bike Sharing regresyon + değişken önemi)
🗄️ SQL (DuckDB): Heart Disease kök düğümü için Gini impuritesinin hesabı,
   Bike Sharing `yr` bölünmesi için `VAR_POP` ile SSE azalmasının hesabı
✏️ Egzersizler: E1–E6 (Python, R, SQL ve sentez) — **çözülmeden** teslim edildi
```

**📦 Veri Setleri:**

| Dataset | Kaynak | Kullanım |
| --- | --- | --- |
| Heart Disease Cleveland | [UCI](https://archive.ics.uci.edu/dataset/45/heart+disease) / GitHub mirror `dphi-official/Datasets` | Sıfırdan CART, Gini/Entropi karşılaştırması (303 hasta, 13 klinik özellik) |
| Adult Census Income | [UCI](https://archive.ics.uci.edu/dataset/2/adult) / GitHub mirror `jbrownlee/Datasets` | Aşırı öğrenme eğrisi, ön/son-budama, MDI (48.842 kişi) |
| Bike Sharing | [UCI](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) / GitHub mirror `christophM/interpretable-ml-book` | Regresyon ağacı, sızıntı kontrolü (728 günlük kayıt) |

---

## 📓 09 · Bagging ve Random Forest (`09_bagging_random_forest.ipynb`)

Tek bir kararsız (yüksek varyans) ağacın Bootstrap Aggregating (Bagging) ile
nasıl kararlılaştırıldığını işleyip, Random Forest'ın ek rastgele özellik
alt kümesiyle ağaçlar arası korelasyonu nasıl kırdığını, Extra Trees'in
ekstra rastgeleliğini ve Out-of-Bag (OOB) tahmininin "ücretsiz" çapraz
doğrulama olarak nasıl çalıştığını kapsıyor — `08`'deki MDI'yi permütasyon
önemiyle karşılaştırarak yeniden ziyaret ediyor.

```
1. Ensemble Öğrenmeye Giriş: Neden Birden Çok Model?
2. Bootstrap Yeniden Örnekleme (Bootstrap Resampling) ve Bagging (Bootstrap Aggregating)
3. Out-of-Bag (OOB) Tahmin: Ücretsiz Çapraz Doğrulama
4. Random Forest: Bagging + Rastgele Özellik Alt Kümesi (Feature Subsampling)
5. Extra Trees (Extremely Randomized Trees): Ekstra Rastgelelik
6. Özellik Önemi Yeniden Ziyaret: MDI vs Permütasyon Önemi (Permutation Importance)
7. Hiperparametreler ve Pratik Rehber

🐍 Python: 13 uygulama adımı — $1-1/e$ bootstrap teorisinin deneysel
   doğrulanması, sıfırdan manuel Bagging sınıflandırıcı (`BaggingClassifier`
   ile doğrulama), tek ağaç vs Bagging varyans azaltma demosu, OOB skorunun
   manuel hesabı + `oob_score_` doğrulaması, `RandomForestClassifier`'da
   `n_estimators`/`max_features` taramaları, MDI vs permütasyon önemi
   (Telco Churn sürücüleri), `RandomForestRegressor` vs
   `ExtraTreesRegressor` hız/performans karşılaştırması, üç veri seti sentezi
📊 R: `randomForest` — Glass Identification (OOB hata + değişken önemi),
   Telco Churn (MeanDecreaseAccuracy vs MeanDecreaseGini), California
   Housing (`%IncMSE` ve OOB $R^2$)
🗄️ SQL (DuckDB): Glass'ta Bagging'in çoğunluk oylamasının yeniden üretimi,
   Telco'da Random Forest özellik öneminin `GROUP BY`/`AVG` ile toplanması
✏️ Egzersizler: E1–E6 (Python, R, SQL ve sentez) — **çözülmeden** teslim edildi
```

**📦 Veri Setleri:**

| Dataset | Kaynak | Kullanım |
| --- | --- | --- |
| Glass Identification | [UCI](https://archive.ics.uci.edu/dataset/42/glass+identification) / GitHub mirror `jbrownlee/Datasets` | Bootstrap doğrulama, manuel Bagging (214 örnek, 6 dengesiz sınıf) |
| Telco Customer Churn | [Kaggle (blastchar)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) / GitHub mirror `IBM/telco-customer-churn-on-icp4d` | `n_estimators`/`max_features` taraması, MDI vs permütasyon önemi (7.043 müşteri) |
| California Housing | Pace & Barry (1997) / GitHub mirror `ageron/handson-ml2` | `RandomForestRegressor` vs `ExtraTreesRegressor` (20.640 blok grubu) |

---

## 📓 10 · Gradient Boosting Ailesi (`10_gradient_boosting_family.ipynb`)

Sıralı (sequential) ensemble mantığını — zayıf öğrenicilerin hatalarını
düzelte düzelte toplanması — AdaBoost'un ağırlıklı oylamasından Gradient
Boosting Machines'in fonksiyon uzayında gradyan inişine, oradan üç modern
kütüphaneye (XGBoost, LightGBM, CatBoost) uzanarak işliyor. Kategori C'nin
(Ağaç Tabanlı Yöntemler) son ve en kapsamlı notebook'u — `09`'un paralel
ensemble yaklaşımıyla doğrudan karşılaştırma yapıyor.

```
1. Boosting'e Giriş: Sıralı Ensemble ve Zayıf Öğreniciler
2. AdaBoost (Adaptive Boosting)
3. Gradient Boosting Machines (GBM): Fonksiyon Uzayında Gradyan İnişi
4. XGBoost (Extreme Gradient Boosting)
5. LightGBM: Histogram Tabanlı, Yaprak-Öncelikli Büyüme
6. CatBoost: Sıralı Boosting ve Native Kategorik Değişken Desteği
7. Beş Algoritmanın Karşılaştırılması
8. Pratik Hiperparametre Rehberi ve Erken Durdurma (Early Stopping)

🐍 Python: 12 uygulama adımı — AdaBoost'u sentetik veride sıfırdan inşa
   etme, Bank Marketing'te gerçek veri + sızıntı kontrolü + derinlik etkisi,
   GBM'i küçük bir regresyon örneğinde sıfırdan inşa etme, Obesity Levels'ta
   AdaBoost vs GBM çok sınıflı karşılaştırma, XGBoost'ta dengesizlik
   telafisi (`scale_pos_weight`) + erken durdurma + özellik önemi, LightGBM
   hız karşılaştırması + `num_leaves` aşırı öğrenme demosu, CatBoost'ta
   native kategorik destek (`Pool`) vs one-hot karşılaştırması, Student
   Performance'ta beş algoritmalı büyük karşılaştırma tablosu, üç veri seti
   özeti + `09` ile karşılaştırma
📊 R: `adabag::boosting` (Obesity Levels çok sınıflı AdaBoost), `gbm`
   paketi (Student Performance GBM regresyonu, `gbm.perf` CV budama),
   `xgboost` R paketi (Bank Marketing — Python ile aynı C++ çekirdeği),
   `lightgbm` R paketi (Bank Marketing `num_leaves` demosu) — **CatBoost'un R
   paketi CRAN'de yok**, bu yüzden R uygulamasına dahil edilmedi
🗄️ SQL (DuckDB): AdaBoost'un ağırlıklı oy toplamının (`decision_function()`
   ile max fark 0.0), GBM'in ardışık ağaç toplamının (max fark ~1.8e-15)
   yeniden üretimi
✏️ Egzersizler: E1–E6 (Python, R, SQL ve sentez) — **çözülmeden** teslim edildi
```

**📦 Veri Setleri:**

| Dataset | Kaynak | Kullanım |
| --- | --- | --- |
| Bank Marketing | [UCI](https://archive.ics.uci.edu/dataset/222/bank+marketing), Moro/Cortez/Rita (2014) / GitHub mirror `harsh21476/Machine-Learning-on-Bank-Marketing-Dataset` | AdaBoost sızıntı+derinlik, XGBoost dengesizlik+erken durdurma, LightGBM hız (45.211 örnek) |
| Estimation of Obesity Levels | [UCI](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition), Palechor & de la Hoz Manotas (2019) / GitHub mirror `pymche/Machine-Learning-Obesity-Classification` | AdaBoost vs GBM çok sınıflı (2.111 örnek, 7 sınıf) |
| Student Performance | [UCI](https://archive.ics.uci.edu/dataset/320/student+performance), Cortez & Silva (2008) / GitHub mirror `KunjalJethwani/StudentPerformance` | CatBoost native kategorik vs one-hot, beş model karşılaştırması (395 örnek) |

---

## 📚 Kaynak Haritası

### Kitaplar

| # | Kitap | Yazar | Ücretsiz? | Bağlantı |
| --- | --- | --- | --- | --- |
| 🥇 | The Elements of Statistical Learning (Ch. 9-10) | Hastie, Tibshirani, Friedman | ✅ | [hastie.su.domains/ElemStatLearn](https://hastie.su.domains/ElemStatLearn/) |
| 🥇 | An Introduction to Statistical Learning (Ch. 8) | James, Witten, Hastie, Tibshirani | ✅ | [statlearning.com](https://www.statlearning.com/) |
| 🥈 | Classification and Regression Trees | Breiman, Friedman, Olshen, Stone | ❌ | Wadsworth |
| 🥈 | Interpretable Machine Learning (Ch. 5.2) | Molnar | ✅ | [christophmolnar.com/books/interpretable-machine-learning](https://christophmolnar.com/books/interpretable-machine-learning/) |
| 🥉 | Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (Ch. 7) | Géron | ❌ | O'Reilly |

### Anahtar Makaleler

| Makale | Notebook | Konu |
| --- | --- | --- |
| Quinlan (1986), *Induction of Decision Trees* | 08 | ID3 algoritması, Entropi/Bilgi Kazancı kriteri |
| Breiman (1996), *Bagging Predictors* | 09 | Bagging'i tanıtan orijinal makale |
| Breiman (2001), *Random Forests* | 09 | Random Forest, OOB tahmini ve yakınsama kanıtı |
| Geurts, Ernst & Wehenkel (2006), *Extremely Randomized Trees* | 09 | Extra Trees'i tanıtan orijinal makale |
| Freund & Schapire (1997), *A Decision-Theoretic Generalization of On-Line Learning...* | 10 | AdaBoost'un orijinal makalesi |
| Zhu, Zou, Rosset & Hastie (2009), *Multi-class AdaBoost* | 10 | SAMME — çok sınıflı AdaBoost genellemesi |
| Friedman (2001), *Greedy Function Approximation: A Gradient Boosting Machine* | 10 | GBM'in orijinal makalesi |
| Chen & Guestrin (2016), *XGBoost: A Scalable Tree Boosting System* | 10 | XGBoost'un orijinal makalesi (KDD '16) |
| Ke ve ark. (2017), *LightGBM: A Highly Efficient Gradient Boosting Decision Tree* | 10 | GOSS, EFB, histogram algoritması (NeurIPS) |
| Prokhorenkova ve ark. (2018), *CatBoost: unbiased boosting with categorical features* | 10 | Ordered boosting, tahmin kayması (NeurIPS) |

### Resmi Dokümantasyon

| Kaynak | Kullanım |
| --- | --- |
| [scikit-learn — Decision Trees](https://scikit-learn.org/stable/modules/tree.html) | Notebook 08: `DecisionTreeClassifier/Regressor`, `cost_complexity_pruning_path` |
| [scikit-learn — Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html) | Notebook 09-10: Bagging/RF/Extra Trees, AdaBoost/GBM |
| [scikit-learn — Permutation Importance](https://scikit-learn.org/stable/modules/permutation_importance.html) | Notebook 09: MDI'nin kardinalite yanlılığı üzerine tartışma |
| [XGBoost — Introduction to Boosted Trees](https://xgboost.readthedocs.io/en/stable/tutorials/model.html) | Notebook 10 |
| [LightGBM — Features](https://lightgbm.readthedocs.io/en/latest/Features.html) | Notebook 10 |
| [CatBoost — Concepts](https://catboost.ai/docs/en/concepts/algorithm-main-stages) | Notebook 10 |
| [CRAN — `rpart`](https://cran.r-project.org/web/packages/rpart/rpart.pdf) | Notebook 08 R uygulaması |
| [CRAN — `randomForest`](https://cran.r-project.org/web/packages/randomForest/randomForest.pdf) | Notebook 09 R uygulaması |
| [CRAN — `adabag`](https://cran.r-project.org/package=adabag) / [`gbm`](https://cran.r-project.org/package=gbm) / [`xgboost`](https://cran.r-project.org/package=xgboost) / [`lightgbm`](https://cran.r-project.org/package=lightgbm) | Notebook 10 R uygulaması |
| [DuckDB — Aggregate Functions](https://duckdb.org/docs/sql/functions/aggregates) & [Python API](https://duckdb.org/docs/api/python/overview) | Tüm notebook'ların SQL bölümlerinin temeli |

Notebook içi `Kaynaklar` bölümlerinde her dosya için tam kaynak listesi
mevcut (UCI/Kaggle sayfaları, R paket dokümantasyonları, projedeki önceki
notebook'lara çapraz referanslar dahil).

---
