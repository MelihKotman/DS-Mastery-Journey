# Classical-ML-lab

`DS-Mastery-Journey` projesinin klasik makine öğrenmesi laboratuvarı: **23
notebook**, 6 kategori. Amaç yalnızca "modeli çalıştırmak" değil; her
algoritmayı önce teoriyle ve elle hesaplanabilir küçük bir örnekle anlamak,
çekirdeğini **sıfırdan** kodlayıp `sklearn`/`scipy` ile doğrulamak, sonra
gerçek veri setlerinde metodolojik tuzaklarıyla (veri sızıntısı, yanlış
metrik, yanlış CV, yanlış bölme) birlikte görmek. Notebook'lar hem öğrenme
hem de proje ve iş hayatında başvurulacak bir kaynak olacak şekilde, endüstri
örnekleriyle ve yalnızca bilinen/güvenilir kaynaklara (makale, kitap, resmi
dokümantasyon, GitHub deposu) dayanarak yazıldı.

## 🗺️ Öğrenme Yolu

```
A  Veri Hazırlığı ─▶ B  Denetimli Temel Aileler ─▶ C  Ağaç Tabanlı Yöntemler
                                                          │
F  Olasılıksal & Özel ◀── E  Denetimsiz Öğrenme ◀── D  Değerlendirme & İyileştirme
```

| Kategori | Klasör | Notebook'lar | Ana soru |
| --- | --- | --- | --- |
| A · Veri Hazırlığı | [`01_data_prepreparation`](01_data_prepreparation/README.md) | 01-02 | Veriyi nasıl anlar ve modele hazırlarız? |
| B · Denetimli Öğrenme — Temel Aileler | [`02_supervised_learning`](02_supervised_learning/README.md) | 03-07 | Doğrusal, komşuluk, olasılıksal ve marj tabanlı modeller nasıl çalışır? |
| C · Ağaç Tabanlı Yöntemler | [`03_tree_based_models`](03_tree_based_models/README.md) | 08-10 | Tek ağaçtan bagging ve boosting'e: topluluk yöntemleri neden güçlü? |
| D · Model Değerlendirme ve İyileştirme | [`04_model_evaluation_improvement`](04_model_evaluation_improvement/README.md) | 11-15 | Modeli doğru nasıl ölçer, ayarlar, birleştirir ve açıklarız? |
| E · Denetimsiz Öğrenme | [`05_unsupervised_learning`](05_unsupervised_learning/README.md) | 16-20 | Etiket yokken yapıyı, anomaliyi ve birlikteliği nasıl buluruz? |
| F · Olasılıksal ve Özel Modeller | [`06_probability_special_models`](06_probability_special_models/README.md) | 21-23 | Belirsizlik, zaman ve kullanıcı × öğe verisi nasıl modellenir? |

> Dosya adındaki numara **tüm lab boyunca** artar (01-23); klasör numarası
> ise kategoriyi gösterir. Bu yüzden `02_supervised_learning/` içinde
> `03_...` ile `07_...` notebook'ları bulunur.

## 📓 Notebook Dizini

| # | Notebook | Konu |
| --- | --- | --- |
| 01 | `01_eda_visualization` | Keşifsel veri analizi ve görselleştirme |
| 02 | `02_data_preprocessing` | Eksik veri, aykırı değer, kodlama, ölçekleme, özellik mühendisliği |
| 03 | `03_linear_classification_family` | Lojistik regresyon, SGD, Perceptron, LDA/QDA |
| 04 | `04_linear_regression_family` | OLS, Ridge/Lasso/ElasticNet, GLM'ler, robust regresyon |
| 05 | `05_knn_instance_based` | k-NN ve örnek tabanlı öğrenme |
| 06 | `06_naive_bayes_family` | Gaussian / Multinomial / Bernoulli / Complement NB |
| 07 | `07_svm_kernels` | SVM, soft margin, kernel trick |
| 08 | `08_decision_trees` | CART, Gini/entropi, budama |
| 09 | `09_bagging_random_forest` | Bootstrap, Bagging, Random Forest, Extra Trees |
| 10 | `10_gradient_boosting_family` | AdaBoost, GBM, XGBoost, LightGBM, CatBoost |
| 11 | `11_model_evaluation_metrics` | Sınıflandırma/regresyon metrikleri, ROC/PR, kalibrasyon |
| 12 | `12_imbalanced_learning` | SMOTE ailesi, sınıf ağırlığı, eşik taşıma, yeniden örnekleme sızıntısı |
| 13 | `13_hyperparameter_optimization` | CV stratejileri, Grid/Random/Bayesian arama, Nested CV |
| 14 | `14_ensemble_stacking_voting` | Voting, Stacking, Blending |
| 15 | `15_model_interpretability` | Permutation importance, PDP/ICE, LIME, SHAP |
| 16 | `16_clustering_partitioning` | K-Means, k-means++, K-Medoids, Mini-Batch |
| 17 | `17_clustering_hierarchical_density` | Hiyerarşik, DBSCAN/OPTICS/HDBSCAN, GMM |
| 18 | `18_dimensionality_reduction_advanced` | LDA, Kernel PCA, NMF, ICA, t-SNE, UMAP |
| 19 | `19_anomaly_detection` | Isolation Forest, LOF, One-Class SVM, MCD |
| 20 | `20_association_rule_learning` | Apriori, Eclat, FP-Growth |
| 21 | `21_bayesian_probabilistic_models` | Bayesçi regresyon, MCMC, Gaussian Process |
| 22 | `22_time_series_ml` | ARIMA/SARIMA, ETS, ML ile forecasting |
| 23 | `23_recommendation_systems_basics` | Bias modeli, item-kNN, ALS, HKV, soğuk başlangıç |

## 🧱 Her Notebook'un Yapısı

1. **Başlık ve kapsam** — "Bu notebook'ta ne öğreneceğiz?", önkoşullar, veri, araçlar.
2. **Teori** — formüller, elle hesaplanan somut örnekler (🔍), endüstri örnekleri.
3. **🐍 Python** — adım adım uygulama; çekirdek algoritma sıfırdan yazılıp kütüphane ile doğrulanır.
4. **📊 R** — `%%R` (rpy2) ile konunun standart R paketi.
5. **🗄️ SQL (DuckDB)** — modelin çekirdek mantığının SQL'de yeniden üretimi.
6. **✏️ Egzersizler** — E1–E6.
7. **📚 Kaynaklar** — makaleler, kitaplar, resmi dokümantasyon, veri seti sayfaları.

R ve SQL bölümleri `14`'ten itibaren bilinçli olarak kısa tutuldu; ana
derinlik Python tarafında. Egzersizler `09`'dan itibaren çoğunlukla çözülmeden
(boş hücre / `# TODO`) teslim edildi; bazı notebook'larda kendi çözümlerin de
bulunuyor. Her kategori klasörünün `README.md` dosyası ilgili notebook'ların
bölüm başlıklarını, veri setlerini ve kaynak haritasını içerir.

## 🧰 Çalıştırma Notları

- Çekirdek: `python3` (kernelspec `base`), nbformat 4.5.
- Python tarafı: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `statsmodels`, `duckdb`; kategoriye göre `imbalanced-learn`, `optuna`, `shap`, `lime`, `xgboost`, `lightgbm`, `catboost`, `umap-learn`, `mlxtend`, `scikit-learn-extra`.
- R hücreleri için `rpy2` ve ilgili CRAN paketleri gerekir (paket listesi her kategori README'sinin kaynak haritasında); R hücreleri çıktısız teslim edilir, kendi makinende çalıştırılır.
- Veri dosyaları her kategori klasörünün `data/` altında; notebook'lar bu yerel kopyaları okur.

## 🔗 İlgili Klasörler

`Linear-Algebra-lab`, `Calculus-lab`, `Probability-lab` ve `Statistic-Inference-lab`
bu lab'ın matematiksel ön hazırlığıdır (özellikle PCA/SVD, olasılık ve
çıkarım); bu lab'daki notebook'lar gerektiğinde onlara atıf yapar.
