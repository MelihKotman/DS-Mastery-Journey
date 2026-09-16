# 04 · Model Değerlendirme ve İyileştirme (Model Evaluation & Improvement)

`Classical-ML-lab` serisinin dördüncü kategorisi (23 notebook'luk serinin D
bölümü, roadmap'teki notebook'lar 11-15). Önceki kategorilerde (02-10)
modelleri neredeyse hep varsayılan `accuracy`/`.score()` ile "eğitip
değerlendirdik" — bu klasör bu döngüyü kırıyor: önce bir modeli **doğru
ölçmeyi** (`11_model_evaluation_metrics`), sonra dengesiz veride bu ölçümün
neden yanıltıcı olduğunu ve nasıl düzeltileceğini (`12_imbalanced_learning`),
ardından bir modelin en iyi ayarlarını dürüstçe nasıl bulacağımızı
(`13_hyperparameter_optimization`), farklı model ailelerini nasıl
birleştirip ek performans kazanacağımızı (`14_ensemble_stacking_voting`) ve
son olarak bir modelin tahminlerini NEDEN yaptığını nasıl açıklayacağımızı
(`15_model_interpretability`) işliyor. Notebook'lar birbirinin üzerine
inşa edildi — `12`, `11`'deki metrikleri artık *ölçmek* değil *dengesizliği
düzeltmek* için kullanıyor; `13`, `11`-`12`'nin skorlama/`Pipeline`
desenlerini hiperparametre aramasına taşıyor; `14`, model değerlendirme (11)
ve hiperparametre optimizasyonu (13) üzerine kurulu; `15` ise serinin
tamamında (özellikle 07-10 ağaç tabanlı modellerdeki MDI) gördüğümüz
"özellik önemi" kavramını model-agnostik yöntemlerle (permütasyon, SHAP,
LIME) yeniden ziyaret ederek kategori D'yi kapatıyor.

## 📁 Klasör Yapısı

```
04_model_evaluation_improvement/
├── README.md
├── 11_model_evaluation_metrics.ipynb
├── 12_imbalanced_learning.ipynb
├── 13_hyperparameter_optimization.ipynb
├── 14_ensemble_stacking_voting.ipynb
├── 15_model_interpretability.ipynb
└── data/
    ├── creditcard.csv
    ├── mammography.csv
    └── oil-spill.csv
```

> **Hücre kuralı**: Her notebook'ta sırayla Python → R → SQL (DuckDB) uygulaması.
> R hücreleri `%%R` (rpy2) sihirbazıyla çalışıyor. `14` ve `15`'te (proje
> kararı gereği) R ve SQL bölümleri bilinçli olarak kısa/sembolik tutuldu —
> ana teknik derinlik Python tarafında.

---

## 📓 11 · Model Değerlendirme Metrikleri (`11_model_evaluation_metrics.ipynb`)

Accuracy tuzağından başlayıp confusion matrix'ten türeyen tüm ikili
metrikleri (precision, recall, F-beta, MCC, Cohen's Kappa), eşikten bağımsız
ROC/PR eğrilerini, çok sınıflı macro/micro/weighted ortalamaları,
kalibrasyonu ve regresyon hata metriklerini kapsıyor — kategori D'nin temel
taşı, `12`-`15`'in tamamı bu notebook'un metriklerine geri atıf yapıyor.

```
1. Neden Metrik Seçimi Önemli? Accuracy Tuzağı
2. Confusion Matrix ve Temel İkili Sınıflandırma Metrikleri
   (Precision, Recall, Specificity, F1/F-beta, Balanced Accuracy, MCC, Cohen's Kappa)
3. Eşikten Bağımsız Değerlendirme: ROC Eğrisi ve AUC
4. Precision-Recall Eğrisi ve Average Precision
5. Çok Sınıflı (Multi-class) Metrikler: Macro, Micro, Weighted
6. Kalibrasyon: Olasılıklar Gerçekten "Doğru" mu?
7. Regresyon Metrikleri (MAE, RMSE, MAPE, R², vb.)
8. Sentez: Hangi Problemde Hangi Metrik?

🐍 Python: 12 uygulama adımı — German Credit'te confusion matrix + temel
   ikili metrikler (elle + `sklearn`), elle eşik taraması ile ROC/AUC +
   Youden's J, PR eğrisi + AP + dengesizlik arttıkça ROC-AUC vs AP deneyi,
   kalibrasyon eğrisi + Brier skoru (LR vs GaussianNB) + `CalibratedClassifierCV`,
   Ecoli'de çok sınıflı confusion matrix + macro/micro/weighted F1 (elle +
   `sklearn`) + One-vs-Rest ROC/AUC, Boston Housing'de regresyon metrikleri
   + aykırı değer duyarlılığı (MAE vs RMSE) deneyi, üç problemden sentez
📊 R: `pROC`/`caret::confusionMatrix` (German Credit ROC/AUC, Ecoli çok
   sınıflı rapor), `Metrics` paketi (Boston Housing regresyon metrikleri)
🗄️ SQL (DuckDB): confusion matrix'ten precision/recall/F1, kümülatif pencere
   fonksiyonlarıyla ROC eğrisi/AUC yeniden üretimi, regresyon metriklerinin
   toplama fonksiyonlarıyla hesabı — hepsi `sklearn` ile makine hassasiyetinde
   doğrulandı
✏️ Egzersizler: E1–E6 (Python, R, SQL ve sentez) — **çözülerek** teslim edildi
```

**📦 Veri Setleri:**

| Dataset | Kaynak | Kullanım |
| --- | --- | --- |
| Statlog (German Credit Data) | [UCI](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data), Hofmann (1994) | İkili confusion matrix, ROC/PR/AP, kalibrasyon (1.000 kredi, 700/300 dengesiz) |
| Ecoli | Nakai & Kanehisa (1991) | Çok sınıflı macro/micro/weighted F1, One-vs-Rest ROC (336 protein, 8→7 sınıf, aşırı dengesiz) |
| Boston Housing | Harrison & Rubinfeld (1978) | Regresyon metrikleri, aykırı değer duyarlılığı (506 mahalle) |

---

## 📓 12 · Dengesiz Veri Öğrenimi (`12_imbalanced_learning.ipynb`)

`11`'deki metriklerin (özellikle recall/precision/PR-AUC) neden gerekli
olduğunu somutlaştırarak, sınıf dengesizliğini düzeltmenin tüm ana ailesini
— rastgele/akıllı yeniden örnekleme, SMOTE ailesi, hibrit yöntemler, sınıf
ağırlıklandırma, karar eşiği taşıma ve dengesizliğe duyarlı ensemble'ları —
işliyor; en kritik metodolojik tuzağı (yeniden örnekleme + veri sızıntısı)
somut sayılarla gösteriyor.

```
1. Dengesiz Veri Problemi ve "Doğruluk Paradoksu"
2. Çözüm Ailelerinin Taksonomisi
3. Rastgele Yeniden Örnekleme (Random Over/Undersampling)
4. SMOTE Ailesi — Sentetik Azınlık Örneği Üretimi (SMOTE / Borderline-SMOTE / ADASYN)
5. Akıllı Undersampling Ailesi (Tomek Links / ENN / NearMiss)
6. Hibrit (Kombine) Yöntemler (SMOTEENN / SMOTETomek)
7. Sınıf Ağırlıklandırma ve Maliyet-Duyarlı Öğrenme
8. Karar Eşiği Taşıma (Threshold Moving)
9. Dengesizliğe Duyarlı Ensemble Yöntemleri (Balanced RF / EasyEnsemble / RUSBoost)
10. KRİTİK Metodolojik Tuzak: Yeniden Örnekleme ve Veri Sızıntısı

🐍 Python: 13 uygulama adımı — üç veri setinin yüklenmesi + Oil Spill
   temizleme (ID + sıfır-varyanslı sütun), Mammography'de temel model +
   doğruluk paradoksu, rastgele over/undersampling, SMOTE/Borderline-SMOTE/
   ADASYN (sentetik 2B görselleştirme + gerçek veri), Oil Spill'de Tomek/
   ENN/NearMiss (PCA görselleştirmeli), Mammography'de SMOTEENN/SMOTETomek,
   Credit Card Fraud'da (IR≈1:578) sınıf ağırlıklandırma + maliyet-duyarlı
   öğrenme (`class_weight`, `scale_pos_weight`/`is_unbalance`), eşik taşıma
   ($F_2$ maksimizasyonu), Mammography'de BalancedRF/EasyEnsemble/RUSBoost,
   Oil Spill'de YANLIŞ vs DOĞRU CV (imblearn `Pipeline`) karşılaştırması,
   Credit Card Fraud'da büyük karşılaştırma tablosu, sentez karar rehberi
📊 R: `ROSE::ovun.sample`/`ROSE`, `smotefamily::SMOTE`/`ADAS`,
   `unbalanced::ubTomek`/`ubENN`/`ubNCL`/`ubSMOTE`, sınıf ağırlıklandırma
   (`randomForest(classwt=...)`, `glm(weights=...)`, R `xgboost`),
   `pROC::coords` (eşik taşıma), native `randomForest(sampsize=...,
   strata=...)` + `ebmc::rus`, `caret::trainControl(sampling=...)` (doğru CV)
🗄️ SQL (DuckDB): confusion matrix'ten dengesizlik metrikleri, sınıf-ağırlıklı
   lojistik regresyonun karar fonksiyonu, maliyet-duyarlı eşik seçimi
✏️ Egzersizler: E1–E6 (Python, R, SQL ve sentez) — **çözülerek** teslim
   edildi; ayrıca G-Mean (geometrik ortalama) metriğini tanıtan ek bir
   teorik bölüm içeriyor
```

**📦 Veri Setleri:**

| Dataset | Kaynak | Kullanım |
| --- | --- | --- |
| Credit Card Fraud Detection | [Kaggle (mlg-ulb)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), Dal Pozzolo ve ark. (2015) | Sınıf ağırlıklandırma, eşik taşıma, büyük karşılaştırma tablosu (~285.000 işlem, IR≈1:578) |
| Oil Spill | Kubat, Holte & Matwin (1998), UCI | Akıllı undersampling görselleştirmesi, YANLIŞ vs DOĞRU CV deneyi (937 örnek) |
| Mammography | Woods ve ark. (1993), UCI | Doğruluk paradoksu, SMOTE ailesi, hibrit yöntemler, dengesizliğe duyarlı ensemble (11.183 örnek) |

---

## 📓 13 · Hiperparametre Optimizasyonu (`13_hyperparameter_optimization.ipynb`)

`11`-`12`'nin skorlama/`Pipeline` desenlerini temel alıp, parametre ile
hiperparametre ayrımından başlayarak Çapraz Doğrulama ailesinin tamamını
(K-Fold, Stratified, Repeated, LOOCV, ShuffleSplit, Group K-Fold, Time
Series Split), Grid/Random/Successive-Halving/Bayesian aramayı,
`Pipeline`+`ColumnTransformer`'ı ve — en kritik olarak — Nested CV'nin
neden gerekli olduğunu işliyor.

```
1. Parametre mi, Hiperparametre mi?
2. Çapraz Doğrulama (Cross-Validation) — K-Fold ailesi, özel CV stratejileri,
   `learning_curve`/`validation_curve`, k seçimi
3. Grid Search (Izgara Arama)
4. Random Search (Rastgele Arama)
5. Successive Halving (Ardışık Yarılama)
6. Bayesian Optimization (Bayesçi Optimizasyon)
7. Pipeline ve ColumnTransformer
8. Nested Cross-Validation
9. Skorlama Metriği Seçimi ve `refit` Stratejisi

🐍 Python: 15 uygulama adımı — Haberman'da baseline model + CV stratejileri
   karşılaştırması (katlama görselleştirmesi, `cross_validate`, LOOCV vs
   5/10-Fold, `learning_curve`, `validation_curve`), Grid Search (SVM,
   C×γ×çekirdek) ve Random Search karşılaştırması, Horse Colic'te sızıntı
   riskli sütunları temizleme + `Pipeline`+`ColumnTransformer` + Grid Search
   ile model-tipi-arası arama (LR vs RF) + Successive Halving + naive vs
   Nested CV karşılaştırması, Wage'de regresyon `ColumnTransformer` + Random
   Search + `optuna`/TPE Bayesian Optimization (aynı bütçede karşılaştırma) +
   Nested CV, tüm sonuçların sentez tablosu ve karar rehberi
📊 R: `tidymodels` ekosistemi — `recipe`+`workflow` (Grid/Random Search,
   Haberman), model-tipi-arası arama (`workflowsets`, Horse Colic),
   `rsample::nested_cv` (Horse Colic, Wage), `tune_bayes` (Wage, Gauss
   Süreci), `rsample` CV stratejileri ailesi
🗄️ SQL (DuckDB): Grid Search'ün kendisinin `GROUP BY`+`ORDER BY` ile yeniden
   üretimi, pencere fonksiyonuyla arama ilerlemesinin ("şu ana kadarki en
   iyi") izlenmesi, tuned lojistik regresyonun karar fonksiyonu, CV
   katlamalarının SQL'de yeniden üretilmesi
✏️ Egzersizler: E1–E8 (Python, R, SQL ve sentez) — **çözülerek** teslim edildi
```

**📦 Veri Setleri:**

| Dataset | Kaynak | Kullanım |
| --- | --- | --- |
| Haberman's Survival | [UCI](https://archive.ics.uci.edu/dataset/43/haberman+s+survival), Haberman (1976) | Baseline model, CV stratejileri, Grid/Random Search (SVM) — 306 örnek, 3 öznitelik |
| Horse Colic | [UCI](https://archive.ics.uci.edu/dataset/47/horse+colic), McLeish & Cecile | Pipeline/ColumnTransformer, model-tipi-arası arama, Successive Halving, Nested CV — eksik veri + sızıntı riskli sütunlar |
| Wage | James, Witten, Hastie & Tibshirani, *ISLR* eşlik veri seti | Regresyon ColumnTransformer, Random Search vs Bayesian Optimization (`optuna`), Nested CV — 3.000 işçi |

---

## 📓 14 · Ensemble Öğrenme: Stacking, Voting ve Blending (`14_ensemble_stacking_voting.ipynb`)

Bagging (09) ve boosting (10) *aynı tip* zayıf öğreniciyi birleştirirken, bu
notebook **farklı tip** modelleri (LR, k-NN, SVM, ağaç, Naive Bayes) tek bir
tahminde birleştiren Voting, Stacking ve Blending'i işliyor — `11`'deki
metrikleri ve `13`'teki `cross_val_predict`/CV disiplinini kullanarak
Stacking'in en kritik tuzağı olan veri sızıntısını hem elle hem
`StackingClassifier`/`Regressor` ile gösteriyor. R ve SQL bölümleri proje
kararı gereği (2026-09-08) bilinçli olarak kısa tutuldu.

```
1. Ensemble Öğrenmeye Genel Bakış: Homojen mi, Heterojen mi?
2. Voting: Hard ve Soft Oylama
3. Stacking (Yığınlı Genelleme): Meta-Öğrenici ve Sızıntı Riski
4. Blending: Basitleştirilmiş Alternatif
5. Model Çeşitliliği (Diversity) ve Hata Korelasyonu
6. Pratik Tuzaklar, Hesaplama Maliyeti ve Ne Zaman Kullanılmamalı

🐍 Python: 12 uygulama adımı — Phoneme'de beş taban modelin (LR, k-NN, SVM,
   Karar Ağacı, GaussianNB) tek tek değerlendirilmesi, Hard vs Soft Voting,
   CV-ROC-AUC'ye göre ağırlıklı oylama, çeşitlilik (korelasyon/hata
   çakışması) analizi, CreditCard'da klasik sızıntı tuzağının (`expenditure`,
   `share`) tespiti + manuel out-of-fold stacking (naif vs doğru) +
   `StackingClassifier` karşılaştırması, Beijing PM2.5'te `VotingRegressor` +
   `StackingRegressor` vs Blending karşılaştırması, üç problemden sentez ve
   karar rehberi
📊 R: `caret`+`caretEnsemble` (Phoneme stacking), paket olmadan basit soft
   voting — **bilinçli olarak kısa tutuldu**
🗄️ SQL (DuckDB): soft voting'in `AVG` ile, ağırlıklı oylamanın `JOIN`+
   `GROUP BY` ile yeniden üretimi — **bilinçli olarak kısa tutuldu**
✏️ Egzersizler: E1–E6 (Python, R, SQL ve sentez) — **çözülerek** teslim edildi
```

**📦 Veri Setleri:**

| Dataset | Kaynak | Kullanım |
| --- | --- | --- |
| Phoneme | ELENA Projesi (ESPRIT-5516 ROARS) / [OpenML #1489](https://www.openml.org/d/1489) | Taban model karşılaştırması, Hard/Soft/Ağırlıklı Voting, çeşitlilik analizi (5.404 örnek, 5 öznitelik) |
| CreditCard (AER) | Greene, *Econometric Analysis* / Kleiber & Zeileis, `AER::CreditCard` | Manuel OOF Stacking, `StackingClassifier`, klasik sızıntı tuzağı (1.319 başvuru, 12 değişken) |
| Beijing PM2.5 | Liang ve ark. (2015), UCI | `VotingRegressor`, `StackingRegressor` vs Blending (hava kirliliği regresyonu) |

---

## 📓 15 · Model Yorumlanabilirliği (`15_model_interpretability.ipynb`)

Kategori D'nin son notebook'u. `11`'de modelin "ne kadar iyi" tahmin
ettiğini ölçmüştük; burada modelin **NEDEN** o tahmini yaptığını açıklıyoruz
— model-agnostik (Permutation Importance, PDP/ICE, LIME, SHAP
KernelExplainer) ve model-spesifik (SHAP TreeExplainer/LinearExplainer)
yöntemleri, global/local ayrımını ve korelasyonlu özniteliklerin bu
yöntemleri nasıl yanıltabileceğini (07-10'daki MDI tartışmasının devamı
olarak) işliyor.

```
1. Yorumlanabilirlik Neden Önemli?
2. Taksonomi: Model-Spesifik / Model-Agnostik, Global / Local
3. Permutation Feature Importance
4. Partial Dependence Plot (PDP) ve Individual Conditional Expectation (ICE)
5. LIME (Local Interpretable Model-agnostic Explanations)
6. SHAP (SHapley Additive exPlanations)
7. Yöntem Karşılaştırması ve Ortak Tuzaklar

🐍 Python: 13 uygulama adımı — üç veri setinin (Default, Hitters, Mroz)
   yüklenmesi + baseline modeller, model-içi önem ölçütlerinin (katsayılar,
   MDI) hatırlatılması, Permutation Feature Importance (train vs test
   skoru), 1D PDP (`balance`/`income`, `CRuns`/`Years`), 2D PDP + ICE
   (`balance`×`student` etkileşimi ve heterojenlik), LIME ile yerel açıklama
   (Default + Mroz), SHAP TreeExplainer (Hitters, beeswarm + waterfall),
   SHAP LinearExplainer vs KernelExplainer (Default/Mroz, kesin vs yaklaşık
   karşılaştırması), SHAP dependence plot (`CRuns`×`Years` etkileşimi),
   MDI vs Permutation vs SHAP karşılaştırması (Spearman korelasyonu),
   korelasyonlu öznitelik tuzağı deneyi (`CRuns`'u çıkarma), global vs local
   özet ve yöntem seçim rehberi
📊 R: `iml` (permutation importance) ve `pdp` (partial dependence) —
   **bilinçli olarak kısa/sembolik tutuldu** (proje kararı)
🗄️ SQL (DuckDB): lojistik regresyonun doğrusal yapısından PDP'nin kapalı-
   formunun (`balance` için $\sigma(\beta_0+\beta\cdot x)$) tek bir odaklı
   örnekle yeniden üretimi — **bilinçli olarak kısa tutuldu**
✏️ Egzersizler: E1–E6 (Python, R, SQL ve sentez) — **çözülmeden** teslim edildi
```

**📦 Veri Setleri:**

| Dataset | Kaynak | Kullanım |
| --- | --- | --- |
| Default | James, Witten, Hastie & Tibshirani, *ISLR* paketi | LR+RF baseline, PDP/ICE, LIME, SHAP LinearExplainer (10.000 müşteri, %3.3 temerrüt — dengesiz) |
| Hitters | *ISLR* paketi (orijinal: 1986-87 MLB sezonu, StatLib/ASA) | RF Regressor, Permutation Importance, SHAP TreeExplainer, korelasyonlu öznitelik deneyi (322 oyuncu, 59 eksik satır çıkarıldı) |
| Mroz | Mroz, T.A. (1987), *Econometrica* / `carData` R paketi | LR+RF, LIME (kategorik+sayısal karışık), SHAP KernelExplainer (753 evli kadın) |

---

## 📚 Kaynak Haritası

### Kitaplar

| # | Kitap | Yazar | Ücretsiz? | Bağlantı |
| --- | --- | --- | --- | --- |
| 🥇 | An Introduction to Statistical Learning | James, Witten, Hastie, Tibshirani | ✅ | [statlearning.com](https://www.statlearning.com/) |
| 🥇 | Interpretable Machine Learning | Molnar | ✅ | [christophmolnar.com/books/interpretable-machine-learning](https://christophmolnar.com/books/interpretable-machine-learning/) |
| 🥈 | The Elements of Statistical Learning | Hastie, Tibshirani, Friedman | ✅ | [hastie.su.domains/ElemStatLearn](https://hastie.su.domains/ElemStatLearn/) |
| 🥈 | Automated Machine Learning: Methods, Systems, Challenges | Hutter, Kotthoff, Vanschoren (Ed.) | ✅ | [automl.org/book](https://www.automl.org/book/) |
| 🥉 | Imbalanced Learning: Foundations, Algorithms, and Applications | He & Ma (Ed.) | ❌ | Wiley-IEEE Press |
| 🥉 | Ensemble Methods: Foundations and Algorithms | Zhou | ❌ | CRC Press |

### Anahtar Makaleler

| Makale | Notebook | Konu |
| --- | --- | --- |
| Fawcett (2006), *An Introduction to ROC Analysis* | 11 | ROC/AUC analizinin kapsamlı referansı |
| Davis & Goadrich (2006), *The Relationship Between Precision-Recall and ROC Curves* | 11 | PR eğrisinin dengesiz verilerde neden daha bilgilendirici olduğu |
| Niculescu-Mizil & Caruana (2005), *Predicting Good Probabilities With Supervised Learning* | 11 | Model ailelerinin kalibrasyon davranışı |
| Chawla ve ark. (2002), *SMOTE: Synthetic Minority Over-sampling Technique* | 12 | SMOTE'un orijinal makalesi (40.000+ atıf) |
| Liu, Wu & Zhou (2009), *Exploratory Undersampling for Class-Imbalance Learning* | 12 | EasyEnsemble/BalanceCascade'in orijinal makalesi |
| Bergstra & Bengio (2012), *Random Search for Hyper-Parameter Optimization* | 13 | Random Search'ün teorik gerekçesi |
| Snoek, Larochelle & Adams (2012), *Practical Bayesian Optimization of Machine Learning Algorithms* | 13 | Gauss Süreci tabanlı Bayesian optimizasyonun klasik makalesi |
| Cawley & Talbot (2010), *On Over-fitting in Model Selection and Subsequent Selection Bias* | 13 | Nested CV'nin gerekliliğinin kanıtı |
| Akiba ve ark. (2019), *Optuna: A Next-generation Hyperparameter Optimization Framework* | 13 | TPE örnekleyicisi ve define-by-run API |
| Wolpert (1992), *Stacked Generalization* | 14 | Stacking'i tanımlayan temel makale |
| Lundberg & Lee (2017), *A Unified Approach to Interpreting Model Predictions* | 15 | SHAP'ın orijinal makalesi (NeurIPS) |
| Ribeiro, Singh & Guestrin (2016), *"Why Should I Trust You?"* | 15 | LIME'ın orijinal makalesi (KDD) |

### Resmi Dokümantasyon

| Kaynak | Kullanım |
| --- | --- |
| [scikit-learn — Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html) & [Probability Calibration](https://scikit-learn.org/stable/modules/calibration.html) | Notebook 11 |
| [imbalanced-learn — Resmi Dokümantasyon](https://imbalanced-learn.org/stable/) & [Common Pitfalls](https://imbalanced-learn.org/stable/common_pitfalls.html) | Notebook 12 |
| [scikit-learn — Tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html) & [Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) | Notebook 13 |
| [Optuna — Resmi Dokümantasyon](https://optuna.readthedocs.io/) | Notebook 13 |
| [scikit-learn — Ensemble methods (Voting/Stacking)](https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier) | Notebook 14 |
| [scikit-learn — Permutation Importance](https://scikit-learn.org/stable/modules/permutation_importance.html) & [Partial Dependence and ICE Plots](https://scikit-learn.org/stable/modules/partial_dependence.html) | Notebook 15 |
| [SHAP — resmi GitHub deposu ve dokümantasyon](https://github.com/shap/shap) | Notebook 15 |
| [LIME — resmi GitHub deposu](https://github.com/marcotcr/lime) | Notebook 15 |
| [CRAN — `pROC`](https://cran.r-project.org/package=pROC) / [`caret`](https://cran.r-project.org/package=caret) / [`Metrics`](https://cran.r-project.org/package=Metrics) | Notebook 11 R uygulaması |
| [CRAN — `ROSE`](https://cran.r-project.org/package=ROSE) / [`smotefamily`](https://cran.r-project.org/package=smotefamily) / [`unbalanced`](https://cran.r-project.org/package=unbalanced) / [`ebmc`](https://cran.r-project.org/package=ebmc) | Notebook 12 R uygulaması |
| [`tidymodels` — Resmi Dokümantasyonu](https://www.tidymodels.org/) (`recipes`, `parsnip`, `workflows`, `rsample`, `dials`, `tune`) | Notebook 13 R uygulaması |
| [CRAN — `caretEnsemble`](https://cran.r-project.org/package=caretEnsemble) | Notebook 14 R uygulaması |
| [CRAN — `iml`](https://cran.r-project.org/package=iml) / [`pdp`](https://cran.r-project.org/package=pdp) | Notebook 15 R uygulaması |
| [DuckDB — SQL Fonksiyonları](https://duckdb.org/docs/sql/functions/numeric) & [Window Functions](https://duckdb.org/docs/sql/window_functions) | Tüm notebook'ların SQL bölümlerinin temeli |

Notebook içi `Kaynaklar` bölümlerinde her dosya için tam kaynak listesi
mevcut (UCI/Kaggle sayfaları, R paket dokümantasyonları, projedeki önceki
notebook'lara çapraz referanslar dahil).

---
