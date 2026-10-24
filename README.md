# DS-Mastery-Journey

> Veri bilimi ve makine öğrenmesini **matematiksel temelden başlayarak**, gerçek veri setleri ve
> üç farklı araçla (Python · R · SQL) öğrenmek için kurduğum, yapılandırılmış ve kendi kendine
> ilerleyen bir müfredat. Hedef: yöntemleri "kullanmayı bilmek" değil, **neden çalıştıklarını
> anlayıp** proje ve iş hayatında güvenle başvurulabilecek bir referans kaynağı üretmek.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![R](https://img.shields.io/badge/R-276DC3?style=for-the-badge&logo=r&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-336791?style=for-the-badge&logo=postgresql&logoColor=white)
![DuckDB](https://img.shields.io/badge/DuckDB-FFF000?style=for-the-badge&logo=duckdb&logoColor=black)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-4051B5?style=for-the-badge&logo=python&logoColor=white)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![SymPy](https://img.shields.io/badge/SymPy-3B5526?style=for-the-badge&logo=sympy&logoColor=white)
![scikit--image](https://img.shields.io/badge/scikit--image-EC6B23?style=for-the-badge&logo=scikit-image&logoColor=white)

---

## 📌 Bu Depo Nedir?

Depo, her biri belirli bir konu alanını kapsayan **"lab"**lardan oluşuyor. Her lab, aynı kalıbı
izleyen Jupyter notebook'larından meydana geliyor:

1. **Teori** — formüller yalnızca yazılmaz; küçük sayısal örneklerle elle türetilir.
2. **Uygulama** — gerçek veri setleri üzerinde Python'la (ve konuya uygunsa R ve SQL ile) aynı analiz yapılır.
3. **Egzersizler** — okuyucunun kendisinin çözmesi için bırakılmış sorular.
4. **Kaynaklar** — kitap, makale, resmi dokümantasyon ve GitHub depoları.

Üç şeyi özellikle önemsiyorum:

- **Önce matematik, sonra model.** Faz 1'de olasılık, istatistik, lineer cebir ve kalkülüs;
  Faz 2'de bunların üzerine kurulu klasik makine öğrenmesi. Örneğin PCA'yı SVD'den,
  lojistik regresyonun konveksliğini Hessian'dan, Ridge/Lasso farkını norm geometrisinden türetiyorum.
- **Doğrulanmış iddialar.** Elle hesaplanan sonuç ile kütüphane çıktısının örtüştüğü, iki
  yöntemin aynı cevabı verdiği gibi iddialar notebook içinde kod çalıştırılarak gösterilir.
- **Metodolojik tuzaklar dahil.** Veri sızıntısı, yanlış metrik, hatalı çapraz doğrulama gibi
  gerçek projelerde asıl can yakan hatalar, örnekleriyle birlikte işlenir.

Kaynakların tamamı bilinen ve güvenilir olanlardan seçilir (ders kitapları, hakemli makaleler,
resmi dokümantasyon, saygın GitHub depoları).

---

## 🗺️ Yol Haritası

```
FAZ 1 — Matematiksel Temeller                           FAZ 2 — Klasik Makine Öğrenmesi
Probability ─▶ Statistic-Inference ─▶ Linear-Algebra ─▶ Calculus  ═▶  Classical-ML-lab (23 notebook)
 (17 nb)           (6 nb)                (5 nb)          (5 nb)
```

| Faz | Lab                                                             | Notebook |     Durum     | Dil(ler)           | Ana soru                                                                   |
| --- | --------------------------------------------------------------- | :------: | :------------: | ------------------ | -------------------------------------------------------------------------- |
| 1   | [`Probability-lab`](Probability-lab/README.md)                 |    17    | ✅ Tamamlandı | Python · R · SQL | Belirsizliği ve rastgeleliği nasıl modelleriz?                          |
| 1   | [`Statistic-Inference-lab`](Statistic-Inference-lab/README.md) |    6    | ✅ Tamamlandı | Python · R        | Örneklemden popülasyon hakkında nasıl güvenilir sonuç çıkarırız? |
| 1   | [`Linear-Algebra-lab`](Linear-Algebra-lab/README.md)           |    5    | ✅ Tamamlandı | Python             | ML algoritmalarının geometrik ve cebirsel omurgası nedir?               |
| 1   | [`Calculus-lab`](Calculus-lab/README.md)                       |    5    | ✅ Tamamlandı | Python             | Modeller nasıl "öğrenir"? Gradyan, backprop ve optimizasyon             |
| 2   | [`Classical-ML-lab`](Classical-ML-lab/README.md)               |    23    | ✅ Tamamlandı | Python · R · SQL | Klasik ML'in tüm hattı: veri hazırlığından öneri sistemlerine       |

Toplam: **56 notebook**, 5 lab.

---

## 🧱 Faz 1 — Matematiksel Temeller

### Probability-lab · 17 notebook

Olasılık teorisinin temellerinden Bayes çıkarımına, parametre tahmininden bilgi teorisine ve
simülasyona uzanan seri.

- **Temeller ve dağılımlar:** PMF / PDF / CDF; Bernoulli, Binomial, Poisson; Normal, Uniform,
  Exponential, Gamma, Chi-Square, Beta, Log-Normal
- **Çok değişkenli olasılık:** ortak / marjinal / koşullu dağılımlar, kovaryans ve korelasyon
- **Beklenti ve varyans** ile özellikleri
- **Limit teoremleri:** Büyük Sayılar Yasası, Merkezi Limit Teoremi
- **Bayes:** Bayes teoremi, eşlenik (conjugate) priorlar
- **Tahmin:** MLE ve MAP
- **Bilgi teorisi:** entropi, çapraz entropi, KL diverjansı
- **Simülasyon:** Monte Carlo, bootstrap

Her notebook Python, R ve SQL (DuckDB) uygulamalarını birlikte içerir; aynı analiz üç araçla
tekrarlanıp sonuçların birbirini doğrulaması sağlanır. DuckDB, büyük dosyaları (bazı
örneklerde 100 MB üzeri) pandas'a yüklemeden doğrudan sorgulamak için kullanılır.

### Statistic-Inference-lab · 6 notebook

Klasik frekantist istatistiğin araç kutusu: Probability-lab'ın bıraktığı yerden devam eder.

- **Hipotez testleri:** tek örneklem, iki örneklem ve eşleştirilmiş t-testleri
- **Güç analizi** ve örneklem büyüklüğü hesabı
- **Regresyon tanılaması:** OLS varsayımları, çoklu doğrusal bağlantı, değişen varyans, artık analizi
- **Parametrik olmayan testler:** Mann-Whitney U, Wilcoxon, Kruskal-Wallis
- **Çoklu test düzeltmesi:** Bonferroni, Benjamini-Hochberg (FDR)
- **ANOVA:** tek yönlü, iki yönlü ve post-hoc karşılaştırmalar

Python ve R birlikte kullanılır; konular SQL'e doğal olarak oturmadığı için SQL yoktur.

### Linear-Algebra-lab · 5 notebook

ML'in matematiksel omurgasını soyut örneklerle ve gerçek verilerle (finansal getiriler,
ikinci el araç fiyatları, hava kalitesi, görüntüler vb.) işler.

- **Vektör ve matris temelleri:** matris çarpımının geometrik dönüşüm olarak yorumu, determinant, ters matris
- **Özdeğer ve özvektörler:** karakteristik denklem, geometrik yorum, kovaryans matrisi bağlantısı
- **SVD ve PCA:** sıfırdan kurulum, görüntü sıkıştırma, boyut indirgeme
- **Normlar ve regularizasyon:** L1/L2 geometrisi; Ridge ve Lasso'nun neden farklı davrandığı
- **Özel matrisler:** pozitif tanımlılık, kovaryans matrisinin neden her zaman pozitif yarı tanımlı olduğu, konvekslik bağlantısı

### Calculus-lab · 5 notebook

Faz 1'in son halkası: modellerin parametrelerini nasıl güncellediğinin matematiği.

- **Türevler ve gradyan vektörü:** analitik (SymPy) ve sayısal (sonlu farklar) türev; gerçek zaman serisinde değişim hızı ve ivmelenme
- **Jacobian, Hessian ve konvekslik:** Hessian özdeğerleriyle konvekslik testi; lojistik regresyon kayıp yüzeyinin konveksliği; konveks olmayan yüzeylerde yerel minimum tuzağı
- **Zincir kuralı ve backpropagation:** tek nöronda elle türetme, NumPy ile sıfırdan mini ağ ve PyTorch `autograd` ile gradyan karşılaştırması
- **Gradient descent ve optimizer'lar:** öğrenme oranının etkisi, SGD / Momentum / Adam karşılaştırması, Rosenbrock gibi zorlu yüzeyler
- **Taylor serisi:** polinom yaklaşıklama, Newton yöntemi ve ikinci dereceden optimizasyon

Bu lab Python ile ilerler; SymPy sembolik doğrulama, PyTorch ise otomatik türev için kullanılır.

---

## 🤖 Faz 2 — Classical-ML-lab · 23 notebook

Klasik makine öğrenmesinin tüm hattını altı kategoride işler. Her algoritma önce teoriyle ve
elle hesaplanabilir küçük bir örnekle anlaşılır, çekirdeği **sıfırdan** kodlanıp
`sklearn` / `scipy` ile doğrulanır, sonra gerçek veri setlerinde metodolojik tuzaklarıyla görülür.

| Kategori                                       | Notebook | Kapsam                                                                                                                 |
| ---------------------------------------------- | :------: | ---------------------------------------------------------------------------------------------------------------------- |
| **A · Veri Hazırlığı**              |  01–02  | Keşifsel veri analizi ve görselleştirme; eksik veri, aykırı değer, kodlama, ölçekleme, özellik mühendisliği |
| **B · Denetimli Öğrenme**             |  03–07  | Doğrusal sınıflandırma ve regresyon aileleri, k-NN, Naive Bayes, SVM ve kernel'lar                                 |
| **C · Ağaç Tabanlı Yöntemler**      |  08–10  | Karar ağaçları, Bagging / Random Forest, Gradient Boosting (XGBoost, LightGBM, CatBoost)                            |
| **D · Değerlendirme ve İyileştirme** |  11–15  | Metrikler, dengesiz veri, hiperparametre optimizasyonu, ensemble, model yorumlanabilirliği (SHAP, LIME)               |
| **E · Denetimsiz Öğrenme**            |  16–20  | Kümeleme, ileri boyut indirgeme (t-SNE, UMAP), anomali tespiti, birliktelik kuralları                                |
| **F · Olasılıksal ve Özel Modeller** |  21–23  | Bayesçi modeller ve MCMC, zaman serisi (ARIMA / ETS / ML), öneri sistemleri                                          |

Notebook numarası lab boyunca (01–23) artar; klasör numarası kategoriyi gösterir. Notebook
bazında ayrıntılı dizin, veri setleri ve kaynak haritası için
[`Classical-ML-lab/README.md`](Classical-ML-lab/README.md) dosyasına bakın.

Notlar:

- R ve SQL bölümleri 14. notebook'tan itibaren bilinçli olarak kısa tutuldu; ana derinlik Python tarafında.
- Egzersizlerin büyük kısmı (özellikle 09'dan itibaren) çözümsüz bırakıldı.
- R hücreleri çıktısız teslim edilir; `rpy2` ve ilgili CRAN paketleriyle kendi makinede çalıştırılır.

---

## 🧪 Yardımcı Çalışmalar

Ana müfredatın dışında kalan, bağımsız uygulama klasörleri:

| Klasör             | İçerik                                                                                                                                                                                                                                                                            |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Kaggle-Examples` | Uçtan uca Kaggle çalışmaları: banka müşteri kaybı (churn), HDI için PCA + K-Means, Netflix içerik analizi, ağ anomali tespiti, Pokémon efsanevi tahmini, Spotify mega-hit tahmini, su içilebilirliği                                                                  |
| `OOP-Examples`    | Sınıf tabanlı (OOP) tasarımla yazılmış küçük ML uygulamaları: CoinGecko API'sinden canlı kripto piyasa verisi analizi, K-Means ile laptop termal profilleme, Isolation Forest + PCA ile canlı deprem anomali radarı, Isolation Forest ile MOBA oyuncu anomali tespiti |
| `Documents`       | Çeşitli notlar, görseller ve `learning_lab.ipynb` deneme defteri                                                                                                                                                                                                               |

---

## 🛠️ Kullanılan Araçlar

**Python** tüm serilerin ortak dilidir: sayısal hesaplama için NumPy ve SciPy, veri işleme için
Pandas, görselleştirme için Matplotlib ve Seaborn, istatistiksel modelleme için Statsmodels, ML
için scikit-learn. Konuya göre ek kütüphaneler kullanılır: SymPy ve PyTorch (Calculus),
scikit-image ve `yfinance` (Linear-Algebra), XGBoost / LightGBM / CatBoost, `imbalanced-learn`,
Optuna, SHAP, LIME, `umap-learn`, `mlxtend` (Classical-ML).

**R**, istatistiksel testler ve modeller için Python'a paralel bir doğrulama katmanıdır;
Jupyter içinden `rpy2` ve `%%R` hücreleriyle çağrılır. Mümkün olduğunda base R fonksiyonları
(`t.test`, `aov`, `TukeyHSD`, `wilcox.test`, `p.adjust`, `lm` …) tercih edilir; böylece
notebook'lar ek paket kurulumuna bağlı kalmadan çalışır.

**SQL (DuckDB)**, Probability-lab ve Classical-ML-lab'da kullanılır: örnekleme simülasyonları,
koşullu olasılık hesapları ve modellerin çekirdek mantığını SQL'de yeniden üretmek için.

---

## 🚀 Çalıştırma

```bash
git clone <repo-url>
cd DS-Mastery-Journey

python3 -m venv .venv && source .venv/bin/activate
pip install -r Probability-lab/requirements.txt   # temel bilimsel yığın + Jupyter + rpy2 + DuckDB
pip install jupyterlab                              # henüz kurulu değilse
jupyter lab
```

- Her lab kendi `data/` klasöründeki yerel veri kopyalarını okur; internetten ek indirme gerektirenler notebook içinde belirtilmiştir.
- Classical-ML-lab için kategoriye göre ek paketler gerekir (`xgboost`, `lightgbm`, `catboost`, `optuna`, `shap`, `lime`, `imbalanced-learn`, `umap-learn`, `mlxtend` …); tam liste ilgili kategori README'sindedir.
- R hücrelerini çalıştırmak için sistemde R ve `rpy2` kurulu olmalıdır.

---

## 🔭 Sırada Ne Var?

Sonraki adım, bu temelin üzerine **derin öğrenme** ve **LLM / RAG**
konularını eklemek. LLM / RAG çalışması bu depodan bağımsız, ayrı bir projede yürütülüyor.

---

## 📚 Referans Kaynak Türleri

Depodaki notebook'lar şu tür kaynaklara dayanır: *Mathematics for Machine Learning*
(Deisenroth, Faisal, Ong), *Introduction to Statistical Learning* ve *Elements of Statistical
Learning* (Hastie, Tibshirani, Friedman), Gilbert Strang'in MIT OCW lineer cebir ve kalkülüs
dersleri, scikit-learn / statsmodels / PyTorch resmi dokümantasyonu, 3Blue1Brown görsel
anlatımları ve ilgili hakemli makaleler. Her notebook'un sonundaki "Kaynaklar" bölümü o konuya
özel tam listeyi verir.
