# 06 · Olasılıksal ve Özel Modeller (Probabilistic & Special Models)

`Classical-ML-lab` serisinin altıncı ve son kategorisi (23 notebook'luk serinin
F bölümü, roadmap'teki notebook'lar 21-23). Bu klasör serinin "özel durumlar"
bölümü: şimdiye kadarki modellerin üç sessiz varsayımını bozuyor. **(1)** Model
tek bir parametre vektörü üretiyordu — `21`'de parametrelerin kendisi bir
*dağılım* oluyor ve tahminle birlikte tahminin ne kadar güvenilir olduğunu
da alıyoruz (Bayesçi doğrusal/lojistik regresyon, MCMC, Gaussian Process).
**(2)** Satırların birbirinden bağımsız olduğunu varsayıp CV'de karıştırıyorduk
— `22`'de sıra bilgidir, gelecek geçmişten tahmin edilir ve doğru
değerlendirme ancak zamanı koruyan bölmeyle mümkündür (ARIMA/ETS ve ML ile
forecasting). **(3)** Her satır bir "örnek"ti — `23`'te veri bir
kullanıcı × öğe matrisidir, %98'den fazlası boştur ve görev boş hücreleri
(ya da sıralamayı) tahmin etmektir. Notebook'lar önceki kategorilere sıkı
bağlı: `21`, Ridge'in "ceza = prior" yorumunu (02/04) ve `13`'teki Bayesçi
optimizasyonun (Optuna/TPE) kardeşi olan GP tabanlı BO'yu işliyor; `22`,
`13`'teki CV disiplinini ve `09`-`10`'daki ağaç topluluklarını zaman
serisine uyarlıyor, `21`'in kapsama (coverage) fikrini aralık tahminlerinde
kullanıyor; `23`, ALS'nin her adımının bir Ridge olduğunu, `15`'teki
SVD/PCA'yı ve `22`'deki zamansal değerlendirmeyi kullanarak kategoriyi — ve
23 notebook'luk seriyi — kapatıyor.

## 📁 Klasör Yapısı

```
06_probability_special_models/
├── README.md
├── 21_bayesian_probabilistic_models.ipynb
├── 22_time_series_ml.ipynb
├── 23_recommendation_systems_basics.ipynb
└── data/
    ├── howell1.csv            ├── wineind.csv
    ├── wells.csv              ├── us_gasoline.csv
    ├── mcycle.csv             ├── melb_temp.csv
    ├── co2_mauna_loa.csv      ├── mt_movies.csv
    ├── gb_books.csv           ├── mt_ratings.csv
    ├── gb_book_tags_top.csv   └── gb_ratings.csv
```

> **Hücre kuralı**: Her notebook'ta sırayla Python → R → SQL (DuckDB) uygulaması.
> R hücreleri `%%R` (rpy2) sihirbazıyla çalışıyor. Proje kararı gereği
> (2026-09-08) R ve SQL bölümleri bilinçli olarak kısa tutuldu (R: tipik olarak
> önce Python referans sayıları, ardından konunun standart R paketiyle 1 hücre;
> SQL: 1-2 hücre) — ana teknik derinlik Python tarafında. Çekirdek algoritmalar
> (Beta–Binom, Metropolis–Hastings, Bayesçi doğrusal regresyon, Laplace,
> Gaussian Process, AR(p)/Yule–Walker, SES, yanlılık modeli, item-kNN, ALS,
> HKV-ALS) önce **sıfırdan** yazılıp `sklearn`/`statsmodels`/`scipy` ile
> doğrulandı. Üç notebook'un da egzersizleri **çözülmeden** (boş kod hücresi +
> `# TODO` adımları) teslim edildi.

---

## 📓 21 · Bayesçi ve Olasılıksal Modeller (`21_bayesian_probabilistic_models.ipynb`)

"Parametreler de belirsizdir; onlara bir olasılık dağılımı atayalım, veri
geldikçe güncelleyelim." Beta–Binom'daki kapalı-form posterior'dan başlayıp,
kapalı formun olmadığı yerde MCMC'ye (Metropolis–Hastings), Bayesçi doğrusal
regresyonun kanıt (evidence) maksimizasyonuna ve Occam usturasına, Bayesçi
lojistik regresyonda Laplace yaklaşımının ne zaman yetersiz kaldığına ve
fonksiyonlar üzerinde prior kuran **Gaussian Process** regresyonuna — Mauna
Loa CO₂'de bileşik çekirdek, aradeğerleme ve öngörünün *nerede çöktüğü*
dahil — ilerliyor. Son adım GP'yi Bayesçi optimizasyon olarak kullanıyor.

```
1. Bayes Kuralı, Eşlenik Prior ve "Neden Dağılım?" (Beta–Binom, MCMC'ye giriş)
2. Bayesçi Doğrusal Regresyon (Gauss–Gauss, kapalı form)
3. Bayesçi Lojistik Regresyon: Laplace Yaklaşımı ve MCMC
4. Gaussian Process (GP) Regresyonu
5. Ne Zaman Bayesçi? Özet

🐍 Python: 6 uygulama adımı — Beta–Binom'da kapalı-form posterior ve
   sıfırdan Metropolis–Hastings (birebir karşılaştırma), Howell1'de Bayesçi
   doğrusal regresyon (Ridge bağlantısı, kanıt maksimizasyonu ile ceza
   katsayısını veriden öğrenme — `sklearn.BayesianRidge` ile birebir —,
   polinom derecesini marjinal olabilirlikle seçme), Wells'te Bayesçi lojistik
   regresyon (Laplace vs rastgele yürüyüşlü Metropolis, R-hat/ESS tanıları,
   küçük veride Laplace'ın yetersizliği ve MLE'ye göre öngörü farkı), mcycle'da
   GP regresyonu (çekirdek seçimi, marjinal olabilirlikle hiperparametre
   öğrenimi, `GaussianProcessRegressor` ile birebir), Mauna Loa CO₂'de bileşik
   çekirdek + aradeğerleme + öngörü, GP ile Bayesçi optimizasyon (Expected
   Improvement)
📊 R: önce Python referans sayıları, ardından `rstanarm::stan_glm` (Stan
   HMC/NUTS) ile Howell yetişkin boy~kilo ve Wells lojistik modeli
   (𝒩(0,1) eğim prior'u)
🗄️ SQL (DuckDB): Bayesçi doğrusal regresyon posterior'unun yeterli
   istatistiklerden (n, Σx, Σx², Σy, Σxy) hesabı — ham veriyi çekmeden —
   ve GP ortalama öngörüsünün eğitim/test noktalarının `CROSS JOIN`'i ile
   yeniden üretimi (Python sonuçlarıyla karşılaştırıldı)
✏️ Egzersizler: E1–E6 (Python) — **çözülmeden** (TODO'lu) teslim edildi:
   prior duyarlılığı, polinom derecesi seçimi, prior genişliği sınavı,
   Metropolis öneri ölçeği ayarı, çekirdek bileşen ablasyonu, edinim
   fonksiyonu karşılaştırması
```

**📦 Veri Setleri:**

| Dataset | Kaynak | Kullanım |
| --- | --- | --- |
| Howell1 | McElreath, [*Statistical Rethinking*](https://xcelab.net/rm/) / [rethinking](https://github.com/rmcelreath/rethinking/blob/master/data/Howell1.csv); Nancy Howell (Dobe !Kung) | Bayesçi doğrusal regresyon, evidence, polinom derecesi seçimi (544 kişi) |
| Wells (Bangladeş kuyu arseniği) | Gelman & Hill (2007), [ARM](https://sites.stat.columbia.edu/gelman/arm/) | Bayesçi lojistik regresyon, Laplace vs MCMC (3.020 hane) |
| mcycle | Silverman (1985), `MASS` / [Rdatasets](https://vincentarelbundock.github.io/Rdatasets/) | GP regresyonu, değişen gürültü (133 ölçüm) |
| Mauna Loa CO₂ | [NOAA GML](https://gml.noaa.gov/ccgg/trends/), [Keeling eğrisi](https://keelingcurve.ucsd.edu/) | Bileşik GP çekirdeği, aradeğerleme, öngörü (aylık, 1958–2026, 822 ay) |

---

## 📓 22 · Zaman Serisi Tahmini (`22_time_series_ml.ipynb`)

Zaman serisinde "bugün dünün devamıdır": satırları karıştırıp CV yapmak çöker.
Notebook iki dünyayı karşılaştırıyor — **klasik istatistiksel modeller**
(durağanlık, ACF/PACF, AR(p) için OLS ve Yule–Walker, SES/ETS ve
ARIMA(0,1,1) ≡ SES denkliği, SARIMA, AICc ile model seçimi, Ljung–Box tanısı) ve
**ML ile forecasting** (seriyi gecikme/yuvarlanan pencere/takvim/Fourier
öznitelikleriyle tabloya çevirme, direct vs recursive çok adımlı tahmin, Ridge
ve LightGBM). Üçüncü ayak **doğru değerlendirme**: MASE, rolling-origin ve
mevsimsel naif taban çizgisi; dördüncüsü **sızıntı ve belirsizlik**
(`rolling(center=True)` gibi sinsi hatalar, quantile ve conformal aralıklar,
kapsama testi). En öğretici bulgulardan biri, ağaç tabanlı modellerin trendi
neden ekstrapole edemediğinin sentetik bir deneyle gösterilmesi.

```
1. Zaman Serisinin Anatomisi ve Durağanlık
2. ARIMA, SARIMA ve Üstel Düzleştirme (ETS)
3. Doğru Ölçü, Doğru Değerlendirme (MASE, rolling-origin)
4. ML ile Forecasting: Seriyi Tabloya Çevirmek
5. Ne Zaman Hangisi? (Özet)

🐍 Python: 6 uygulama adımı — wineind'de STL ayrıştırması, ADF/KPSS
   durağanlık testleri, ACF/PACF; sıfırdan AR(p) (OLS vs Yule–Walker), SES ve
   ARIMA(0,1,1) denkliğinin sayısal doğrulaması; klasik modeller (SARIMA, ETS)
   yürüyen başlangıçla (MASE); haftalık benzinde kesirli mevsimsellik (52,18
   hafta), Fourier öznitelikleri ve ML (Ridge / LightGBM, direct vs
   recursive); "ağaçlar trendi ekstrapole edemez" sentetik deneyi; Melbourne
   günlük sıcaklıkta sızıntı (`rolling(center=True)`, karıştırılmış CV vs
   `TimeSeriesSplit`) ve belirsizlik (quantile regresyon, conformal / CQR
   aralıkları, kapsama)
📊 R: önce Python'da aynı tek bölme (son 24 ay) için referans MASE'ler,
   ardından `forecast` paketiyle (`auto.arima`, `ets`, `snaive`, `tslm` +
   `fourier`, `accuracy`) wineind karşılaştırması
🗄️ SQL (DuckDB): `LAG` ve **geçmişe** bakan yuvarlanan pencere
   (`ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING`) öznitelikleri + MASE paydası
   (karşıt örnek: `3 PRECEDING AND 3 FOLLOWING` bir sızıntıdır) ve SES'in
   `WITH RECURSIVE` ile yeniden üretimi — Python sonuçlarıyla karşılaştırıldı
✏️ Egzersizler: E1–E6 (Python) — **çözülmeden** (TODO'lu) teslim edildi:
   durağanlık reçetesi, ETS aday kümesi, Fourier terim sayısı seçimi,
   LightGBM öznitelik ablasyonu, çok-adımlı hedef/sızıntı, aralık kalibrasyonu
```

**📦 Veri Setleri:**

| Dataset | Kaynak | Kullanım |
| --- | --- | --- |
| wineind | R `forecast` paketi (Avustralya İstatistik Bürosu), [Rdatasets](https://vincentarelbundock.github.io/Rdatasets/) | STL, SARIMA/ETS, yürüyen başlangıç, MASE (aylık şarap satışları, Ocak 1980–Ağustos 1994, 176 ay) |
| us_gasoline | EIA, [FPP3 / `fpp3`](https://otexts.com/fpp3/) | Kesirli mevsimsellik (52,18 hafta), Fourier, ML (haftalık ABD benzin tedariki, Şubat 1991–Ocak 2017, 1.355 hafta) |
| Melbourne günlük minimum sıcaklık | Avustralya Meteoroloji Bürosu (1981–1990) | Sızıntı deneyleri, belirsizlik / conformal aralıklar (3.650 gün) |

---

## 📓 23 · Öneri Sistemlerinin Temelleri (`23_recommendation_systems_basics.ipynb`)

Seriyi kapatan notebook. Veri bir **kullanıcı × öğe matrisidir**; görev boş
hücreleri (ya da sıralamayı) tahmin etmek. Dört aile sıfırdan yazılıp
doğrulanıyor: Netflix Prize'dan beri "yenilmesi şaşırtıcı derecede zor" ilk adım
olan **yanlılık (bias) modeli** ($\hat r_{ui}=\mu+b_u+b_i$, `Ridge` ile birebir),
**komşuluk tabanlı** iş birlikçi filtreleme (item-kNN, bias-düzeltilmiş
kosinüs), **matris çarpanlarına ayırma** (ALS kapalı form + SGD; sıfırla
doldurulmuş SVD'nin neden kötü olduğu) ve **kapalı geri bildirim** için
Hu–Koren–Volinsky ALS. Değerlendirmenin asıl tuzakları ayrıca ele alınıyor:
rastgele vs zamansal bölme, RMSE'nin sıralama kalitesini ölçmemesi, popülerlik
taban çizgisini yenememek ve soğuk başlangıç.

```
1. Problem Tanımı ve Değerlendirme (RMSE, HR@K, Recall@K, NDCG@K)
2. Yanlılık (Bias) Modeli: En Güçlü Basit Taban Çizgisi
3. Komşuluk Tabanlı İş Birlikçi Filtreleme (Item-kNN)
4. Matris Çarpanlarına Ayırma ve ALS
5. Kapalı Geri Bildirim: Hu–Koren–Volinsky (HKV) ALS
6. İçerik Tabanlı Öneri ve Soğuk Başlangıç
7. Özet

🐍 Python: 7 uygulama adımı (4b dahil) — MovieTweetings'te keşif ve zamansal
   bölme, taban çizgileri + yanlılık modeli (`Ridge` ile doğrulama), item-kNN
   (bias-düzeltilmiş kosinüs), ALS / SGD / sıfırla doldurulmuş SVD ile matris
   çarpanlarına ayırma, "RMSE iyi ama sıralama?" (MovieTweetings'te Top-N),
   bölme stratejisinin etkisi (rastgele vs kullanıcı-içi kronolojik),
   Goodbooks'ta kapalı geri bildirim: HKV-ALS ve popülerlik yanlılığı, etiketlerle
   içerik tabanlı öneri ve soğuk başlangıç (TF-IDF)
📊 R: önce Python referans sayıları, ardından `recommenderlab`
   (`realRatingMatrix`, `evaluationScheme`; POPULAR / UBCF / IBCF / SVDF / ALS)
   — bölme farklı olduğundan sayılar birebir değil, sıralama beklenir
🗄️ SQL (DuckDB): yanlılık modelinin bir geçişi (`GROUP BY` + iki `JOIN`;
   Python `bias_fit(iters=1)` ile aynı katsayı ve doğrulama RMSE'si) ve
   item–item kosinüs benzerliğinin self-join ile hesabı (Goodbooks, ikili) —
   Python ile birebir
✏️ Egzersizler: E1–E6 (Python) — **çözülmeden** (TODO'lu) teslim edildi:
   ayrı λ_u/λ_i büzülmesi, benzerlik ölçüsü karşılaştırması, ALS-WR (ağırlıklı
   düzenleme), puana bağlı HKV güveni, içerik özelliklerinin zenginleştirilmesi,
   yeniden sıralama ile popülerlik yanlılığını azaltma
```

**Dürüst bulgular** (notebook çıktılarında bilinçli olarak olduğu gibi bırakıldı):
MF ve item-kNN, yanlılık modeline göre RMSE'de yalnızca ~%1 kazandırdı;
MovieTweetings'te sıralama metriklerinde popülerlik taban çizgisi kazandı;
Goodbooks'ta item-kNN, HKV-ALS'yi geçti (ALS ızgarası $k=64$ kenarında kaldı,
daha geniş ızgara denenmedi — kesin bir üstünlük iddiası değil); soğuk
başlangıçta içerik tabanlı öneri rastgeleye göre yaklaşık 13,5× iyiyken
ALS'nin soğuk öğelerdeki performansı rastgeleyle aynı çıktı; rastgele bölme,
zamansal bölmeye göre RMSE'yi 0,033 iyimser gösterdi.

**📦 Veri Setleri:**

| Dataset | Kaynak | Kullanım |
| --- | --- | --- |
| MovieTweetings | [GitHub (sidooms)](https://github.com/sidooms/MovieTweetings), Dooms, De Pessemier & Martens (2013) | Açık geri bildirim, bias modeli, item-kNN, ALS/SGD, zamansal bölme (filtreden sonra 593.400 puan, 8.576 kullanıcı, 4.779 film, 0–10 ölçeği) |
| Goodbooks-10k | [GitHub (Zygmunt Zając)](https://github.com/zygmuntz/goodbooks-10k) | Kapalı geri bildirim (HKV-ALS), popülerlik yanlılığı, içerik tabanlı soğuk başlangıç (3.000 kullanıcı örneği: 335.699 etkileşim, 9.993 kitap) |

---

## 📚 Kaynak Haritası

### Kitaplar

| # | Kitap | Yazar | Ücretsiz? | Bağlantı |
| --- | --- | --- | --- | --- |
| 🥇 | Forecasting: Principles and Practice (3. baskı, FPP3) | Hyndman & Athanasopoulos | ✅ | [otexts.com/fpp3](https://otexts.com/fpp3/) |
| 🥇 | Gaussian Processes for Machine Learning | Rasmussen & Williams | ✅ | [gaussianprocess.org/gpml](https://gaussianprocess.org/gpml/) |
| 🥇 | Pattern Recognition and Machine Learning (Bölüm 3, 4.4, 6.4) | Bishop | ✅ | [Microsoft Research](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) |
| 🥈 | Statistical Rethinking | McElreath | ❌ | [xcelab.net/rm](https://xcelab.net/rm/) |
| 🥈 | Bayesian Data Analysis (BDA3) | Gelman ve ark. | ✅ (yazar sayfasında PDF) | [Gelman — BDA3](https://sites.stat.columbia.edu/gelman/book/) |
| 🥈 | Recommender Systems Handbook | Ricci, Rokach, Shapira (Ed.) | ❌ | [Springer](https://link.springer.com/book/10.1007/978-1-4899-7637-6) |
| 🥉 | Recommender Systems: The Textbook | Aggarwal | ❌ | [Springer](https://doi.org/10.1007/978-3-319-29659-3) |
| 🥉 | Time Series Analysis: Forecasting and Control | Box, Jenkins, Reinsel, Ljung | ❌ | Wiley |

### Anahtar Makaleler

| Makale | Notebook | Konu |
| --- | --- | --- |
| Metropolis ve ark. (1953) / Hastings (1970), *Biometrika* | 21 | Metropolis–Hastings algoritması |
| MacKay (1992), *Bayesian Interpolation* | 21 | Kanıt maksimizasyonu, etkin parametre sayısı |
| Gelman & Rubin (1992) / Vehtari ve ark. (2021), *Rank-normalization… improved $\hat R$* | 21 | $\hat R$ ve ESS tanıları |
| Shahriari ve ark. (2016), *Taking the Human Out of the Loop* | 21 | Bayesçi optimizasyon incelemesi |
| Hyndman & Koehler (2006), *Another look at measures of forecast accuracy* | 22 | MASE |
| Hyndman & Khandakar (2008), *Automatic Time Series Forecasting: The forecast Package for R* | 22 | `auto.arima`, `ets` |
| Cleveland ve ark. (1990), *STL* | 22 | Mevsimsel-trend ayrıştırması |
| Bergmeir & Benítez (2012), *On the use of cross-validation for time series predictor evaluation* | 22 | Zaman serisinde CV |
| Romano, Patterson, Candès (2019), *Conformalized Quantile Regression* | 22 | CQR aralıkları |
| Koren, Bell, Volinsky (2009), *Matrix Factorization Techniques for Recommender Systems* | 23 | Yanlılık + MF, ALS/SGD |
| Hu, Koren, Volinsky (2008), *Collaborative Filtering for Implicit Feedback Datasets* | 23 | HKV-ALS |
| Sarwar, Karypis, Konstan, Riedl (2001), *Item-based Collaborative Filtering Recommendation Algorithms* | 23 | Item-kNN |
| Cremonesi, Koren, Turrin (2010), *Performance of Recommender Algorithms on Top-N Recommendation Tasks* | 23 | RMSE'nin sıralamayı ölçmemesi |
| Rendle, Zhang, Koren (2019), *On the Difficulty of Evaluating Baselines* | 23 | Güçlü taban çizgilerinin önemi |

### Resmi Dokümantasyon

| Kaynak | Kullanım |
| --- | --- |
| [scikit-learn — Gaussian Processes](https://scikit-learn.org/stable/modules/gaussian_process.html) | Notebook 21 |
| [statsmodels — Time Series Analysis](https://www.statsmodels.org/stable/tsa.html) | Notebook 22 |
| [LightGBM](https://lightgbm.readthedocs.io/) | Notebook 22 |
| [implicit — ALS / BPR kütüphanesi](https://github.com/benfred/implicit) | Notebook 23 (üretim referansı) |
| [CRAN — `rstanarm`](https://cran.r-project.org/package=rstanarm) / [Stan](https://mc-stan.org/rstanarm/) | Notebook 21 R uygulaması |
| [CRAN — `forecast`](https://cran.r-project.org/package=forecast) | Notebook 22 R uygulaması |
| [CRAN — `recommenderlab`](https://cran.r-project.org/package=recommenderlab) | Notebook 23 R uygulaması |
| [DuckDB — SQL Fonksiyonları](https://duckdb.org/docs/sql/functions/numeric) & [Window Functions](https://duckdb.org/docs/sql/window_functions) | Tüm notebook'ların SQL bölümlerinin temeli |

Notebook içi `Kaynaklar` bölümlerinde her dosya için tam kaynak listesi
mevcut (veri seti sayfaları, R paket dokümantasyonları, projedeki önceki
notebook'lara çapraz referanslar dahil).

---
