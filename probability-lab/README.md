# Probability Lab — Tam Rehber
## Yapı · Taslak · Dataset · Kaynaklar

---

## 📁 Klasör Yapısı

```
probability-lab/
├── README.md
├── 01_foundations/
│   ├── pmf_pdf_cdf.ipynb
│   └── data/
├── 02_discrete_distributions/
│   ├── bernoulli_binomial.ipynb
│   ├── poisson_geometric.ipynb
│   └── data/
├── 03_continuous_distributions/
│   ├── normal_uniform.ipynb
│   ├── exponential_gamma_chi2.ipynb
│   ├── beta_lognormal.ipynb
│   └── data/
├── 04_joint_distributions/
│   ├── joint_marginal_conditional.ipynb
│   ├── covariance_correlation.ipynb
│   └── data/
├── 05_expectation_variance/
│   ├── expectation_variance_props.ipynb
│   └── data/
├── 06_limit_theorems/
│   ├── law_of_large_numbers.ipynb
│   ├── central_limit_theorem.ipynb
│   └── data/
├── 07_bayes/
│   ├── bayes_theorem.ipynb
│   ├── conjugate_priors.ipynb
│   └── data/
├── 08_inference/
│   ├── mle_estimation.ipynb
│   ├── map_estimation.ipynb
│   └── data/
├── 09_information_theory/
│   ├── entropy_kl_divergence.ipynb
│   └── data/
└── 10_simulation/
    ├── monte_carlo.ipynb
    ├── bootstrap.ipynb
    └── data/
```

> **Hücre kuralı**: Her notebook'ta sırayla Python → R → SQL hücreleri.  
> Magic: `%%R` (rpy2), `%%sql` (ipython-sql / DuckDB).

---

## 📓 Notebook Taslakları & Dataset Önerileri

---

### 01 · PMF / PDF / CDF (`pmf_pdf_cdf.ipynb`)

```
# 1. Giriş — Neden PMF/PDF/CDF?
## 1.1 Teorik Özet (Markdown)
   - PMF: P(X=x) | Kesikli
   - PDF: f(x) = dF/dx | Sürekli (yoğunluk ≠ olasılık)
   - CDF: F(x) = P(X≤x) | Her ikisinde geçerli
   - İlişki: PMF→CDF (toplama), PDF→CDF (integral), CDF→PMF/PDF (türev)

## 1.2 [PYTHON] PMF Görselleştirme (scipy, matplotlib)
   - Zar atışı PMF çizimi
   - CDF'e geçiş (cumsum)

## 1.3 [PYTHON] PDF vs Histogram
   - KDE ile ampirik PDF
   - scipy.stats.norm.pdf, .cdf

## 1.4 [R] Aynı dağılımlar — base R + ggplot2
   - dpois(), ppois(), qpois()

## 1.5 [SQL] CDF hesabı
   - Window function: SUM() OVER (ORDER BY x) / total_count
   - Ampirik CDF bir tabloda

## 1.6 Egzersiz: 3 farklı dağılım için PMF/PDF ve CDF yan yana çiz
```

**📦 Dataset:**
| Dataset | Kaynak | Kullanım |
|---|---|---|
| `seaborn.load_dataset("tips")` | Seaborn built-in | Bahşiş tutarı → sürekli hist/PDF |
| Zar simülasyonu (`np.random.randint`) | Synthetic | PMF demonstrasyonu |
| `scipy.stats` frozen dist | scipy | Teorik PMF/PDF/CDF |

---

### 02 · Bernoulli & Binomial (`bernoulli_binomial.ipynb`)

```
# 2. Bernoulli ve Binomial Dağılımlar

## 2.1 Teorik Özet (Markdown)
   - Bernoulli: tek deneme, E=p, Var=p(1-p)
   - Binomial: n bağımsız Bernoulli, kombinatorik formül
   - Binom katsayısı hesabı ve Pascal üçgeni

## 2.2 [PYTHON] scipy.stats.binom — pmf, cdf, rvs
   - Farklı n,p değerleri için şekil değişimi
   - Animasyonlu: n arttıkça Normal'e yaklaşma

## 2.3 [PYTHON] Gerçek Veri Uyumu
   - Coin flip deneyi simülasyonu
   - Kalp krizi verisi: hasta/sağlıklı sınıflandırma

## 2.4 [R] dbinom(), pbinom(), rbinom()
   - ggplot2 ile bar chart

## 2.5 [SQL] Binomial olasılık tablosu
   - Faktöriyel hesabı ile PMF manuel

## 2.6 Egzersiz: Email campaign click-through rate modellemesi
```

**📦 Dataset:**
| Dataset | Kaynak | Kullanım |
|---|---|---|
| Heart Disease UCI | [Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-ms) | Hasta=1/Sağlıklı=0 → Bernoulli |
| Titanic | `seaborn.load_dataset("titanic")` | Hayatta kalma → Binomial |
| A/B Test Sonuçları | [Kaggle](https://www.kaggle.com/datasets/zhangluyuan/ab-testing) | CTR modellemesi |

---

### 02b · Poisson & Geometric (`poisson_geometric.ipynb`)

```
# 3. Poisson ve Geometric Dağılımlar

## 3.1 Teorik Özet (Markdown)
   - Poisson: λ parametresi, E=Var=λ (eşsiz özellik)
   - Poisson süreci: zaman/alan içinde sayma
   - Geometric: ilk başarıya kadar bekleme, hafızasızlık özelliği

## 3.2 [PYTHON] Poisson — Gerçek Dünya
   - Saatte gelen çağrı sayısı simülasyonu
   - λ değişiminin dağılım şekline etkisi

## 3.3 [PYTHON] Geometric — Memoryless Property İspatı
   - P(X>m+n | X>m) = P(X>n) gösterimi

## 3.4 [R] dpois(), dgeom() + ggplot2

## 3.5 [SQL] Poisson tablosu — çağrı merkezi analizi
   - GROUP BY saat, COUNT(*) → λ tahmini

## 3.6 Egzersiz: Wikipedia edit sayısı → Poisson fit
```

**📦 Dataset:**
| Dataset | Kaynak | Kullanım |
|---|---|---|
| NYC 311 Service Requests | [NYC Open Data](https://data.cityofnewyork.us) | Saatlik çağrı sayısı → Poisson |
| Bike Sharing (UCI) | [UCI ML Repo](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) | Saatlik kiralama → Poisson |
| Manufacturing Defects | Synthetic (`np.random.poisson`) | Üretim hata sayısı |

---

### 03a · Normal & Uniform (`normal_uniform.ipynb`)

```
# 4. Normal ve Uniform Dağılımlar

## 4.1 Teorik Özet (Markdown)
   - Normal: 68-95-99.7 kuralı, simetri, Z-dönüşümü
   - Standardizasyon: Z = (X-μ)/σ
   - Uniform: a, b parametreleri; E=(a+b)/2

## 4.2 [PYTHON] QQ-Plot ile Normallik Testi
   - scipy.stats.probplot
   - Shapiro-Wilk testi

## 4.3 [PYTHON] Uniform → Normal dönüşümü (CLT bağlantısı)
   - Box-Muller dönüşümü

## 4.4 [R] dnorm(), pnorm(), qqnorm() + ggplot2

## 4.5 [SQL] Z-score hesabı
   - (value - AVG()) / STDDEV() OVER ()

## 4.6 Egzersiz: Iris veri seti — her feature için normallik testi
```

**📦 Dataset:**
| Dataset | Kaynak | Kullanım |
|---|---|---|
| Iris | `sklearn.datasets.load_iris()` | Petal uzunluğu → Normal kontrol |
| Heights (SOCR) | [SOCR Data](http://wiki.socr.umich.edu/index.php/SOCR_Data) | Boy dağılımı → Normal |
| Rolling dice | Synthetic | Uniform demonstrasyonu |

---

### 03b · Exponential, Gamma, Chi-Square (`exponential_gamma_chi2.ipynb`)

```
# 5. Exponential · Gamma · Chi-Square Ailesi

## 5.1 Teorik Özet (Markdown)
   - Aile ağacı: Gamma → Exp (α=1) → Chi² (α=k/2, β=2)
   - Exponential: hafızasızlık özelliği, λ yorumu
   - Gamma: α olay bekleme süresi
   - Chi-Square: hipotez testinde dağılım

## 5.2 [PYTHON] Aile Ağacı Görselleştirmesi
   - Tek grafik: Gamma farklı α değerleri ile
   - Exp ve Chi² Gamma'nın özel hali olarak göster

## 5.3 [PYTHON] Gerçek Veri: Müşteri Bekleme Süresi
   - MLE ile Exponential fit

## 5.4 [R] dexp(), dgamma(), dchisq() + facet_wrap

## 5.5 [SQL] Ortalama bekleme süresi → λ tahmini

## 5.6 Egzersiz: Chi-Square bağımsızlık testi (iki kategorik değişken)
```

**📦 Dataset:**
| Dataset | Kaynak | Kullanım |
|---|---|---|
| Hospital Wait Times | [Kaggle](https://www.kaggle.com/datasets/nehaprabhavalkar/av-healthcare-analytics-ii) | Bekleme süresi → Exponential |
| Income Distribution | `seaborn.load_dataset("diamonds")` | Fiyat → Log-normal/Gamma |
| Chi-Square Test | `seaborn.load_dataset("titanic")` | Cinsiyet × Hayatta kalma |

---

### 03c · Beta & Log-Normal (`beta_lognormal.ipynb`)

```
# 6. Beta ve Log-Normal Dağılımlar

## 6.1 Teorik Özet (Markdown)
   - Beta: [0,1] aralığı, α ve β şekil parametreleri
   - Beta özel haller: α=β=1 → Uniform; α=β → Simetrik
   - Log-Normal: X ~ LogNormal ↔ ln(X) ~ Normal
   - Sağa çarpıklık ve doğal log dönüşümü

## 6.2 [PYTHON] Beta — Bayesian Prior Görselleştirmesi
   - Farklı α,β ile şekil oyunu
   - A/B test: conversion rate prior → posterior

## 6.3 [PYTHON] Log-Normal — Gelir Verisi
   - Raw vs log-transform karşılaştırması

## 6.4 [R] dbeta(), dlnorm() + interactive plotly

## 6.5 [SQL] Log transform + histogram bucket

## 6.6 Egzersiz: Kullanıcı session süresi → Log-Normal fit
```

**📦 Dataset:**
| Dataset | Kaynak | Kullanım |
|---|---|---|
| UCI Machine Learning 1994 Census Bureau Adult Income | [Kaggle Adult Income](https://www.kaggle.com/datasets/uciml/adult-census-income)   | Gelir → Log-Normal |
| Web Session Duration | [Kaggle E-Commerce Electronic Store](https://www.kaggle.com/datasets/mkechinov/ecommerce-purchase-history-from-electronics-store) | Session süresi → Log-Normal |
| Beta simülasyonu | Synthetic | Prior elicitation |

---

### 04 · Joint Distributions (`joint_marginal_conditional.ipynb`)

```
# 7. Ortak, Marjinal ve Koşullu Dağılımlar

## 7.1 Teorik Özet (Markdown)
   - Joint: f(x,y) veya P(X=x, Y=y)
   - Marginal: ∫ f(x,y)dy veya Σ_y P(x,y)
   - Conditional: f(x|y) = f(x,y)/f(y)
   - Bağımsızlık koşulu: f(x,y) = f(x)·f(y)

## 7.2 [PYTHON] 2D Dağılım Görselleştirmesi
   - sns.jointplot(), heatmap, contour
   - Marjinal dağılımlar kenar panellerinde

## 7.3 [PYTHON] Koşullu Dağılım Analizi
   - P(fiyat | şehir) — gerçek veri

## 7.4 [R] ggplot2 + geom_density2d

## 7.5 [SQL] Koşullu olasılık tablosu
   - P(survived | sex) — Titanic

## 7.6 Egzersiz: Diamonds — fiyat ve karat ortak dağılımı
```

**📦 Dataset:**
| Dataset | Kaynak | Kullanım |
|---|---|---|
| `seaborn.load_dataset("diamonds")` | Seaborn | Karat × Fiyat joint dist |
| `seaborn.load_dataset("penguins")` | Seaborn | Bill length × depth |
| Boston Housing | `sklearn.datasets` | Çok değişkenli joint |

---

### 04b · Covariance & Correlation (`covariance_correlation.ipynb`)

```
# 8. Kovaryans ve Korelasyon

## 8.1 Teorik Özet (Markdown)
   - Cov(X,Y) = E[(X-μx)(Y-μy)]
   - Pearson r = Cov(X,Y) / (σx·σy), r ∈ [-1,1]
   - Spearman: rank tabanlı, non-linear ilişki
   - Korelasyon ≠ Nedensellik (klasik örnekler)

## 8.2 [PYTHON] Anscombe Quartet Demonstrasyonu
   - 4 farklı veri seti, aynı r değeri
   - "Veriyi görmeden istatistiğe güvenme" dersi

## 8.3 [PYTHON] Korelasyon Matrisi + Heatmap
   - Finans verisi: hisse senetleri

## 8.4 [R] cor(), corrplot paketi

## 8.5 [SQL] Pearson korelasyon hesabı
   - CORR() fonksiyonu (PostgreSQL/DuckDB)

## 8.6 Egzersiz: Sahte korelasyonlar bulma (spurious correlations)
```

**📦 Dataset:**
| Dataset | Kaynak | Kullanım |
|---|---|---|
| Anscombe Quartet | `seaborn.load_dataset("anscombe")` | Kovaryans yanıltıcılığı |
| S&P 500 Stocks | `yfinance` / [Kaggle](https://www.kaggle.com/datasets/camnugent/sandp500) | Hisse kovaryans matrisi |
| WHO Health Stats | [WHO](https://www.who.int/data) | Ülke sağlık göstergeleri |

---

### 05 · Expectation & Variance (`expectation_variance_props.ipynb`)

```
# 9. Beklenti, Varyans ve Özellikleri

## 9.1 Teorik Özet (Markdown)
   - E[aX+b] = aE[X]+b (lineerlik)
   - E[X+Y] = E[X]+E[Y] (her zaman)
   - Var(aX) = a²Var(X)
   - Var(X+Y) = Var(X)+Var(Y)+2Cov(X,Y)
   - Bağımsızlıkta: Var(X+Y) = Var(X)+Var(Y)

## 9.2 [PYTHON] Loss Fonksiyonu Geometrisi
   - MSE = E[(Y - Ŷ)²] — varyans bağlantısı
   - Bias-Variance tradeoff görsel

## 9.3 [PYTHON] Portfolio Varyansı
   - İki hisse senedi; ağırlıklı toplam

## 9.4 [R] sapply + custom E/Var fonksiyonları

## 9.5 [SQL] VAR_POP(), VAR_SAMP() karşılaştırması

## 9.6 Egzersiz: Simüle edilmiş öğrenci notları — E ve Var hesabı
```

**📦 Dataset:**
| Dataset | Kaynak | Kullanım |
|---|---|---|
| Student Performance | [UCI](https://archive.ics.uci.edu/dataset/320/student+performance) | Not → E, Var |
| Portfolio Simulation | `yfinance` | Hisse portföy varyansı |
| California Housing | `sklearn.datasets` | Fiyat E ve Var |

---

### 06a · Law of Large Numbers (`law_of_large_numbers.ipynb`)

```
# 10. Büyük Sayılar Yasası (LLN)

## 10.1 Teorik Özet (Markdown)
   - Zayıf LLN: X̄_n → μ (olasılıkta)
   - Güçlü LLN: X̄_n → μ (neredeyse kesinlikle)
   - Pratikte: n büyüdükçe örneklem ortalaması gerçeğe yaklaşır
   - Monte Carlo'nun matematiği bu

## 10.2 [PYTHON] Animasyonlu LLN
   - Kümülatif ortalama; yatay μ çizgisi
   - Farklı dağılımlar için (Normal, Poisson, Exponential)

## 10.3 [PYTHON] Casino Simülasyonu
   - Roulette: ev kenarı (house edge) LLN ile ortaya çıkar

## 10.4 [R] cumsum / cummean animasyonu

## 10.5 [SQL] Kümülatif ortalama
   - AVG() OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)

## 10.6 Egzersiz: Önyargılı madeni para → kaç denemede p=0.7 ortaya çıkar?
```

**📦 Dataset:**
| Dataset | Kaynak | Kullanım |
|---|---|---|
| Coin flip simülasyonu | `np.random.binomial` | Klasik LLN |
| Stock Returns | `yfinance` | Günlük getiri ortalaması |
| Survey data | [ANES](https://electionstudies.org) | Örneklem büyüklüğü etkisi |

---

### 06b · Central Limit Theorem (`central_limit_theorem.ipynb`)

```
# 11. Merkezi Limit Teoremi (CLT)

## 11.1 Teorik Özet (Markdown)
   - X̄_n ~ N(μ, σ²/n) yeterince büyük n için
   - "Yeterince büyük" n ≥ 30 (pratik kural)
   - Orijinal dağılımdan bağımsız!
   - A/B testi, güven aralığı, hipotez testinin temeli

## 11.2 [PYTHON] CLT Gösterimi — 4 Farklı Dağılım
   - Uniform, Exponential, Bimodal, Bernoulli
   - Her birinden 1000 örneklem ortalaması → Normal

## 11.3 [PYTHON] Standart Hata ve Güven Aralığı
   - SE = σ/√n
   - %95 CI = X̄ ± 1.96·SE

## 11.4 [R] replicate() + ggplot histogram

## 11.5 [SQL] Bootstrap resampling ile CLT gösterimi

## 11.6 Egzersiz: E-ticaret sipariş tutarları — CLT ile CI hesabı
```

**📦 Dataset:**
| Dataset | Kaynak | Kullanım |
|---|---|---|
| E-Commerce Transactions | [Kaggle](https://www.kaggle.com/datasets/carrie1/ecommerce-data) | Sipariş tutarı CLT |
| NYC Taxi Fares | [NYC TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) | Büyük veri CLT |
| Synthetic populations | `scipy.stats` | Kontrollü CLT deneyi |

---

### 07a · Bayes Theorem (`bayes_theorem.ipynb`)

```
# 12. Bayes Teoremi

## 12.1 Teorik Özet (Markdown)
   - P(A|B) = P(B|A)·P(A) / P(B)
   - Prior → Likelihood → Posterior zinciri
   - Naive Bayes bağlantısı (MAP tahmini)

## 12.2 [PYTHON] Tıbbi Test Paradoksu
   - Hastalık prevalansı %1, test %95 doğru
   - P(hastalık | pozitif test) = ?

## 12.3 [PYTHON] Spam Filtresi — Naive Bayes
   - sklearn.naive_bayes.MultinomialNB

## 12.4 [R] bayesAB paketi

## 12.5 [SQL] Koşullu olasılık tablosu + Bayes el hesabı

## 12.6 Egzersiz: Kredi kartı dolandırıcılığı tespiti
```

**📦 Dataset:**
| Dataset | Kaynak | Kullanım |
|---|---|---|
| SMS Spam Collection | [UCI](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) | Naive Bayes spam |
| Credit Card Fraud | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | Fraud tespiti |
| Medical Test Simulation | Synthetic | Bayes paradoksu |

---

### 07b · Conjugate Priors (`conjugate_priors.ipynb`)

```
# 13. Conjugate Priors — Bayesian Workflow

## 13.1 Teorik Özet (Markdown)
   - Prior × Likelihood → Posterior (aynı ailede)
   - Beta-Binomial: p tahmini
   - Gamma-Poisson: λ tahmini
   - Normal-Normal: μ tahmini

## 13.2 [PYTHON] Beta-Binomial Güncellemesi
   - A/B test: her yeni gözlemle posterior güncelle
   - Animasyonlu: prior → posterior serisi

## 13.3 [PYTHON] Gamma-Poisson ile λ Tahmini
   - Çağrı merkezi: λ'yı öğren

## 13.4 [R] bayesplot paketi

## 13.5 [SQL] N/A (teorik → Python/R odaklı bölüm)

## 13.6 Egzersiz: Website conversion rate — Beta-Binomial ile canlı güncelleme
```

**📦 Dataset:**
| Dataset | Kaynak | Kullanım |
|---|---|---|
| A/B Test Data | [Kaggle](https://www.kaggle.com/datasets/zhangluyuan/ab-testing) | Beta-Binomial |
| Call Center Data | Synthetic | Gamma-Poisson |
| Cookie Cats A/B | [Kaggle](https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing-cookie-cats) | Mobile game retention |

---

### 08a · MLE (`mle_estimation.ipynb`)

```
# 14. Maximum Likelihood Estimation (MLE)

## 14.1 Teorik Özet (Markdown)
   - L(θ|x) = Π f(xᵢ|θ) — likelihood fonksiyonu
   - Log-likelihood: ℓ(θ) = Σ log f(xᵢ|θ)
   - MLE: θ̂ = argmax ℓ(θ)
   - Normal, Binomial, Poisson için kapalı form çözümler

## 14.2 [PYTHON] MLE — Adım Adım
   - Log-likelihood surface görselleştirme
   - scipy.optimize.minimize ile numerik MLE

## 14.3 [PYTHON] MLE = Lineer Regresyon Bağlantısı
   - OLS'in istatistiksel yorumu

## 14.4 [R] fitdistr() (MASS paketi)

## 14.5 [SQL] MLE'nin SQL analogu — MAP ile karşılaştır

## 14.6 Egzersiz: Gerçek veri ile Exponential MLE
```

**📦 Dataset:**
| Dataset | Kaynak | Kullanım |
|---|---|---|
| Survival Analysis | [SEER](https://seer.cancer.gov) | Hayatta kalma MLE |
| Airbnb Prices | [Kaggle](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) | Lognormal MLE fit |
| Insurance Claims | [Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance) | Normal/Gamma MLE |

---

### 09 · Information Theory (`entropy_kl_divergence.ipynb`)

```
# 15. Entropi & KL Divergence

## 15.1 Teorik Özet (Markdown)
   - Shannon Entropy: H(X) = -Σ p(x)·log₂p(x)
   - Cross-Entropy: H(p,q) = -Σ p(x)·log q(x)
   - KL Divergence: D_KL(P‖Q) = Σ P(x)·log[P(x)/Q(x)]
   - KL ≥ 0 (Gibbs' inequality); asimetrik!
   - ML bağlantısı: cross-entropy loss = KL + H(Y)

## 15.2 [PYTHON] Entropi Geometrisi
   - Uniform dağılımda maksimum entropi
   - Deterministic dağılımda sıfır entropi

## 15.3 [PYTHON] Cross-Entropy Loss → Neural Net Bağlantısı
   - Binary classification loss hesabı

## 15.4 [R] entropy paketi

## 15.5 [SQL] Entropi hesabı (metin sınıflandırma)
   - Karar ağacı split kriteri

## 15.6 Egzersiz: İki dil modeli arasında KL divergence
```

**📦 Dataset:**
| Dataset | Kaynak | Kullanım |
|---|---|---|
| MNIST (class dist.) | `sklearn.datasets` | Etiket dağılımı entropisi |
| Text Classification | [20 Newsgroups](https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset) | Metin sınıf entropisi |
| Language Model Outputs | HuggingFace | KL divergence iki model arası |

---

### 10a · Monte Carlo (`monte_carlo.ipynb`)

```
# 16. Monte Carlo Simülasyonu

## 16.1 Teorik Özet (Markdown)
   - LLN'nin uygulaması: E[f(X)] ≈ (1/n)Σ f(xᵢ)
   - π tahmini (daire/kare yöntemi)
   - MCMC (Markov Chain Monte Carlo) girişi
   - Bayesian inference bağlantısı

## 16.2 [PYTHON] π Tahmini — Klasik Demo
   - Animasyonlu nokta saçılımı

## 16.3 [PYTHON] Finansal Simülasyon
   - Portföy değeri dağılımı (Geometric Brownian Motion)
   - Value at Risk (VaR) hesabı

## 16.4 [R] replicate() + Shiny widget

## 16.5 [SQL] Pseudo-random ile basit Monte Carlo

## 16.6 Egzersiz: Sigorta risk simülasyonu
```

**📦 Dataset:**
| Dataset | Kaynak | Kullanım |
|---|---|---|
| S&P 500 Returns | `yfinance` | Portföy simülasyonu |
| Insurance Claims | [Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance) | Risk simülasyonu |
| Casino Game Simulation | Synthetic | House edge gösterimi |

---

### 10b · Bootstrap (`bootstrap.ipynb`)

```
# 17. Bootstrap Yöntemi

## 17.1 Teorik Özet (Markdown)
   - CLT varsayım gerektirmez → Bootstrap dağılım serbest
   - Yeniden örnekleme (resampling with replacement)
   - Bootstrap CI: Percentile method, BCa method
   - Kullanım: medyan, korelasyon, herhangi bir istatistik

## 17.2 [PYTHON] Bootstrap vs Analitik CI Karşılaştırması
   - Ortalama için: aynı sonuç
   - Medyan için: bootstrap kazanır

## 17.3 [PYTHON] Korelasyon Bootstrap CI
   - scipy.stats.bootstrap

## 17.4 [R] boot paketi

## 17.5 [SQL] Bootstrap analogu — TABLESAMPLE

## 17.6 Egzersiz: Gerçek veri ile medyan CI hesabı
```

**📦 Dataset:**
| Dataset | Kaynak | Kullanım |
|---|---|---|
| `seaborn.load_dataset("tips")` | Seaborn | Bahşiş medyan CI |
| Salary Data | [Kaggle](https://www.kaggle.com/datasets/mohithsairamreddy/salary-data) | Maaş medyan bootstrap |
| Medical Expenses | [Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance) | Sağlık harcama CI |

---

## 📚 Kaynak Haritası

### Kitaplar (Öncelik Sırasıyla)

| # | Kitap | Yazar | Ücretsiz? | Bağlantı |
|---|---|---|---|---|
| 🥇 | Probability for Data Science | Stanley H. Chan | ✅ | [probability4datascience.com](https://probability4datascience.com) |
| 🥇 | Think Stats | Allen Downey | ✅ | [greenteapress.com](https://greenteapress.com/thinkstats2/) |
| 🥈 | Practical Statistics for Data Scientists | Bruce & Gedeck | ❌ | O'Reilly |
| 🥈 | ISLP (Python baskısı) | James vd. | ✅ | [statlearning.com](https://www.statlearning.com) |
| 🥉 | Statistical Rethinking | McElreath | ❌ | CRC Press |
| 🥉 | Bayesian Methods for Hackers | Davidson-Pilon | ✅ | [GitHub](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers) |
| ⭐ | Elements of Statistical Learning | Hastie vd. | ✅ | [ESL](https://hastie.su.domains/ElemStatLearn/) |

### Kurslar

| Kurs | Platform | Konu |
|---|---|---|
| Harvard STAT 110 | [YouTube](https://youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo) | Genel olasılık |
| MIT 6.041SC | [OCW](https://ocw.mit.edu/courses/6-041sc-probabilistic-systems-analysis-and-applied-probability-fall-2013/) | Olasılık teorisi |
| Stanford CS109 | YouTube | CS için olasılık |
| Stanford CS229 | [cs229.stanford.edu](https://cs229.stanford.edu) | ML matematiği |
| MITx Probability | edX | Uygulamalı olasılık |

### Görselleştirme & Referans

| Site | Kullanım |
|---|---|
| [seeing-theory.brown.edu](https://seeing-theory.brown.edu) | İnteraktif dağılım görselleştirme |
| [distill.pub](https://distill.pub) | ML kavramları görsel açıklama |
| [betanalpha.github.io](https://betanalpha.github.io) | Bayesian teknik yazılar |
| [machinelearningmastery.com](https://machinelearningmastery.com) | ML odaklı istatistik |
