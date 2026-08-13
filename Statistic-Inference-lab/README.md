# Statistical Inference Lab — Tam Rehber
## Yapı · Taslak · Dataset · Kaynaklar

---

## 📁 Klasör Yapısı

```
statistical-inference-lab/
├── README.md
├── 01_hypothesis_testing/
│   ├── hypothesis_testing.ipynb
│   └── data/
├── 02_power_analysis/
│   ├── power_analysis.ipynb
│   └── data/
├── 03_regression_diagnostics/
│   ├── regression_diagnostics.ipynb
│   └── data/
├── 04_nonparametric_tests/
│   ├── nonparametric_tests.ipynb
│   └── data/
├── 05_multiple_testing/
│   ├── multiple_testing.ipynb
│   └── data/
└── 06_anova/
    ├── anova.ipynb
    └── data/
```

> **Hücre kuralı**: Her notebook'ta sırayla Python → R hücreleri 
> Magic: `%%R` (rpy2).  

---

## 📓 Notebook Taslakları & Dataset Önerileri

---

### 01 · Hipotez Testi (`hypothesis_testing.ipynb`)

```
# 1. Hipotez Testinin Temelleri

## 1.1 Teorik Özet (Markdown)
   - H0 (null) vs H1 (alternatif) hipotez kurulumu
   - Test istatistiği, p-value'nun DOĞRU yorumu (yaygın yanlış anlaşılmalar dahil)
   - Tip I Hata (α, yanlış pozitif) vs Tip II Hata (β, yanlış negatif)
   - Karar kuralı: p < α ise H0'ı reddet
   - Önizleme: güç (power) kavramı — detayı 02'de

## 1.2 [PYTHON] Tek Örneklemli t-test
   - scipy.stats.ttest_1samp
   - Örnek: bir ürün grubunun ortalama değeri, belirtilen standarttan farklı mı?

## 1.3 [PYTHON] İki Örneklemli t-test (Bağımsız Gruplar)
   - scipy.stats.ttest_ind (Welch's t-test: eşit olmayan varyans varsayımı)
   - A/B test bağlamı — probability-lab'daki Bayesian A/B testiyle KARŞILAŞTIR
     (aynı soruya iki farklı felsefeden - frekantist vs Bayesyen - cevap)

## 1.4 [PYTHON] Eşleştirilmiş (Paired) t-test
   - Öncesi/sonrası karşılaştırma senaryosu

## 1.5 [R] t.test() — tüm üç varyant
   - Python sonuçlarıyla birebir karşılaştırma

## 1.6 Egzersiz: Gerçek veride kendi hipotezini kur ve test et
```

**📦 Dataset:**
| Dataset | Kaynak | Kullanım |
|---|---|---|
| Students Performance in Exams | [Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams) | test prep course alan/almayan öğrencilerin matematik notu farkı → iki örneklemli t-test |
| `sleep` | R built-in (`data(sleep)`) | İki uyku ilacının aynı hastalarda etkisi → eşleştirilmiş (paired) t-test — klasik bir istatistik veri seti |
| World Happiness Report | [Kaggle](https://www.kaggle.com/datasets/unsdsn/world-happiness) | Bir bölgenin ortalama mutluluk skoru, dünya ortalamasından farklı mı → tek örneklemli t-test |

---

### 02 · Güç Analizi (`power_analysis.ipynb`)

```
# 2. İstatistiksel Güç ve Örneklem Büyüklüğü

## 2.1 Teorik Özet (Markdown)
   - Güç = 1 - β (gerçek bir etkiyi doğru tespit etme olasılığı)
   - Etki büyüklüğü (Effect Size, Cohen's d) — "farkın pratik önemi"
   - Güç, örneklem büyüklüğü, etki büyüklüğü, α arasındaki ilişki (4'ünden 3'ünü bilirsen 4.'yü bulursun)
   - Neden ÖNCEDEN hesaplanmalı (deney sonrası "post-hoc power" tartışmalı bir pratiktir)

## 2.2 [PYTHON] Cohen's d Hesaplama
   - statsmodels veya elle formül

## 2.3 [PYTHON] Gerekli Örneklem Büyüklüğünü Hesaplama
   - statsmodels.stats.power.TTestIndPower
   - "Bu etkiyi %80 güçle tespit etmek için kaç kişiye ihtiyacım var?"

## 2.4 [PYTHON] Güç Eğrisi Görselleştirme
   - Örneklem büyüklüğü arttıkça gücün nasıl arttığını çizme

## 2.5 [R] pwr paketi (pwr.t.test, pwr.anova.test)

## 2.6 Egzersiz: Kendi A/B testini tasarla — gereken min. örneklem büyüklüğünü hesapla
```

**📦 Dataset:**
| Dataset | Kaynak | Kullanım |
|---|---|---|
| Pima Indians Diabetes | [Kaggle](https://www.kaggle.com/datasets/mragpavank/diabetes) | Diyabetli/değil gruplarında Glucose farkının etki büyüklüğü → geriye dönük güç hesabı |
| Synthetic | `scipy.stats` | Kontrollü güç eğrisi simülasyonu (n değiştikçe gücün artışı) |

---

### 03 · Regresyon Tanılama (`regression_diagnostics.ipynb`)

```
# 3. OLS Varsayımları ve Regresyon Tanılaması

## 3.1 Teorik Özet (Markdown)
   - OLS'in 4 temel varsayımı: Doğrusallık, Bağımsızlık, Homoscedasticity, Normal Dağılan Hatalar
   - Multicollinearity (çoklu-doğrusal bağlantı) sorunu
   - HATIRLATMA: 08'de (MLE=OLS) bu varsayımların MATEMATİKSEL kökenini görmüştük -
     burada PRATİK olarak nasıl test edileceğini işliyoruz

## 3.2 [PYTHON] Residual Plot Analizi
   - Tahmin vs Residual grafiği — desen var mı (doğrusallık ihlali sinyali)

## 3.3 [PYTHON] Homoscedasticity Testi
   - Breusch-Pagan testi (statsmodels)

## 3.4 [PYTHON] Multicollinearity Tespiti
   - VIF (Variance Inflation Factor) hesaplama

## 3.5 [PYTHON] Normallik Kontrolü
   - Residual'lar için Q-Q plot + Shapiro-Wilk testi

## 3.6 [R] lm() + plot(model) — R'ın yerleşik 4'lü diagnostic plot çıktısı

## 3.7 Egzersiz: Bir regresyon modelini uçtan uca tanıla, hangi varsayım(lar) ihlal ediliyor?
```

**📦 Dataset:**
| Dataset | Kaynak | Kullanım |
|---|---|---|
| Auto MPG | [UCI](https://archive.ics.uci.edu/dataset/9/auto+mpg) | mpg ~ weight+horsepower+displacement modeli — klasik multicollinearity örneği (weight ve displacement birbirine çok bağlı) |
| California Housing | `sklearn.datasets.fetch_california_housing()` (built-in, indirme gerekmez) | Fiyat tahmininde heteroscedasticity — yedek/ikinci örnek |

---

### 04 · Parametrik Olmayan Testler (`nonparametric_tests.ipynb`)

```
# 4. Nonparametrik Testler

## 4.1 Teorik Özet (Markdown)
   - Ne zaman parametrik varsayımlar (normallik) bozulur
   - Rank-tabanlı testlerin mantığı (03'teki Spearman korelasyonuyla AYNI felsefe)
   - Parametrik ↔ Nonparametrik eşleştirme tablosu (t-test↔Mann-Whitney, ANOVA↔Kruskal-Wallis)

## 4.2 [PYTHON] Mann-Whitney U Testi
   - İki bağımsız grup, normallik varsayımı olmadan

## 4.3 [PYTHON] Wilcoxon Signed-Rank Testi
   - Eşleştirilmiş verinin nonparametrik versiyonu

## 4.4 [PYTHON] Kruskal-Wallis Testi
   - 3+ grup, ANOVA'nın nonparametrik versiyonu (06'ya köprü)

## 4.5 [R] wilcox.test(), kruskal.test()

## 4.6 Egzersiz: Çarpık bir değişkende t-test ile Mann-Whitney sonuçlarını karşılaştır
```

**📦 Dataset:**
| Dataset | Kaynak | Kullanım |
|---|---|---|
| Wine Quality (Red) | [Kaggle/UCI](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009) | `quality` skoru (çarpık, ordinal) — iki asitlik grubu arası Mann-Whitney |
| Absenteeism at Work | [UCI](https://archive.ics.uci.edu/dataset/445/absenteeism+at+work) | Departmanlara göre devamsızlık saatleri (aşırı çarpık) — Kruskal-Wallis |

---

### 05 · Çoklu Karşılaştırma Düzeltmesi (`multiple_testing.ipynb`)

```
# 5. Çoklu Karşılaştırma Problemi

## 5.1 Teorik Özet (Markdown)
   - Neden 20 test yaparsan en az biri "yanlışlıkla" anlamlı çıkar (α=0.05 ile)
   - Aile-bazlı hata oranı (FWER) vs False Discovery Rate (FDR)
   - Bonferroni (basit, muhafazakar) vs Benjamini-Hochberg (daha az muhafazakar, FDR kontrolü)

## 5.2 [PYTHON] Bonferroni Düzeltmesi
   - Elle uygulama + statsmodels.stats.multitest

## 5.3 [PYTHON] Benjamini-Hochberg (FDR) Düzeltmesi
   - Aynı veri, iki yöntemin sonucu ne kadar farklı çıkıyor

## 5.4 [PYTHON] Gerçek Senaryo: Çok Sayıda Grup/Özellik Karşılaştırması
   - Örneğin çok sayıda şehir/ülke/grup arası ikili karşılaştırmalar

## 5.5 [R] p.adjust(method="bonferroni"/"BH")

## 5.6 Egzersiz: Kendi çoklu karşılaştırma senaryonu — düzeltmeden önce/sonra kaç sonuç anlamlı kalıyor?
```

**📦 Dataset:**
| Dataset | Kaynak | Kullanım |
|---|---|---|
| Breast Cancer Wisconsin (Diagnostic) | `sklearn.datasets.load_breast_cancer()` (built-in, indirme gerekmez) | 30 özelliğin her biri için malignant/benign t-testi — 30 testin hepsini düzeltmeden yorumlarsan ne kadar yanıltıcı olur, klasik genomik-tarzı çoklu test senaryosu |
| FIFA 21/22 Player Ratings | [Kaggle](https://www.kaggle.com/datasets/bryanb/fifa-player-stats-database) | Çok sayıda ligin/ülkenin ikili karşılaştırması |

---

### 06 · ANOVA (`anova.ipynb`) — Opsiyonel/Düşük Öncelik

```
# 6. Varyans Analizi (ANOVA)

## 6.1 Teorik Özet (Markdown)
   - One-way ANOVA: 3+ grup ortalamasını aynı anda karşılaştırma
   - F-istatistiği: gruplar-arası varyans / grup-içi varyans oranı
   - HATIRLATMA: probability-lab 05'teki "Law of Total Variance" (grup içi + gruplar arası
     varyans ayrıştırması) ANOVA'nın matematiksel temeliydi — burada pratiğini görüyoruz
   - Neden çoklu t-test yerine ANOVA (05'teki çoklu karşılaştırma sorununa çözüm)

## 6.2 [PYTHON] One-way ANOVA
   - scipy.stats.f_oneway

## 6.3 [PYTHON] Two-way ANOVA
   - statsmodels ile iki faktörlü tasarım

## 6.4 [PYTHON] Post-hoc Test: Tukey HSD
   - ANOVA "en az bir grup farklı" der ama HANGİSİ olduğunu söylemez - Tukey bunu çözer

## 6.5 [R] aov() + TukeyHSD() + summary()

## 6.6 Egzersiz: Bölgeler arası çoklu grup karşılaştırması, ANOVA + Tukey HSD ile
```

**📦 Dataset:**
| Dataset | Kaynak | Kullanım |
|---|---|---|
| `PlantGrowth` | R built-in (`data(PlantGrowth)`) | Kontrol + 2 farklı gübre grubunun bitki ağırlığı — ANOVA'nın "ders kitabı" veri seti |
| `iris` | Seaborn/sklearn built-in | 3 türün sepal/petal uzunluğu karşılaştırması — two-way ANOVA'ya genişletilebilir |

---

## 📚 Kaynak Haritası

### Kitaplar

| # | Kitap | Yazar | Ücretsiz? | Bağlantı |
|---|---|---|---|---|
| 🥇 | Practical Statistics for Data Scientists | Bruce & Gedeck | ❌ | O'Reilly |
| 🥇 | Think Stats | Allen Downey | ✅ | [greenteapress.com](https://greenteapress.com/thinkstats2/) |
| 🥈 | ISLP (Python baskısı) | James vd. | ✅ | [statlearning.com](https://www.statlearning.com) |
| 🥉 | All of Statistics | Larry Wasserman | ❌ | Springer |

### Kurslar & Referans

| Kaynak | Kullanım |
|---|---|
| [Harvard STAT 110](https://youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo) | Hipotez testi bölümleri (probability-lab'dan tanıdık) |
| [seeing-theory.brown.edu](https://seeing-theory.brown.edu) | Frequentist Inference — interaktif |
| [statsmodels dokümantasyonu](https://www.statsmodels.org) | Bu lab'ın ana Python kütüphanesi |
| [pwr paketi (CRAN)](https://cran.r-project.org/web/packages/pwr/) | R'da güç analizi referansı |

---
