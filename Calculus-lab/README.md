# Calculus Lab — Tam Rehber

## Yapı · Taslak · Dataset · Kaynaklar

---

## 📁 Klasör Yapısı

```
calculus-lab/
├── README.md
├── 01_derivatives_gradients/
│   ├── derivatives_gradients.ipynb
│   └── data/
├── 02_jacobian_hessian_convexity/
│   ├── jacobian_hessian_convexity.ipynb
│   └── data/
├── 03_chain_rule_backprop/
│   ├── chain_rule_backprop.ipynb
│   └── data/
├── 04_gradient_descent_optimizers/
│   ├── gradient_descent_optimizers.ipynb
│   └── data/
└── 05_taylor_series/
    ├── taylor_series.ipynb
    └── data/
```

---

## 📓 Notebook Taslakları & Dataset Önerileri

---

### 01 · Türevler ve Gradyanlar (`derivatives_gradients.ipynb`)

```
# 1. Türev, Kısmi Türev ve Gradyan Vektörü

## 1.1 Teorik Özet (Markdown)
   - Türev: bir fonksiyonun bir noktadaki ANLIK değişim hızı (eğim)
   - Kısmi türev: çok değişkenli bir fonksiyonda, SADECE bir değişkeni değiştirip
     diğerlerini sabit tutarsan ne olur
   - Gradyan vektörü (∇f): tüm kısmi türevlerin bir araya toplandığı vektör — fonksiyonun
     EN DİK ARTIŞ yönünü gösterir
   - Sayısal (numerical) türev vs analitik (kapalı-form) türev — ikisini karşılaştırma

## 1.2 [PYTHON] Temel Türev Kuralları + `sympy` ile Sembolik Doğrulama
   - Polinom, üstel, logaritmik fonksiyonların türevini elle al, sympy ile doğrula

## 1.3 [PYTHON] Sayısal Türev: Sonlu Farklar (Finite Differences)
   - f'(x) ≈ [f(x+h)-f(x)]/h formülüyle türevi YAKLAŞIK hesaplama
   - h küçüldükçe analitik türeve nasıl yakınsadığını gösterme

## 1.4 [PYTHON] Gerçek Veride Gradyan: COVID-19 Vaka Artış Hızı
   - Günlük vaka sayısından, "artış HIZINI" (türevi) sayısal olarak hesaplama
   - Artış hızının kendisinin de nasıl değiştiğini (ikinci türev — ivmelenme/yavaşlama) gösterme

## 1.5 [PYTHON] Çok Değişkenli Fonksiyonda Gradyan Vektörü
   - f(x,y) = x²+y² gibi basit bir yüzeyde, farklı noktalardaki gradyan vektörlerini
     3D yüzey üzerinde ok olarak çizme (en dik artış yönü görsel kanıtı)

## 1.6 Egzersiz: Başka bir gerçek zaman serisinde (kendi seçimin) günlük değişim hızını hesapla
```

**📦 Dataset:**

| Dataset                       | Kaynak                                                                                                                      | Kullanım                                                   |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| COVID-19 Günlük Vaka Verisi | [Kaggle — Novel Corona Virus 2019 Dataset](https://www.kaggle.com/datasets/sudalairajkumar/novel-corona-virus-2019-dataset) | Günlük artış hızı = türev, ivmelenme = ikinci türev |
| Sentetik 2D yüzeyler         | `numpy`/`sympy`                                                                                                         | Gradyan vektörü görselleştirmesi                        |

---

### 02 · Jacobian, Hessian ve Konvekslik (`jacobian_hessian_convexity.ipynb`)

```
# 2. Jacobian, Hessian ve Optimizasyonda Konvekslik

## 2.1 Teorik Özet (Markdown)
   - Jacobian: ÇOK ÇIKTILI bir fonksiyonun tüm kısmi türevlerinin matrisi (gradyanın genellemesi)
   - Hessian: İKİNCİ dereceden kısmi türevlerin matrisi — fonksiyonun "eğriliğini" ölçer
   - HATIRLATMA: `linear-algebra-lab` 05'te Hessian'ın pozitif tanımlı olması gerektiğini
     söylemiştik (konvekslik için) — burada Hessian'ı GERÇEKTEN hesaplayıp bunu doğruluyoruz
   - Konveks fonksiyon: TEK BİR global minimuma sahip (yerel minimum tuzağı yok) —
     optimizasyon için "en iyi" durum

## 2.2 [PYTHON] Jacobian Hesaplama (`sympy` + Elle)
   - İki çıktılı basit bir fonksiyonun Jacobian matrisini elle ve sembolik olarak bulma

## 2.3 [PYTHON] Hessian Hesaplama ve Konvekslik Testi
   - Bir fonksiyonun Hessian'ını hesaplayıp özdeğerlerinin (linear-algebra-lab 05'teki
     testin AYNISI) pozitif olup olmadığını kontrol etme

## 2.4 [PYTHON] Gerçek Veride Konvekslik: Lojistik Regresyon Loss Yüzeyi
   - Heart Disease verisinde lojistik regresyonun log-likelihood'unun (negatifinin)
     Hessian'ını hesaplayıp konveks olduğunu doğrulama — bu NEDEN lojistik regresyonun
     her zaman tek bir global optimuma yakınsadığını açıklıyor

## 2.5 [PYTHON] Konveks OLMAYAN Bir Yüzey: Yerel Minimum Tuzağı
   - Konveks olmayan bir fonksiyon (örn. derin öğrenme loss yüzeylerine benzeyen, birden
     fazla "çukuru" olan bir yüzey) üzerinde gradient descent'in başlangıç noktasına göre
     FARKLI yerel minimumlara takılabileceğini gösterme

## 2.6 Egzersiz: Başka bir 2 değişkenli fonksiyonun Hessian'ını hesaplayıp konveks olup
   olmadığını belirle
```

**📦 Dataset:**

| Dataset                                                                    | Kaynak              | Kullanım                                                            |
| -------------------------------------------------------------------------- | ------------------- | -------------------------------------------------------------------- |
| [Heart Disease (UCI)](https://archive.ics.uci.edu/dataset/45/heart+disease) | UCI                 | Lojistik regresyon loss yüzeyinin Hessian'ı → konvekslik kanıtı |
| Sentetik yüzeyler                                                         | `numpy`/`sympy` | Konveks vs konveks-olmayan yüzey karşılaştırması               |

---

### 03 · Zincir Kuralı ve Backpropagation (`chain_rule_backprop.ipynb`)

```
# 3. Zincir Kuralı: Derin Öğrenmenin Matematiksel Kalbi

## 3.1 Teorik Özet (Markdown)
   - Zincir kuralı: d/dx[f(g(x))] = f'(g(x))·g'(x) — iç içe fonksiyonların türevi
   - Neden önemli: bir sinir ağı, katman katman iç içe geçmiş fonksiyonlardır
   - Backpropagation = zincir kuralının, ağın SONUNDAN BAŞINA doğru sistematik uygulanması
   - İleri yayılım (forward pass) vs geri yayılım (backward pass)

## 3.2 [PYTHON] Zincir Kuralını Elle Uygulama (Basit Örnek)
   - f(x) = (3x+1)² gibi basit bir bileşik fonksiyonun türevini ELLE (zincir kuralıyla) al,
     sympy ile doğrula

## 3.3 [PYTHON] Tek Nöronlu Bir Ağda Backprop'u ELLE Türetme
   - Titanic verisinde TEK bir özellikten (örn. yaş) TEK bir çıktıya (hayatta kalma
     olasılığı) giden minik bir sinir ağı (1 nöron, sigmoid aktivasyon) kur
   - Loss'tan ağırlığa giden gradyanı ADIM ADIM elle türet (zincir kuralı uygulaması)

## 3.4 [PYTHON] Gerçek Bir Mini Ağda Backprop: Kütüphanesiz vs PyTorch
   - Titanic verisinde 2 katmanlı minik bir ağı NumPy ile (backprop'u elle kodlayarak) eğit
   - Aynı ağı PyTorch'un `autograd`'ıyla eğit — İKİ YÖNTEMİN aynı gradyanları ürettiğini doğrula

## 3.5 Egzersiz: Ağa bir katman daha ekle, zincirin nasıl uzadığını (ve gradyanın nasıl
   hesaplandığını) göster
```

**📦 Dataset:**

| Dataset     | Kaynak                                         | Kullanım                                                             |
| ----------- | ---------------------------------------------- | --------------------------------------------------------------------- |
| `titanic` | `seaborn.load_dataset("titanic")` (built-in) | Minik sinir ağı ile hayatta kalma tahmini, backprop'u elle türetme |

---

### 04 · Gradient Descent ve Optimizerlar (`gradient_descent_optimizers.ipynb`)

```
# 4. Gradient Descent, SGD, Momentum, Adam

## 4.1 Teorik Özet (Markdown)
   - Gradient descent: θ_yeni = θ_eski - α·∇f(θ) — gradyanın TERSİ yönünde küçük adımlar
   - Öğrenme oranı (α) çok büyükse ıraksama, çok küçükse çok yavaş yakınsama
   - SGD (Stochastic): her adımda TÜM veri yerine küçük bir alt-örneklem kullanma
   - Momentum: önceki adımların "hızını" koruyarak yerel çukurlardan daha kolay çıkma
   - Adam: momentum + adaptif öğrenme oranını birleştiren modern standart

## 4.2 [PYTHON] Gradient Descent'i Sıfırdan Kodlama
   - Basit bir 2D fonksiyonda (örn. bir "vadi" şeklinde) gradient descent'i adım adım
     çalıştırıp, her adımı yüzey üzerinde bir nokta olarak çizme (yörünge görselleştirmesi)

## 4.3 [PYTHON] Öğrenme Oranının Etkisi
   - Çok küçük / uygun / çok büyük öğrenme oranlarıyla AYNI fonksiyonda gradient descent
     çalıştırıp yörüngelerin nasıl farklılaştığını (yavaş yakınsama / düzgün yakınsama /
     ıraksama) gösterme

## 4.4 [PYTHON] SGD vs Momentum vs Adam Karşılaştırması
   - Gerçek bir regresyon probleminde (Enerji Verimliliği verisi) üç optimizer'ı da
     uygulayıp, loss'un epoch'lara göre nasıl azaldığını aynı grafikte karşılaştırma

## 4.5 [PYTHON] Zorlu Bir Yüzeyde Optimizer Karşılaştırması
   - Dar, eğri bir "vadi" (Rosenbrock fonksiyonu gibi klasik bir test yüzeyi) üzerinde
     SGD'nin zorlanıp Momentum/Adam'ın daha hızlı yakınsadığını gösterme

## 4.6 Egzersiz: Kendi öğrenme oranı/optimizer kombinasyonunu dene, en hızlı yakınsayanı bul
```

**📦 Dataset:**

| Dataset                                                                                     | Kaynak    | Kullanım                                                               |
| ------------------------------------------------------------------------------------------- | --------- | ----------------------------------------------------------------------- |
| [Energy Efficiency Dataset (UCI)](https://archive.ics.uci.edu/dataset/242/energy+efficiency) | UCI       | Bina enerji tüketimi tahmininde SGD/Momentum/Adam karşılaştırması |
| Sentetik yüzeyler (Rosenbrock fonksiyonu)                                                  | `numpy` | Optimizer'ların zorlu yüzeylerdeki davranış farkları               |

---

### 05 · Taylor Serisi (`taylor_series.ipynb`)

```
# 5. Taylor Serisi: Fonksiyonu Polinomla Yaklaşıklama

## 5.1 Teorik Özet (Markdown)
   - Taylor serisi: karmaşık bir fonksiyonu, bir noktadaki türevlerini kullanarak
     polinomla YAKLAŞIK ifade etme
   - Formül: f(x) ≈ f(a) + f'(a)(x-a) + f''(a)(x-a)²/2! + ...
   - Kaç terim eklersen o kadar "iyi" yaklaşık (ama sonsuza kadar mükemmel eşitlik değil,
     yakınsama yarıçapı var)
   - ML bağlantısı: bazı optimizasyon algoritmaları (Newton's Method) Taylor'ın İKİNCİ
     dereceden yaklaşıklamasını kullanıyor

## 5.2 [PYTHON] Taylor Yaklaşıklamasını Görselleştirme
   - sin(x), e^x gibi bilinen fonksiyonları farklı derecelerde (1, 3, 5, 7 terim) Taylor
     polinomuyla yaklaşıklayıp gerçek fonksiyonla üst üste çizme — derece arttıkça
     yaklaşıklığın nasıl iyileştiğini gösterme

## 5.3 [PYTHON] Gerçek Veride Taylor Yaklaşıklaması: Nüfus Büyümesi
   - Dünya nüfus verisine üstel bir büyüme modeli (e^x tarzı) fit et, bu modeli düşük
     dereceden bir Taylor polinomuyla yaklaşıklayıp gerçek veriyle karşılaştır

## 5.4 [PYTHON] Newton's Method: İkinci Dereceden Taylor'ın Optimizasyonda Kullanımı
   - Bir fonksiyonun kökünü (veya minimumunu) Newton's Method ile bulma — 4'teki gradient
     descent'ten NEDEN daha hızlı yakınsadığını (ikinci türev bilgisini kullandığı için) gösterme

## 5.5 Egzersiz: Kendi seçtiğin bir fonksiyonu 3 farklı derecede Taylor polinomuyla
   yaklaşıklayıp hata payını karşılaştır
```

**📦 Dataset:**

| Dataset                                                                                               | Kaynak              | Kullanım                                          |
| ----------------------------------------------------------------------------------------------------- | ------------------- | -------------------------------------------------- |
| [World Population Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/world-population-dataset) | Kaggle              | Üstel büyüme modelinin Taylor yaklaşıklaması |
| Bilinen fonksiyonlar (sin, exp)                                                                       | `numpy`/`sympy` | Taylor serisi yakınsama görselleştirmesi        |

---

## 📚 Kaynak Haritası

### Kitaplar

| #  | Kitap                                          | Yazar                         | Ücretsiz? | Bağlantı                                                                             |
| -- | ---------------------------------------------- | ----------------------------- | ---------- | -------------------------------------------------------------------------------------- |
| 🥇 | Calculus                                       | Gilbert Strang                | ✅         | [MIT OCW](https://ocw.mit.edu/courses/res-18-001-calculus-online-textbook-spring-2005/) |
| 🥇 | Mathematics for Machine Learning               | Deisenroth, Faisal, Ong       | ✅         | [mml-book.github.io](https://mml-book.github.io)                                        |
| 🥈 | Deep Learning (Ch. 4 — Numerical Computation) | Goodfellow, Bengio, Courville | ✅         | [deeplearningbook.org](https://www.deeplearningbook.org)                                |
| 🥉 | Convex Optimization                            | Boyd & Vandenberghe           | ✅         | [web.stanford.edu/~boyd/cvxbook/](https://web.stanford.edu/~boyd/cvxbook/)              |

### Kurslar & Görsel Referans

| Kaynak                                                                                                     | Kullanım                                                  |
| ---------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| [3Blue1Brown — Essence of Calculus](https://www.3blue1brown.com/topics/calculus)                           | Bu lab'ın TEK EN ÖNEMLİ referansı                      |
| [3Blue1Brown — Neural Networks (Backprop bölümleri)](https://www.3blue1brown.com/topics/neural-networks) | Zincir kuralı/backprop için birebir görsel anlatım     |
| [Stanford CS231n — Backpropagation Notları](https://cs231n.github.io/optimization-2/)                     | Backprop'un "elle türetme" pratiği için klasik referans |
| [distill.pub — Momentum](https://distill.pub/2017/momentum/)                                               | Momentum/Adam'ın görsel/interaktif anlatımı            |
| [PyTorch `autograd` Dokümantasyonu](https://pytorch.org/docs/stable/autograd.html)                       | Bu lab'ın 03-04'te kullanılan ana aracı                 |

---
