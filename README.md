# DS-Mastery-Journey

### Python & Data Science

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-4051B5?style=for-the-badge&logo=python&logoColor=white)

### Machine Learning & Scientific Computing

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![scikit--image](https://img.shields.io/badge/scikit--image-EC6B23?style=for-the-badge&logo=scikit-image&logoColor=white)
![SymPy](https://img.shields.io/badge/SymPy-3B5526?style=for-the-badge&logo=sympy&logoColor=white)
![Yahoo Finance](https://img.shields.io/badge/Yahoo%20Finance-6001D2?style=for-the-badge&logo=yahoo&logoColor=white)

### R & Statistical Computing

![R](https://img.shields.io/badge/R-276DC3?style=for-the-badge&logo=r&logoColor=white)
![Base R](https://img.shields.io/badge/Base%20R-276DC3?style=for-the-badge&logo=r&logoColor=white)
![rpy2](https://img.shields.io/badge/rpy2-276DC3?style=for-the-badge&logo=r&logoColor=white)

### SQL & Data Analytics

![SQL](https://img.shields.io/badge/SQL-336791?style=for-the-badge&logo=postgresql&logoColor=white)
![DuckDB](https://img.shields.io/badge/DuckDB-FFF000?style=for-the-badge&logo=duckdb&logoColor=black)


Bu depo, veri bilimi ve makine öğrenmesi alanında yapılandırılmış bir öğrenim sürecinin
ana merkezidir. Amaç, konuları yüzeysel biçimde "kullanmayı öğrenmek" değil, her yöntemin
arkasındaki matematiksel temeli anlayıp, aynı zamanda gerçek veri setleri üzerinde
üç farklı araçla (Python, R, SQL) pratik yapmaktır. Depo, olasılık ve istatistikten
başlayıp lineer cebir ve kalkülüse uzanan bir matematiksel temel aşamasından geçtikten
sonra makine öğrenmesi ve derin öğrenme konularına ilerleyecek şekilde tasarlanmıştır.

## Bu Depo Ne İşe Yarıyor

Depodaki her alt proje ("lab"), belirli bir konu alanını kapsayan bir Jupyter notebook
serisinden oluşuyor. Her notebook aynı sabit kalıbı izliyor: önce teorik özet (formüllerin
sadece yazılması değil, sayısal örneklerle adım adım türetilmesi), ardından gerçek veri
setleri üzerinde uygulama, ardından çözümü okuyucuya bırakılan egzersizler, ve son olarak
ileri okuma için kaynak listesi. Notebook'lardaki her iddia (bir formülün doğruluğu, bir
görselin doğru sonucu verdiği, iki farklı yöntemin aynı sonuca ulaştığı) kod çalıştırılarak
doğrulanmış durumda; teorik bir cümle "muhtemelen doğrudur" değil, "gerçek veriyle test
edildi ve doğrulandı" ilkesiyle yazılmıştır.

Bu depo aynı zamanda bir referans kaynağı olarak da kullanılabilir: ileride bir formülü
unuttuğunda veya bir yöntemin neden çalıştığını hatırlamak istediğinde, ilgili notebook'a
dönüp hem teoriyi hem çalışan kodu bir arada bulabilirsin.

## Genel Yapı

Depo kök dizininde, Faz 1'i oluşturan dört matematiksel temel projesi bulunuyor:

- `Probability-lab` — olasılık teorisi ve istatistiksel çıkarımın temelleri
- `Statistic-Inference-lab` — klasik frekantist istatistik yöntemleri
- `Linear-Algebra-lab` — lineer cebir ve makine öğrenmesindeki uygulamaları
- `Calculus-lab` — kalkülüs ve optimizasyonun matematiksel temeli

Bunların dışında `Kaggle-Examples` (bağımsız Kaggle proje çalışmaları), `OOP-Examples`
(nesne yönelimli programlama alıştırmaları) ve `Documents` (çeşitli notlar) klasörleri de
depoda yer alıyor; bunlar yukarıdaki dört lab'ın kapsamı dışında, ayrı çalışmalardır.

## Faz 1: Matematiksel Temeller

### Probability-lab 

Olasılık teorisinin temellerinden başlayıp Bayes çıkarımına, parametre tahmine, bilgi
teorisine ve simülasyon yöntemlerine uzanan 17 notebook'luk bir seri:

- Temel dağılımlar: PMF/PDF/CDF, Bernoulli, Binomial, Poisson, Geometrik, Normal, Uniform,
  Exponential, Gamma, Chi-Square, Beta, Log-Normal
- Çok değişkenli olasılık: ortak/marjinal/koşullu dağılımlar, kovaryans ve korelasyon
- Beklenti, varyans ve bunların özellikleri
- Limit teoremleri: Büyük Sayılar Yasası ve Merkezi Limit Teoremi
- Bayes teoremi ve eşlenik priorlar
- Parametre tahmini: maksimum olabilirlik (MLE) ve maksimum a posteriori (MAP) tahmini
- Bilgi teorisi: entropi, çapraz entropi, Kullback-Leibler diverjansı
- Simülasyon yöntemleri: Monte Carlo ve bootstrap

Her notebook'ta Python, R ve SQL (DuckDB) uygulamaları birlikte yer alıyor; aynı analiz
üç farklı araçla tekrarlanıp sonuçların birbirini doğrulaması sağlanıyor.

### Statistic-Inference-lab 

Probability-lab'ın bıraktığı yerden devam edip klasik frekantist istatistiğin geri kalan
araç kutusuna odaklanan 6 notebook'luk bir seri:

- Hipotez testi: tek örneklemli, iki örneklemli ve eşleştirilmiş t-testleri
- İstatistiksel güç analizi ve örneklem büyüklüğü hesabı
- Regresyon tanılaması: OLS varsayımları, çoklu doğrusal bağlantı (multicollinearity),
  homoscedasticity, residual analizi
- Parametrik olmayan testler: Mann-Whitney U, Wilcoxon işaretli-sıra testi, Kruskal-Wallis
- Çoklu karşılaştırma düzeltmesi: Bonferroni ve Benjamini-Hochberg (FDR) yöntemleri
- Varyans analizi (ANOVA): tek yönlü, iki yönlü ve post-hoc karşılaştırma yöntemleri

Bu seride SQL kullanılmıyor (konular SQL'e doğal olarak oturmuyor); her notebook Python
ve R uygulamalarını birlikte içeriyor.

### Linear-Algebra-lab 

Makine öğrenmesi algoritmalarının matematiksel omurgasını oluşturan lineer cebir
kavramlarını, hem soyut örneklerle hem gerçek veri setleriyle işleyen 5 notebook'luk bir
seri:

- Vektör ve matris temelleri: temel işlemler, matris çarpımının geometrik dönüşüm olarak
  yorumu, determinant, matris tersi
- Özdeğerler ve özvektörler: karakteristik denklem, geometrik yorum, kovaryans matrisi
  bağlantısı
- Tekil değer ayrıştırması (SVD) ve temel bileşen analizi (PCA): sıfırdan kurulumu,
  görüntü sıkıştırma uygulaması, boyut indirgeme
- Normlar ve regularizasyon: L1/L2 normlarının geometrisi, Ridge ve Lasso regresyonun
  neden farklı davrandığının geometrik ispatı
- Özel matrisler: pozitif tanımlılık, kovaryans matrisinin bu özelliği neden her zaman
  taşıdığının cebirsel ispatı, konvekslik ve optimizasyonla bağlantısı

Bu seri sadece Python ile ilerliyor; her notebook'ta konuya özel, gerçek dünyadan bir veri
seti kullanılıyor (finansal getiri verisi, çalışan verisi, ikinci el araç fiyatları,
hava kalitesi ölçümleri gibi).

### Calculus-lab 

Faz 1'in son adımı. Beş notebook planlandı ama henüz yazılmadı:

- Türevler ve gradyan vektörü
- Jacobian, Hessian matrisleri ve konvekslik
- Zincir kuralı ve geri yayılım (backpropagation)
- Gradient descent ve optimizasyon algoritmaları (SGD, Momentum, Adam)
- Taylor serisi ve ikinci dereceden optimizasyon yöntemleri

## Kullanılan Kütüphaneler ve Araçlar

**Python**, tüm serilerin ortak dili. Sayısal hesaplama için ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white) ve ![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white), veri
işleme için ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white), görselleştirme için ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white) ve ![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white) kullanılıyor.
İstatistiksel modelleme ![Statsmodels](https://img.shields.io/badge/Statsmodels-4051B5?style=for-the-badge&logo=python&logoColor=white) ile, makine öğrenmesi bileşenleri (PCA, Ridge, Lasso,
regresyon modelleri) ![Scikit--learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) ile yapılıyor. Görüntü işleme örnekleri için
![scikit--image](https://img.shields.io/badge/scikit--image-EC6B23?style=for-the-badge&logo=scikit-image&logoColor=white), finansal veri için ![Yahoo Finance](https://img.shields.io/badge/Yahoo%20Finance-6001D2?style=for-the-badge&logo=yahoo&logoColor=white), ve calculus-lab'da sembolik türev alma
için ![SymPy](https://img.shields.io/badge/SymPy-3B5526?style=for-the-badge&logo=sympy&logoColor=white) ile otomatik türev alma için ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) kullanılması planlanıyor.

**R**, istatistiksel testler için Python'a paralel bir uygulama katmanı sağlıyor.
Mümkün olduğunca R'a yerleşik (base R) fonksiyonlar tercih edildi — `t.test`, `aov`,
`TukeyHSD`, `wilcox.test`, `kruskal.test`, `power.t.test`, `p.adjust`, `lm`, `dbeta`,
`dgamma` gibi. Bunun sebebi, harici bir R paketinin kullanıcının ortamında kurulu olduğu
garanti edilemediği için, notebook'ların her ortamda çalışabilir kalmasını sağlamak.
R'a geçiş, Jupyter içinde ![rpy2](https://img.shields.io/badge/rpy2-276DC3?style=for-the-badge&logo=r&logoColor=white) kütüphanesinin `%%R` hücre sihirbazı ile yapılıyor.

![SQL](https://img.shields.io/badge/SQL-336791?style=for-the-badge&logo=postgresql&logoColor=white), probability-lab serisinde DuckDB üzerinden kullanıldı — özellikle örnekleme
simülasyonları, koşullu olasılık hesapları ve büyük veri dosyalarını (bazı örneklerde
100 megabaytın üzerinde) pandas'a hiç yüklemeden doğrudan sorgulamak için.

## Notebook Formatı

Her notebook, konudan bağımsız olarak aynı yapıyı izliyor:

1. Başlık ve genel özet — notebook'un neyi kapsadığı, önkoşulları, kullanılan veri setleri
2. Teorik bölüm — formüllerin türetilmesi, mümkün olduğunca küçük sayısal örneklerle elle
   hesaplama, ardından kodla doğrulama
3. Uygulama bölümü — gerçek veri setleri üzerinde, dile göre sıralı olarak (önce Python,
   sonra R, varsa sonra SQL) aynı analizin tekrarlanması
4. Egzersizler — çözümü verilmeyen, okuyucunun kendisinin çözmesi beklenen sorular
5. Kaynaklar — konuyla ilgili kitap, ders ve makale önerileri

Her notebook'taki veri iddiaları (bir formülün elle hesaplanan sonucu ile kütüphane
fonksiyonunun ürettiği sonucun birebir örtüştüğü gibi) kod çalıştırılarak test edilmiş
ve doğrulanmıştır.

## Sırada Ne Var

Calculus-lab tamamlandığında Faz 1 (matematiksel temeller) bitmiş olacak. Sonrasında
Faz 2, makine öğrenmesi yöntemlerinin derinlemesine işlenmesiyle devam edecek.