# 05 · Denetimsiz Öğrenme (Unsupervised Learning)

`Classical-ML-lab` serisinin beşinci kategorisi (23 notebook'luk serinin E
bölümü, roadmap'teki notebook'lar 16-20). 01-15 arası notebook'ların hepsi
**denetimli** öğrenmeydi: her örnek için bir hedef etiket ($y$) vardı ve
$X \to y$ eşlemesini öğreniyorduk. Bu klasörde etiket **yok** — elimizde
yalnızca $X$ var ve verinin kendi içindeki yapıyı soruyoruz: *hangi örnekler
bir arada duruyor* (`16`, `17`), *veriyi az sayıda eksenle nasıl özetler ve
görselleştiririz* (`18`), *hangi gözlem diğerlerine benzemiyor* (`19`) ve
*hangi şeyler birlikte görülüyor* (`20`). Notebook'lar birbirinin üzerine
inşa edildi — `17`, `16`'daki K-Means'in iki sınırlamasını (k'yı önceden
söyleme zorunluluğu, küresel küme varsayımı) çözen yöntemleri anlatıyor; `18`,
kümelemede gördüğümüz uzaklık/komşuluk fikrini yüksek boyutlu veriye taşıyıp
embedding üzerinde kümelemeyi mümkün kılıyor; `19`, `16`-`17`'deki yoğunluk
fikrini (LOF) ve `12`'deki PR-AUC/eşik disiplinini anomali tespitinde
kullanıyor; `20` ise etiketsiz sepet verisinden kural çıkarırken `11`'deki
hold-out mantığını "kural gelecekte de geçerli mi?" sorusuna uyguluyor.

## 📁 Klasör Yapısı

```
05_unsupervised_learning/
├── README.md
├── 16_clustering_partitioning.ipynb
├── 17_clustering_hierarchical_density.ipynb
├── 18_dimensionality_reduction_advanced.ipynb
├── 19_anomaly_detection.ipynb
├── 20_association_rule_learning.ipynb
└── data/
    ├── mall_customers.csv          ├── cardio.csv
    ├── wholesale_customers.csv     ├── letter.csv
    ├── usarrests.csv               ├── arrhythmia.csv
    ├── quakes.csv                  ├── nyc_taxi.csv
    ├── faithful.csv                ├── machine_temp.csv
    ├── fashion_mnist_test.npz      ├── nab_windows.json
    ├── olivetti_faces.npz          ├── groceries.csv
    ├── newsgroups_6cat.csv         ├── online_retail_uk.csv.gz
    └── mushrooms.csv
```

> **Hücre kuralı**: Her notebook'ta sırayla Python → R → SQL (DuckDB) uygulaması.
> R hücreleri `%%R` (rpy2) sihirbazıyla çalışıyor. Proje kararı gereği
> (2026-09-08) R ve SQL bölümleri bilinçli olarak kısa tutuldu (R: 1-3 hücre,
> SQL: 1-2 hücre) — ana teknik derinlik Python tarafında; R/SQL yalnızca konunun
> gerçekten aydınlatıcı olduğu yerde (`mclust`, `isotree`, `arules`, self-join
> ile kural madenciliği vb.) derinleştirildi. Çekirdek algoritmalar (K-Means,
> PAM, Agglomerative, DBSCAN, GMM-EM, PCA, LDA, Kernel PCA, NMF, FastICA,
> t-SNE, Isolation Forest, LOF, Apriori, Eclat, FP-Growth) önce **sıfırdan**
> yazılıp `sklearn`/`scipy`/`mlxtend` ile doğrulandı.

---

## 📓 16 · Kümeleme - Bölümleme Yöntemleri (`16_clustering_partitioning.ipynb`)

Denetimsiz öğrenmeye giriş: kümeleme problemi, **K-Means**'in amaç fonksiyonu
ve Lloyd algoritması, rastgele vs `k-means++` başlatma, küme sayısı $k$'yı
Elbow ve Silhouette ile seçme, aykırı değerlere dayanıklı **K-Medoids (PAM)**
ve büyük veri için **Mini-Batch K-Means**. K-Means'in küresel-olmayan
kümelerdeki başarısızlığı (ay şekli) ve bir sonraki notebook'a köprü.

```
1. Denetimsiz Öğrenme ve Kümeleme Problemi
2. K-Means: Amaç Fonksiyonu ve Lloyd Algoritması
3. Başlatma Stratejisi: Rastgele vs k-means++
4. Küme Sayısını (k) Seçmek: Elbow Method ve Silhouette Score
5. K-Medoids (PAM): K-Means Ne Zaman Yetersiz Kalır?
6. Mini-Batch K-Means: Büyük Veride Ölçeklenebilirlik
7. Özet Karşılaştırma ve Bir Sonraki Adım

🐍 Python: 15 uygulama adımı — Mall Customers'ta K-Means'in sıfırdan
   implementasyonu (`SimpleKMeans`, Lloyd), rastgele vs k-means++ yakınsama
   kararlılığı, `sklearn.KMeans` ile doğrulama, Elbow + Silhouette (elle +
   `sklearn`) ile k seçimi, final segmentasyon ve iş-odaklı küme profilleri;
   Wholesale Customers'ta K-Means'in aykırı değerlere duyarlılığı deneyi,
   PAM'ın sıfırdan implementasyonu, PAM vs K-Means dayanıklılık karşılaştırması,
   `sklearn_extra.KMedoids` ile doğrulama; büyük sentetik veride (`make_blobs`)
   Mini-Batch K-Means hız/kalite karşılaştırması, ay şeklinde ortak sınırlama
   demosu, üç yöntemin sentezi
📊 R: `stats::kmeans` (Mall) ve `cluster::pam` (Wholesale)
🗄️ SQL (DuckDB): K-Means'in atama adımının saf SQL'de yeniden üretimi
   (`CROSS JOIN` + uzaklık + `ROW_NUMBER() OVER (PARTITION BY ...)`) —
   `sklearn`'ün `labels_`'ı ile eşleştirildi
✏️ Egzersizler: E1–E6 (Python) — **çözülerek** (kendi çözümlerinle) bulunuyor
```

**📦 Veri Setleri:**

| Dataset | Kaynak | Kullanım |
| --- | --- | --- |
| Mall Customer Segmentation | [Kaggle (vjchoudhary7)](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) | K-Means, Elbow/Silhouette, iş-odaklı segmentasyon (200 müşteri, gelir + harcama skoru) |
| Wholesale Customers | [UCI](https://archive.ics.uci.edu/dataset/292/wholesale+customers), Cardoso (2014) | Aykırı değer duyarlılığı, K-Medoids (440 müşteri, 6 harcama kategorisi, ağır çarpık) |
| Sentetik (`make_blobs`, `make_moons`) | `sklearn.datasets` | Mini-Batch ölçeklenebilirlik testi, küresel-olmayan küme sınırı |

---

## 📓 17 · Kümeleme - Hiyerarşik ve Yoğunluk Tabanlı Yöntemler (`17_clustering_hierarchical_density.ipynb`)

`16`'da görülen iki K-Means sınırlamasına üç farklı cevap: **hiyerarşik
(agglomerative)** kümeleme ağaç (dendrogram) üretip k'yı sonradan seçtirir;
**DBSCAN** kümeyi "yoğun bölge" olarak tanımlayıp küme sayısını kendisi bulur
ve gürültüyü ayrıca etiketler; **GMM** ise kesin atama yerine olasılıksal
("yumuşak") atama yapar. Her yöntem önce elle hesaplanabilir küçük bir örnekle,
sonra gerçek veriyle işleniyor; kümeleme **metrikleri** (iç/dış) de bu
notebook'ta toplanıyor.

```
1. Neden K-Means Yetmedi? Üç Farklı Cevap
2. Hiyerarşik (Agglomerative) Kümeleme — linkage türleri, dendrogram okuma,
   kofenetik korelasyon, karmaşıklık ve sınırlar
3. DBSCAN — ε/minPts, çekirdek/sınır/gürültü noktaları, k-uzaklık grafiği,
   güçlü ve zayıf yanlar, OPTICS ve HDBSCAN'e kısa bakış
4. Gaussian Mixture Model (GMM) — sorumluluk, EM algoritması, kovaryans
   türleri, BIC/AIC ile model seçimi
5. Kümeleme Metrikleri: Kümelemeyi Nasıl Değerlendiririz?
6. Özet: Hangi Yöntemi Ne Zaman Seçmeli?

🐍 Python: 16 uygulama adımı — USArrests'te sıfırdan 4 linkage'li
   agglomerative (SciPy ile doğrulanmış), dendrogram + ağaç kesme, linkage
   karşılaştırması (zincirleme etkisi), kofenetik korelasyon + silhouette;
   `quakes`'te DBSCAN'in 1B elle örneği + sıfırdan implementasyonu, haversine
   mesafeli k-uzaklık grafiği ile ε seçimi, ε × minPts duyarlılık ızgarası,
   farklı yoğunluklu kümelerde DBSCAN'in zayıflığı + OPTICS/HDBSCAN;
   `faithful`'da GMM sorumluluğunun elle hesabı, EM'in sıfırdan implementasyonu
   (`GaussianMixture` ile doğrulama), yumuşak atama vs K-Means, kovaryans türü ×
   bileşen sayısı BIC/AIC ızgarası, GMM'in üretken yönü; 5 yöntem × 5 veri
   şekli sentezi ve kümeleme metrikleri
📊 R: `hclust` + `cutree` (USArrests) ve `mclust` (faithful — kovaryans türü
   ve bileşen sayısını BIC ile otomatik seçen referans GMM paketi)
🗄️ SQL (DuckDB): DBSCAN çekirdek noktalarının self-join (`CROSS JOIN` +
   `GROUP BY` + `HAVING COUNT(*) >= minPts`, haversine `sin/cos/asin` ile) ve
   GMM sorumluluklarının kapalı-form 2B Gauss yoğunluğuyla yeniden üretimi
✏️ Egzersizler: E1–E6 (Python) — **çözülmeden** (TODO'lu) teslim edildi
```

**📦 Veri Setleri:**

| Dataset | Kaynak | Kullanım |
| --- | --- | --- |
| USArrests | [Rdatasets](https://vincentarelbundock.github.io/Rdatasets/doc/datasets/USArrests.html), *ISLR* klasiği | Hiyerarşik kümeleme, dendrogram (50 ABD eyaleti) |
| quakes | [Rdatasets](https://vincentarelbundock.github.io/Rdatasets/doc/datasets/quakes.html) | DBSCAN, ε × minPts duyarlılığı (Fiji bölgesi, 1.000 deprem, enlem/boylam) |
| faithful | Azzalini & Bowman (1990), [Rdatasets](https://vincentarelbundock.github.io/Rdatasets/doc/datasets/faithful.html) | GMM, EM, BIC/AIC (Old Faithful, 272 patlama) |
| Sentetik şekil veri setleri | `sklearn.datasets` | 5 yöntem × 5 veri şekli karşılaştırması |

---

## 📓 18 · Boyut İndirgeme - İleri Yöntemler (`18_dimensionality_reduction_advanced.ipynb`)

PCA'nın üç sınırını — **doğrusal**, **denetimsiz**, bileşenleri **yorumlanamaz**
— her birini aşan yöntem ailesi: etiket kullanan **LDA**, doğrusal olmayan
**Kernel PCA**, "parçaların toplamı" olarak yazan **NMF**, karışık sinyalleri
ayıran **ICA**, ve 2B görselleştirme için komşuluk koruyan **t-SNE** / **UMAP**.
t-SNE'nin yanlış okunan sonuçları (küme boyutu/uzaklığı yanılgıları) ve
gömmelerin niceliksel doğrulaması (trustworthiness) ayrıca vurgulanıyor.

```
1. Neden "İleri" Boyut İndirgeme? Bir Harita
2. PCA Hatırlatma (ve Sınırları)
3. LDA: Etiketi Kullanan Boyut İndirgeme
4. Kernel PCA: Doğrusal Olmayan PCA
5. NMF: Eksi Olmayan "Parçaların Toplamı"
6. ICA: Bağımsız Kaynakları Ayırma (kokteyl partisi)
7. t-SNE: Komşulukları Koruyan Harita (+ yanlış okunan sonuçlar)
8. UMAP: Hızlı ve Genelleyebilen Manifold Öğrenme
9. Değerlendirme: Bir Boyut İndirgemenin İyi Olduğunu Nasıl Anlarız?
10. Özet: Hangi Yöntemi Ne Zaman Seçmeli?

🐍 Python: 11 uygulama adımı — Fashion-MNIST'te PCA'nın sıfırdan
   implementasyonu, LDA (elle örnek, sıfırdan Fisher LDA, PCA→LDA boru hattı),
   Kernel PCA (elle örnek, sıfırdan implementasyon, γ duyarlılığı), NMF
   (sıfırdan çarpımsal güncelleme, Olivetti yüzlerinde "parçalar", 20
   Newsgroups'ta konu modelleme ve LSA ile karşılaştırma), ICA (kurtosis,
   Merkezi Limit Teoremi, sıfırdan FastICA), yüzlerde PCA/NMF/ICA yan yana,
   t-SNE (sıfırdan kesin t-SNE + `sklearn.TSNE` + yorum tuzakları), UMAP
   (`n_neighbors` × `min_dist`, `transform`, denetimli UMAP), trustworthiness /
   continuity / kNN doğruluğu / süre ile niceliksel karşılaştırma, sekiz
   yöntemin yan yana sentezi
📊 R: `kernlab::kpca` + `fastICA` (Python'daki `make_circles` ve kokteyl
   partisi verisi), `Rtsne` + `uwot` (Fashion-MNIST alt kümesi)
🗄️ SQL (DuckDB): LDA izdüşümünün `JOIN` + `GROUP BY` + `SUM(x*w)` ile
   (boyut indirgemenin üretimde bir matris çarpımı olduğu) ve Kernel PCA
   `transform`'unun RBF kernel satırı (`CROSS JOIN` + `EXP`) ile yeniden üretimi
✏️ Egzersizler: E1–E6 (Python) — **çözülmeden** (TODO'lu) teslim edildi
```

**📦 Veri Setleri:**

| Dataset | Kaynak | Kullanım |
| --- | --- | --- |
| Fashion-MNIST (test kümesi) | [Zalando Research](https://github.com/zalandoresearch/fashion-mnist), Xiao ve ark. (2017) | PCA, LDA, t-SNE, UMAP, karşılaştırma (10.000 görüntü, 784 piksel, 10 sınıf) |
| Olivetti Faces | AT&T Laboratuvarları, [scikit-learn](https://scikit-learn.org/stable/datasets/real_world.html#olivetti-faces-dataset) | NMF "parçalar", PCA/NMF/ICA yan yana (400 yüz, 4.096 piksel) |
| 20 Newsgroups (6 kategori) | [scikit-learn](https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset) | NMF ile konu bulma vs LSA (5.604 yazı) |
| Sentetik (`make_circles`, karışık sinyaller) | `sklearn.datasets` | Kernel PCA, FastICA sezgisi |

---

## 📓 19 · Anomali Tespiti (`19_anomaly_detection.ipynb`)

Kümeleme "doğal grupları bul" diyordu; anomali tespiti tersini sorar:
**"hangi gözlem diğerlerine benzemiyor?"** Etiketli örnek neredeyse hiç
olmadığı için çoğunlukla denetimsiz veya yalnızca "normal" verinin bilindiği
yarı-denetimli çalışılıyor. Basit çizgiler (Z-skoru, IQR, Mahalanobis) ile
başlayıp dört yöntem ailesi işleniyor; hepsi aynı veri setlerinde
ROC-AUC / Average Precision / Precision@n ile kıyaslanıyor ve "etiket varsa
denetimli model ne kazandırır?" sorusu cevaplanıyor.

```
1. Anomali Nedir? Üç Tür (nokta / bağlamsal / kolektif), Üç Kurulum
2. Basit Çizgi: Z-skoru, IQR ve Mahalanobis
3. Isolation Forest: "Anormal Olan Çabuk İzole Edilir"
4. Local Outlier Factor (LOF): "Komşularına Göre Ne Kadar Seyrek?"
5. One-Class SVM: Normal Verinin Etrafına Kernel Sınırı (ν özelliği)
6. Elliptic Envelope ve Robust Kovaryans (MCD, maskeleme)
7. Değerlendirme: Etiket Yokken (veya Az İken) Nasıl Ölçeriz?
8. Zaman Serisinde Anomali: Bağlam ve Pencere
9. Özet: Hangi Yöntemi Ne Zaman Seçmeli?

🐍 Python: 9 uygulama adımı — Cardio'da Z-skoru/IQR/Mahalanobis, sıfırdan
   Isolation Forest (ağaç + orman, `sklearn` karşılaştırması), LOF (elle örnek,
   sıfırdan implementasyon, yerel–global anomali deneyi), One-Class SVM (ν
   özelliği, γ duyarlılığı, yarı-denetimli kullanım), Elliptic Envelope / MCD
   (maskeleme, robust kovaryans, yüksek boyut sınırı), 3 veri seti × 6 yöntem
   büyük karşılaştırma + rank-averaging topluluğu, yakalanan/kaçırılan
   anomalilerin görselleştirmesi, NYC Taksi talebinde zaman serisi anomalisi
   (Numenta Anomaly Benchmark), etiket varsa denetimli model + maliyet-duyarlı
   eşik
📊 R: `isotree` (Isolation Forest + Extended Isolation Forest) ve
   `dbscan::lof` (Cardio), `robustbase::covMcd` (klasik vs robust Mahalanobis)
🗄️ SQL (DuckDB): Mahalanobis skorunun uzun-biçim tablolarda iki self-join +
   `SUM` ile hesabı (`sklearn` ile karşılaştırma) ve zaman serisi anomali
   özniteliklerinin pencere fonksiyonlarıyla üretimi
✏️ Egzersizler: E1–E6 (Python) — **çözülmeden** (TODO'lu) teslim edildi
```

**📦 Veri Setleri:**

| Dataset | Kaynak | Kullanım |
| --- | --- | --- |
| Cardio (Cardiotocography) | [ODDS](http://odds.cs.stonybrook.edu/) / [UCI](https://archive.ics.uci.edu/dataset/193/cardiotocography) | Tüm yöntemler, Mahalanobis, R/SQL (1.831 kayıt, 20 öznitelik — tam doğrusal birleşim olan 1 sütun atıldı, %9.6 anomali) |
| Letter | ODDS / [UCI Letter Recognition](https://archive.ics.uci.edu/dataset/59/letter+recognition) | Yöntem karşılaştırması (1.600 örnek, 32 öznitelik, %6.25 anomali) |
| Arrhythmia | ODDS / [UCI](https://archive.ics.uci.edu/dataset/5/arrhythmia) | *p, n'e yakın* zor durum, MCD'nin yüksek boyut sınırı (452 hasta, 274 öznitelik) |
| NYC Taxi + Machine Temperature | [Numenta Anomaly Benchmark](https://github.com/numenta/NAB), Lavin & Ahmad (2015) | Zaman serisi anomalisi, etiket pencereleri (10.320 yarım-saatlik sayım, 5 gerçek olay) |

---

## 📓 20 · Birliktelik Kuralı Öğrenimi (`20_association_rule_learning.ipynb`)

Denetimsiz bölümün son klasik konusu: **"hangi şeyler birlikte görülür?"**
Elimizde yalnızca sepetler (market fişi, oturumdaki sayfalar, aynı anda tetiklenen
alarmlar) var; amaç `{A} → {B}` kurallarını bulmak ve hangisinin gerçekten işe
yaradığını ayırt etmek. Üç algoritma (**Apriori**, **Eclat**, **FP-Growth**)
sıfırdan yazılıp `mlxtend` ile birebir doğrulanıyor; en önemli ders, kuralın
"sık görünmesinin" gelecekte de geçerli olduğu anlamına gelmediği.

```
1. Problem Tanımı ve Ölçüler (support, confidence, lift, leverage, conviction)
2. Apriori — Anti-Monotonluk ile Seviye Seviye Arama
3. FP-Growth — Aday Üretmeden, Ağaçla
4. Eclat — Dikey Gösterim ve Kesişim
5. Kuralları Üretmek ve Değerlendirmek: Tuzaklar
6. Pratikte Nerede Kullanılır?

🐍 Python: 9 uygulama adımı — Groceries'ı tanıma, 5 fişlik elle örneğin kodla
   doğrulanması, Apriori / Eclat / FP-Growth ve kural üretiminin sıfırdan
   implementasyonu, üçünün `mlxtend` ile aynı sonucu verdiğinin ve hızlarının
   karşılaştırılması, kural ölçülerinin karşılaştırması, Online Retail (UK)'de
   ürün aileleri, **zamana göre ayırma** ile "kurallar gelecekte de geçerli mi?"
   (winner's curse: düşük destekli yüksek-lift kuralları), öneri (hit-rate)
   deneyi — kurallar popülerlik taban çizgisini geçiyor mu?, Mushroom'da yoğun
   veri / kapalı kümeler / kural patlaması ve minimal antecedent, null-
   değişmezlik: Lift mi, Kulczynski mi?
📊 R: `arules` (`apriori`, `eclat`, `interestMeasure`) — aynı eşiklerle
   Python referans sayılarıyla karşılaştırma
🗄️ SQL (DuckDB): sepet tablosunda self-join ile ikili kurallar (support /
   confidence / lift) ve üçlü self-join ile `{a1, a2} ⇒ c` kuralları
   (Apriori'nin 3'lü aşamasının SQL karşılığı) — Python ile birebir
✏️ Egzersizler: E1–E6 (Python) — **çözülmeden** (TODO'lu) teslim edildi
```

**📦 Veri Setleri:**

| Dataset | Kaynak | Kullanım |
| --- | --- | --- |
| Groceries | Hahsler ve ark. (R `arules`), [CSV kopyası](https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/groceries.csv) | Algoritma doğrulama, öneri deneyi (9.835 sepet, 171 ürün) |
| Online Retail (UK) | [UCI](https://archive.ics.uci.edu/dataset/352/online+retail), Chen, Sain & Guo (2012) | Zamana göre ayırma, mevsimsellik (UK faturaları, Aralık 2010–Aralık 2011; filtreden sonra 14.869 fatura, 404 ürün) |
| Mushroom | [UCI](https://archive.ics.uci.edu/dataset/73/mushroom) (Audubon Society Field Guide) | Özellik ⇒ sınıf kuralları, kapalı kümeler, kural patlaması (8.124 mantar, 22 kategorik özellik) |

---

## 📚 Kaynak Haritası

### Kitaplar

| # | Kitap | Yazar | Ücretsiz? | Bağlantı |
| --- | --- | --- | --- | --- |
| 🥇 | An Introduction to Statistical Learning (ISLR2, Bölüm 12) | James, Witten, Hastie, Tibshirani | ✅ | [statlearning.com](https://www.statlearning.com/) |
| 🥇 | The Elements of Statistical Learning (Bölüm 14) | Hastie, Tibshirani, Friedman | ✅ | [hastie.su.domains/ElemStatLearn](https://hastie.su.domains/ElemStatLearn/) |
| 🥇 | Pattern Recognition and Machine Learning (Bölüm 9: GMM + EM) | Bishop | ✅ | [Microsoft Research (PDF)](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) |
| 🥈 | Outlier Analysis (2. baskı) | Aggarwal | ❌ | [Springer](https://link.springer.com/book/10.1007/978-3-319-47578-3) |
| 🥈 | Introduction to Data Mining (Bölüm 5-6: Association Analysis) | Tan, Steinbach, Kumar | ❌ | Pearson |
| 🥉 | Data Mining: Concepts and Techniques (Bölüm 6-7) | Han, Kamber, Pei | ❌ | Morgan Kaufmann |
| 🥉 | Finding Groups in Data (PAM'ın orijinal kitabı) | Kaufman & Rousseeuw | ❌ | [Wiley](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316801) |

### Anahtar Makaleler

| Makale | Notebook | Konu |
| --- | --- | --- |
| Arthur & Vassilvitskii (2007), *k-means++: The Advantages of Careful Seeding* | 16 | k-means++ başlatma ve $O(\log k)$ yaklaşım garantisi |
| Rousseeuw (1987), *Silhouettes: A Graphical Aid…* | 16 | Silhouette katsayısının orijinal makalesi |
| Sculley (2010), *Web-Scale K-Means Clustering* | 16 | Mini-Batch K-Means |
| Ester, Kriegel, Sander, Xu (1996), *A Density-Based Algorithm for Discovering Clusters* | 17 | DBSCAN'in orijinal makalesi |
| Schubert ve ark. (2017), *DBSCAN Revisited, Revisited* | 17 | ε/minPts seçimi ve DBSCAN'in ne zaman kullanılmayacağı |
| Campello, Moulavi, Sander (2013), *Density-Based Clustering Based on Hierarchical Density Estimates* | 17 | HDBSCAN |
| Dempster, Laird, Rubin (1977), *Maximum Likelihood from Incomplete Data via the EM Algorithm* | 17 | EM algoritması |
| van der Maaten & Hinton (2008), *Visualizing Data using t-SNE* | 18 | t-SNE'nin orijinal makalesi |
| Wattenberg, Viégas, Johnson (2016), *How to Use t-SNE Effectively* | 18 | t-SNE yorum tuzakları (Distill) |
| McInnes, Healy, Melville (2018), *UMAP* | 18 | UMAP'in orijinal makalesi |
| Lee & Seung (1999), *Learning the parts of objects by non-negative matrix factorization* | 18 | NMF'in orijinal makalesi (Nature) |
| Hyvärinen & Oja (2000), *Independent component analysis: algorithms and applications* | 18 | ICA / FastICA |
| Liu, Ting, Zhou (2008), *Isolation Forest* | 19 | Isolation Forest'ın orijinal makalesi |
| Breunig, Kriegel, Ng, Sander (2000), *LOF: Identifying Density-Based Local Outliers* | 19 | LOF'un orijinal makalesi |
| Chandola, Banerjee, Kumar (2009), *Anomaly Detection: A Survey* | 19 | Nokta / bağlamsal / kolektif anomali ayrımı |
| Agrawal & Srikant (1994), *Fast Algorithms for Mining Association Rules* | 20 | Apriori'nin orijinal makalesi |
| Han, Pei, Yin (2000), *Mining Frequent Patterns without Candidate Generation* | 20 | FP-Growth |
| Zaki (2000), *Scalable Algorithms for Association Mining* | 20 | Eclat |

### Resmi Dokümantasyon

| Kaynak | Kullanım |
| --- | --- |
| [scikit-learn — Clustering](https://scikit-learn.org/stable/modules/clustering.html) & [Gaussian Mixture Models](https://scikit-learn.org/stable/modules/mixture.html) | Notebook 16-17 |
| [SciPy — `scipy.cluster.hierarchy`](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html) | Notebook 17 |
| [scikit-learn — Decomposition](https://scikit-learn.org/stable/modules/decomposition.html) & [Manifold learning](https://scikit-learn.org/stable/modules/manifold.html) | Notebook 18 |
| [UMAP — resmi dokümantasyon](https://umap-learn.readthedocs.io/) | Notebook 18 |
| [scikit-learn — Novelty and Outlier Detection](https://scikit-learn.org/stable/modules/outlier_detection.html) | Notebook 19 |
| [PyOD](https://github.com/yzhao062/pyod) & [ODDS veri deposu](http://odds.cs.stonybrook.edu/) | Notebook 19 |
| [MLxtend — Frequent Patterns / Association Rules](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/) | Notebook 20 |
| [CRAN — `cluster`](https://cran.r-project.org/package=cluster) / [`mclust`](https://cran.r-project.org/package=mclust) | Notebook 16-17 R uygulaması |
| [CRAN — `kernlab`](https://cran.r-project.org/package=kernlab) / [`fastICA`](https://cran.r-project.org/package=fastICA) / [`Rtsne`](https://cran.r-project.org/package=Rtsne) / [`uwot`](https://cran.r-project.org/package=uwot) | Notebook 18 R uygulaması |
| [CRAN — `isotree`](https://cran.r-project.org/package=isotree) / [`dbscan`](https://cran.r-project.org/package=dbscan) / [`robustbase`](https://cran.r-project.org/package=robustbase) | Notebook 19 R uygulaması |
| [CRAN — `arules`](https://cran.r-project.org/package=arules) | Notebook 20 R uygulaması |
| [DuckDB — SQL Fonksiyonları](https://duckdb.org/docs/sql/functions/numeric) & [Window Functions](https://duckdb.org/docs/sql/window_functions) | Tüm notebook'ların SQL bölümlerinin temeli |

Notebook içi `Kaynaklar` bölümlerinde her dosya için tam kaynak listesi
mevcut (veri seti sayfaları, R paket dokümantasyonları, projedeki önceki
notebook'lara çapraz referanslar dahil).

---
