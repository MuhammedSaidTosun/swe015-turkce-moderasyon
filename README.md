# Türkçe sohbet moderasyonu · SWE015

Kullanıcının yazdığı Türkçe mesaj sözlük, TF-IDF veya Word2Vec + sinir ağı ile sınıflandırılır. Saldırgan olarak işaretlenen mesaj sohbet geçmişinde maskelenir. Yan panel her mesaj için kullanılan modeli ve CPU tahmin süresini gösterir.

## Mac’te demoyu açma

GitHub hesabınız bu özel depoya erişebiliyorsa Terminal’de aşağıdaki komutlarla depoyu klonlayıp Python 3.12 ortamını kurabilirsiniz.

```bash
git clone https://github.com/MuhammedSaidTosun/swe015-turkce-moderasyon.git
cd swe015-turkce-moderasyon
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app:app
```

Tarayıcıda [yerel uygulamayı açın](http://127.0.0.1:8000). Üç modelin dosyaları `artifacts/` klasöründe hazırdır. Önceki sürüm 8000 numaralı portu kullanıyorsa o Terminal penceresinde Ctrl+C ile durdurun. Yeni sunucuyu başlattıktan sonra Helium’daki sayfayı ⌘R ile yenileyin.

## İlk deneme

1. Model menüsünden **Sözlük + düzenleme mesafesi** seçin. “Bu oyunun son güncellemesi çok kötü.” mesajını gönderin. Karar, kullanılan skor ve mesajın işlem süresi görünür.
2. “Sen bir aptalsın.” mesajını gönderin. Sözlük eşleştiğinde metnin yerinde “Mesaj maskelendi” görülür. Sözlükte gösterilen skor elle belirlenmiş bir kural skorudur.
3. **TF-IDF + lojistik regresyon** seçip aynı iki mesajı yeniden deneyin. Gösterilen güven, sınıflandırıcının seçtiği sınıfa verdiği olasılıktır. İki modelin ayrıştığı bir mesaj bulup nedenini tartışın.
4. **Word2Vec + sinir ağı** ile aynı mesajları karşılaştırın. Ekran bu modelin saldırganlık olasılığını ve karar eşiğini gösterir. Olasılık eşiğe ulaştığında mesaj maskelenir.

**Bağlam deneyi.** “Ona aptal demen hiç hoş değil.” cümlesi, bir hakareti eleştiriyor. Sözlük “aptal” sözcüğünü yakalayıp cümleyi maskeledi. TF-IDF aynı cümleyi normal olarak gösterdi. Sinir ağı da cümleyi maskeledi. Saldırganlık olasılığı %46,4 ile doğrulamada seçilen %46 eşiğinin hemen üstündeydi. Bu, demoyu anlamak için yazılmış bir örnektir.

## Eğitimi ve ölçümü tekrarlama

Aynı Python ortamında, proje klasöründen çalıştırın. Veri hazırlama adımı ilk çalıştırmada internet erişimi gerektirir.

```bash
python data_prep.py
python train_classical.py
python train_neural.py
python evaluate_neural.py
```

Eğitim, TF-IDF sözlüğünü eğitim mesajlarından çıkarır. Lojistik regresyonun `C` değerini doğrulama kümesindeki macro-F1 ile seçer. `data/processed/test.csv` tüm modeller için ortak test dosyasıdır. Model veya eşik geliştirirken `python train_classical.py --eval-split validation` kullanın.

Word2Vec, eğitim mesajlarında bir kelimenin sağındaki ve solundaki en fazla üçer kelimeden ortadaki kelimeyi tahmin etmeyi öğrenir (CBOW, beş negatif örnek). En az iki kez görülen kelimeler 48 boyutlu vektörlerle temsil edilir. Sınıflandırıcı bir mesajdaki bilinen kelimelerin vektörlerini ortalar, eğitim kümesinin istatistikleriyle standartlaştırır ve 64 ReLU nöronundan geçirir. Sinir ağı mini-batch SGD ve geri yayılımla eğitilir. Eğitim turu ve karar eşiği doğrulama macro-F1 değerine göre seçilir. `results/neural_training.json` eğitim kaydını içerir.

`results/classical_test.csv` her modelin macro-F1, diskteki model boyutu (ondalık MB), ortalama CPU tahmin süresi (ms) ve donanım bilgisini içerir. JSON dosyasında test dosyasının SHA-256 özeti, seçilen hiperparametre ve ölçüm protokolü de yer alır. Gecikme 10 ısınma çağrısından sonra, aynı sıradaki 200 mesajda, her seferinde tek mesaj işlenerek ölçülür. Sonuçlara ağ ve tarayıcı süresi dahil edilmez.

### Veri bölümü ve ilk Mac ölçümü

Kaynak eğitim bölümünde 31.756 mesaj vardı. Boşlukları ve Unicode biçimini eşitleyerek saptanan 9 tekrar ile resmî testte de bulunan 2 eğitim kaydı çıkarıldı. Kalan örneklerden sınıf oranları korunarak 28.571 eğitim ve 3.174 doğrulama mesajı ayrıldı. Resmî test kümesindeki 3.528 mesajın tamamı korundu. İlk çalıştırmada kaynak dosyalar [OffensEval dağıtımının bir aynasından](https://github.com/vidhur2k/Multilngual-Hate-Speech) alındı. `data_prep.py` varsayılan olarak yazarın sunduğu ZIP’i indirir. Arşivi kendiniz indirirseniz `python data_prep.py --archive offenseval2020-turkish.zip` komutunu kullanabilirsiniz.

| Yöntem | Test macro-F1 | Diskte boyut (MB) | Ortalama CPU gecikmesi (ms) |
|---|---:|---:|---:|
| Sözlük + düzenleme mesafesi | 0,5340 | 0,000588 | 0,042 |
| TF-IDF + lojistik regresyon | 0,7214 | 1,028072 | 0,184 |

Apple M4 üzerinde iki yöntem için aynı 200 tek-mesaj çağrısı ölçüldü. Testteki 716 saldırgan mesajın 70’ini sözlük, 422’sini TF-IDF yakaladı. Normal mesajlarda yanlış alarm sayısı sırasıyla 18 ve 365 oldu.

### Üç modelin aynı CPU'da karşılaştırması

`evaluate_neural.py` üç modeli de yeniden yükleyip aynı 3.528 test mesajında ölçer. Aşağıdaki 200 tek-mesajlık süre ölçümü Linux üzerinde AMD EPYC 9V74 işlemcide yapıldı. Paket sürümleri ve model dosyalarının SHA-256 değerleri `results/neural_test.json`, tam tablo `results/comparison_test.csv` içindedir. Sen kendi Mac'inde bu komutu çalıştırdığında yeni süre ve donanım bilgileri yazılır.

| Model | Test macro-F1 | Boyut (MB) | Ortalama süre (ms) | Yakalanan saldırgan mesaj (/716) | Yanlış alarm (/2812) |
|---|---:|---:|---:|---:|---:|
| Sözlük | 0,5340 | 0,000588 | 0,115 | 70 | 18 |
| TF-IDF + lojistik regresyon | 0,7214 | 1,028072 | 0,415 | 422 | 365 |
| Word2Vec + sinir ağı | 0,6330 | 5,412562 | 0,095 | 305 | 439 |

Word2Vec 29.211 kelime için vektör öğrendi. Sinir ağının 19. eğitim turu ve 0,46 eşiği doğrulamada seçildi (macro-F1 0,6473). Testte TF-IDF hem daha çok saldırgan mesajı yakaladı hem de daha az yanlış alarm verdi. Kelime vektörlerinin ortalaması sözcük sırasını korumaz. Bu, özellikle alıntı ve eleştiri içeren cümleleri incelerken tartışılabilecek bir sınırlamadır.

## Sonraki aşamalar

| Aşama | Yapılacak iş | Tamamlandığını nasıl kontrol ederim? |
|---|---|---|
| Transformer | BERTurk’ü ikili sınıflandırma için eğitim kümesinde ince ayardan geçir. Model seçimini doğrulamada yap. | Dördüncü satırı ve açıklayabileceğin bir hata örneğini elde et. |
| Kuantizasyon | İnce ayarlı modeli CPU için INT8’e dönüştür. Hedef gecikmeyi kıyaslamadan önce belirle. | Beşinci satırı ölç. INT8 modelini uygulama menüsüne ekle ve gerçek mesajlarla hedefi kontrol et. |
| Teslim | 8–12 slaytlık PDF, sesli ve en fazla iki dakikalık demo, kurulum adımları olan GitHub/GitLab deposu. | İlk slaytta iki bağlantıyı, teknik haritayı, gerçek mimariyi, AI kullanım slaydını ve erişim izinlerini kontrol et. |

Sunumda kendi çalıştırmandan aldığın ölçümleri kullan. Gerçek hata örneğini model çıktısıyla birlikte sakla. Etiketli kaynak Twitter mesajlarından oluştuğu için oyun sohbetlerine ne kadar genellenebildiğini sonuçları yorumlarken dikkate al.

## Veri ve atıf

Çağrı Çöltekin (2020), *A Corpus of Turkish Offensive Language on Social Media*. Bu projede OffensEval 2020 Türkçe alt görev A kullanılır. Etiketler `0 = NOT` ve `1 = OFF` olarak kodlanır. [Yazarın veri sayfası](https://coltekin.github.io/offensive-turkish/) · [Makale](https://aclanthology.org/2020.lrec-1.758/)
