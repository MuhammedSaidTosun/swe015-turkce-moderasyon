# Türkçe sohbet moderasyonu · SWE015

Türkçe sohbet mesajlarını beş yöntemle karşılaştıran SWE015 uygulaması. Tam kurulumda varsayılan model, saldırgan dil tespiti için ince ayar yapılmış BERTurk’ün INT8 sürümüdür. Kullanıcı modeli değiştirebilir. Saldırgan olarak işaretlenen mesaj maskelenir. Yan panelde mesaj başına CPU tahmin süresi ve tarayıcının yanıtı beklediği toplam süre gösterilir.

Arayüz ve 10 slaytlık sunum İngilizcedir. Model Türkçe mesajları sınıflandırır. İngilizce PowerPoint’in her slaydında konuşmacı notları vardır. Prova kılavuzunun ilk sayfasında iki dakikalık video için ekranda yapılacaklar ve okunacak İngilizce cümleler, sonraki sayfalarda her slaydın tam konuşma metni, Türkçe açıklamalar ve telaffuz yardımı bulunur. Kapağa teslimden önce öğrenci numarası ve video bağlantısı eklenir.

[İngilizce sunum PDF’si](docs/SWE015_Turkce_Moderasyon_Sunum.pdf) · [İngilizce PowerPoint ve konuşmacı notları](docs/SWE015_Turkce_Moderasyon_Sunum.pptx) · [Türkçe yönlendirmeli İngilizce konuşma kılavuzu](docs/SWE015_Speaking_Guide.pdf)

## Mac’te demoyu açma

Python 3.12 ve özel depoya erişimi olan bir GitHub hesabı gerekir. Ağırlıkları indirmek için GitHub CLI’ı (`gh`) aynı hesapla kullanın. Depoyu daha önce klonladıysanız proje klasöründe `git pull --ff-only` çalıştırarak ilk iki komutu atlayabilirsiniz.

```bash
git clone https://github.com/MuhammedSaidTosun/swe015-turkce-moderasyon.git
cd swe015-turkce-moderasyon
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-bert.txt
gh release download berturk-moderasyon-v1 --pattern berturk-models.zip --dir artifacts --skip-existing
python -m zipfile -e artifacts/berturk-models.zip artifacts
python -m uvicorn app:app
```

Tarayıcıda [yerel uygulamayı açın](http://127.0.0.1:8000). Açılır menüde **BERTurk · INT8** seçili olarak gelir. Eski sunucu aynı portu kullanıyorsa o Terminal penceresinde Ctrl+C ile durdurun veya yeni sunucuyu `python -m uvicorn app:app --port 8001` ile başlatıp `http://127.0.0.1:8001` adresini açın. Üç küçük modelin dosyaları klonla birlikte gelir. İki BERTurk modelini içeren sürüm paketi ayrıca indirilir. [Model paketinin sürüm sayfası](https://github.com/MuhammedSaidTosun/swe015-turkce-moderasyon/releases/tag/berturk-moderasyon-v1)

## Beş modelin aynı test kümesindeki sonucu

OffensEval-TR resmî test kümesindeki 3.528 mesajın 716’sı saldırgan, 2.812’si normal olarak etiketlendi. Makro F1, her iki sınıfın F1 puanına eşit ağırlık verir. Gecikme için önceden belirlenen hedef, CPU’da tek bir mesajı ortalama 50 ms içinde sınıflandırmaktı.

| Model | Test makro F1 | Model boyutu (MB) | Ortalama CPU (ms) | Yakalanan saldırgan mesaj (/716) | Yanlış alarm (/2.812) |
|---|---:|---:|---:|---:|---:|
| Sözlük + düzenleme mesafesi | 0,5340 | 0,000588 | 0,052 | 70 | 18 |
| TF-IDF + lojistik regresyon | 0,7214 | 1,028 | 0,189 | 422 | 365 |
| Word2Vec + sinir ağı | 0,7644 | 8,311 | 0,083 | 409 | 200 |
| BERTurk FP32 · PyTorch | 0,7817 | 443,31 | 13,876 | 421 | 170 |
| **BERTurk INT8 · ONNX Runtime** | **0,7838** | **112,55** | **9,009** | **420** | **162** |

Ölçümler Apple M4’te tek CPU iş parçacığıyla yapıldı. On ısınma çağrısından sonra her modele aynı sırada 200 mesaj, tek tek verildi. Süre metin ön işlemeyi ve model tahminini kapsar. Tarayıcı ve ağ süreleri ayrıca uygulamada görüntülenir. BERTurk boyutları ağırlıkları, tokenizer’ı ve yapılandırmayı kapsar. [Tam CSV](results/five_model_test.csv) · [Ortam, ölçüm protokolü ve dosya özetleri](results/five_model_test.json)

INT8 paketi FP32’ye göre %74,6 daha küçük. INT8 tahminlerinin 95. yüzdeliği 15,33 ms, en yavaş tahmin 16,51 ms idi. Bu 200 ölçümde 50 ms’yi aşan tahmin olmadı. İlk PyTorch INT8 denemesi ise aynı cihazda ortalama 55,83 ms sürdü. Son sürümde çıkarım motoru da değişti. Etkiyi aynı motorla kontrol ettiğimizde ONNX Runtime, doğrulamadan seçilen aynı 200 mesajda FP32 için 12,09 ms, INT8 için 9,25 ms verdi. [İlk INT8 ölçümü](results/five_model_test_1thread.csv) · [Aynı çıkarım motoruyla karşılaştırma](results/ort_paired_validation.json)

## Canlı demoda aynı mesajla beş model

BERTurk INT8, `Sen bir aptalsın` mesajına %91,8 saldırganlık olasılığı verir ve mesajı gizler. Karşılaştırma için `Ona aptal demen hiç hoş değil` cümlesini **sonuna nokta eklemeden** gönderin. Her model değişikliğinden sonra aynı metni tekrar gönderin.

| Seçilen model | Noktasız cümlede uygulamanın kararı |
|---|---|
| BERTurk · INT8 | Clean · %39,7 saldırganlık olasılığı (%62 eşik) |
| Word list + edit distance | Offensive · Message hidden |
| TF-IDF + logistic regression | Clean |
| Word2Vec + neural network | Clean · %3,2 saldırganlık skoru (%53 eşik) |
| BERTurk · FP32 | Clean · %56,6 saldırganlık olasılığı (%62 eşik) |

Cümlede konuşan kişi hakarete karşı çıkıyor. Bu örnekte sözlük yanlış alarm verdi. Yeniden eğitilen Word2Vec modeli cümleyi görünür bıraktı. Cümlenin sonuna nokta eklenirse BERTurk INT8 skoru %65,0’a çıkar ve mesaj bu kez gizlenir. Yukarıdaki kararlar uygulamanın yerel API’sinde ölçüldü. 3.528 mesaj üzerindeki test sonuçları ayrı bir değerlendirmedir.

## Eğitimi ve ölçümü tekrarlama

Aynı Python ortamında, proje klasöründen çalıştırın. Veri hazırlama adımı ilk çalıştırmada internet erişimi gerektirir.

```bash
python data_prep.py
python train_classical.py
python train_neural.py
python train_bert.py
python evaluate_all.py
```

Hazır modellerle yalnızca sonuçları tekrarlamak için veri hazırlama adımından sonra `python evaluate_all.py` yeterlidir. FP32 ve INT8’i aynı ONNX motorunda kıyaslamak için `python compare_ort_validation.py` çalıştırılır. `train_bert.py`, seçilen FP32 modelinin ONNX INT8 sürümünü de üretir. Elinizde hazır FP32 ağırlıkları varken yalnız dönüşümü tekrarlamak için `python export_bert_onnx.py artifacts/bert_fp32 /tmp/berturk-int8-yeni` kullanılabilir. Hedef dizin boşta olmalıdır.

Eğitim, TF-IDF sözlüğünü eğitim mesajlarından çıkarır. Lojistik regresyonun `C` değerini doğrulama kümesindeki macro-F1 ile seçer. `data/processed/test.csv` tüm modeller için ortak test dosyasıdır. Model veya eşik geliştirirken `python train_classical.py --eval-split validation` kullanın.

Word2Vec, eğitim mesajlarında bir kelimenin sağındaki ve solundaki en fazla üçer kelimeden ortadaki kelimeyi tahmin etmeyi öğrenir (CBOW, beş negatif örnek). En az iki kez görülen kelimeler 48 boyutlu vektörlerle temsil edilir. Sınıflandırıcı, standartlaştırılmış ortalama kelime vektörlerine kelimeleri, sıralı kelime çiftlerini ve 3–4 karakterlik parçaları ekler. 12.000 gösterge eğitim etiketleri ile ki-kare puanına göre seçilir. Özellikler birlikte 64 ReLU nöronuna verilir. Word2Vec, OffensEval eğitim metinlerinden öğrenir. Sınıflandırıcı için bunlara 744 dengeli, proje için yazılmış sentetik sohbet mesajı eklenir. Sinir ağı mini-batch SGD ve geri yayılımla eğitilir. Eğitim turu ve karar eşiği doğrulama macro-F1 değerine göre seçilir. `results/neural_training.json` eğitim kaydını içerir.

BERTurk için başlangıç noktası, `dbmdz/bert-base-turkish-cased` modelinin `b6e1de16c983e0f2c70664591ea3f22810072608` sürümüdür. Büyük ve küçük harf korunur. Eğitim maliyetini Mac CPU’suna uygun tutmak için sabit tohumla sınıf oranlarını koruyan 12.288 eğitim örneği kullanılır. On iki encoder bloğunun tamamı çıkarımda çalışır. Eğitim sırasında kelime gömmeleri ve ilk on blok sabittir. Son iki blok, pooler ve iki sınıflı çıkış katmanı güncellenir. İki tur boyunca en fazla 48 token, 24 mesajlık batch ve AdamW ile 0,00004 öğrenme hızı kullanılır. Tur ve sınıflandırma eşiği, 3.174 doğrulama mesajında macro-F1 ile seçilir. İkinci tur doğrulamada 0,79195 macro-F1 ile seçildi. FP32 ve INT8 aynı 0,62 karar eşiğini kullanır. Tam eğitim, doğrulama ve ilk dışa aktarma Apple M4 CPU üzerinde 350 saniye sürdü. Dağıtılan INT8 sürümünde ONNX Runtime kullanılır. Doğrusal katmanların ağırlıkları kanal başına işaretli INT8 olarak, üç gömme tablosu ise tablo başına işaretsiz 8 bit olarak saklanır. Dinamik giriş kuantizasyonu çalışma sırasında yapılır. [BERTurk model kartı](https://huggingface.co/dbmdz/bert-base-turkish-cased)

## Uygulamada veri akışı

```mermaid
flowchart TD
    UI["Tarayıcı · mesaj ve model seçimi"] --> API["FastAPI · CPU"]
    API --> TR["Türkçe metin normalizasyonu"]
    API --> TOK["BERTurk tokenizer · en fazla 48 token"]
    TR --> RULE["Sözlük ve düzenleme mesafesi"]
    TR --> TF["TF-IDF ve lojistik regresyon"]
    TR --> NN["Word2Vec ve sinir ağı"]
    TOK --> FP["BERTurk FP32"]
    TOK --> INT["BERTurk INT8"]
    RULE --> OUT["Etiket, skor ve tahmin süresi"]
    TF --> OUT
    NN --> OUT
    FP --> OUT
    INT --> OUT
    OUT --> VIEW["Sohbet · saldırgan mesajı maskele"]
```

| Ödevdeki teknik alan | Uygulamadaki karşılığı |
|---|---|
| A · Sinir ağları | Eğitim mesajlarından öğrenilen Word2Vec vektörleri ve 64 gizli nöronlu sınıflandırıcı. |
| B · Klasik NLP | Türkçe metin normalizasyonu, düzenleme mesafeli sözlük ve TF-IDF + lojistik regresyon. |
| C · Transformer | Türkçe WordPiece tokenizer ve iki sınıflı görev için ince ayarlı, 12 katmanlı BERTurk. |
| D · Verimli modeller | Seçilen BERTurk modelinin INT8’e dönüştürülmesi, CPU ölçümü ve sohbet ekranında çalıştırılması. |

## Veri bölümü ve sınırlamalar

Kaynak eğitim bölümünde 31.756 mesaj vardı. Boşluk ve Unicode normalizasyonuyla saptanan dokuz tekrar ile resmî testte de bulunan iki eğitim kaydı çıkarıldı. Sınıf oranları korunarak 28.571 eğitim ve 3.174 doğrulama mesajı ayrıldı. Resmî test kümesindeki 3.528 mesajın tamamı korundu. İlk çalıştırmada kaynaklar OffensEval dağıtımının bir aynasından alındı. `data_prep.py` varsayılan olarak yazarın sunduğu ZIP’i indirir. Yerel arşivle çalışmak için `python data_prep.py --archive offenseval2020-turkish.zip` kullanılabilir. [Kullanılan ayna](https://github.com/vidhur2k/Multilngual-Hate-Speech)

Word2Vec 29.211 kelime için vektör öğrendi. Güncellenen sinir ağında 14. eğitim turu ve 0,53 eşiği OffensEval doğrulama kümesinde seçildi. Aynı resmî testte eski model 411 hakareti kaçırıp 439 yanlış alarm verdi; yenisinde bu sayılar 307 ve 200 oldu. Geliştirmeden önce ayrılan 40 sentetik oyun sohbeti kontrolünde eski model 24, yenisi 39 doğru karar verdi. Bu küçük kontrol gerçek oyuncu trafiği performansını ölçmez. [Karşılaştırma ve üç demo çıktısı](results/neural_retraining.json). BERTurk bağlamı kullanabiliyor, ancak canlı örnekte görüldüğü gibi eleştiri ve hakareti karıştırabiliyor. Test kümesi Twitter mesajlarından oluşuyor. Oyun sohbetlerinde başarımı ölçmek için o alandan etiketli bir test kümesi gerekir.

## İki dakikalık demo akışı

| Zaman | Ekranda gösterilecek işlem |
|---|---|
| 0:00–0:13 | Kapakta adını, seçtiğin projeyi ve amacını söyle. |
| 0:13–0:49 | Uygulamada model menüsünü aç. Sözlük, TF-IDF, Word2Vec, BERTurk FP32 ve BERTurk INT8 için birer cümleyle seçim nedenini anlat. |
| 0:49–1:08 | BERTurk INT8’i seç. Examples altındaki selamlaşma, doğrudan hakaret ve hakarete itiraz örneklerine soldan sağa tıkla. |
| 1:08–1:22 | Word2Vec + neural network modelini seç. Aynı üç örneği aynı sırayla gönder ve iki modelin kararlarını karşılaştır. |
| 1:22–1:38 | 4. slaytta tarayıcı, yerel sunucu, CPU üzerinde çalışan model ve sohbete dönen sonucu göster. |
| 1:38–1:58 | 7. slaytta Word2Vec’in makro-F1 artışını (0,6330 → 0,7644), INT8’in puanını (0,7838) ve ortalama CPU süresini (9,009 ms) göster. |

Sesli kaydı en fazla iki dakika olacak şekilde tamamlayın. Video bağlantısını ve öğrenci numarasını düzenlenebilir sunumun ilk slaydına ekleyip PDF’yi yeniden dışa aktarın. Teslimden önce değerlendirecek kişinin özel depoya ve videoya erişebildiğini doğrulayın.

## Veri ve atıf

Çağrı Çöltekin (2020), *A Corpus of Turkish Offensive Language on Social Media*. Bu projede OffensEval 2020 Türkçe alt görev A kullanılır. Etiketler `0 = NOT` ve `1 = OFF` olarak kodlanır. [Yazarın veri sayfası](https://coltekin.github.io/offensive-turkish/) · [Makale](https://aclanthology.org/2020.lrec-1.758/)
