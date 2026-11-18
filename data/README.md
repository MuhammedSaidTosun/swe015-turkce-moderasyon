# OffensEval-TR 2020 verisi

Veri kaynağı Çağrı Çöltekin'in *A Corpus of Turkish Offensive Language on Social Media* çalışmasıdır (LREC 2020, s. 6174–6184). Üretici, etiketleri CC BY ile yayımlar. Projeyi çoğaltmak için aşağıdaki betiği kullanın. [Proje ve indirme sayfası](https://coltekin.github.io/offensive-turkish/) · [Makale](https://aclanthology.org/2020.lrec-1.758/)

```bash
python data_prep.py
```

Alternatif olarak proje kökünde `python data_prep.py --archive /dosya/yolu/offenseval2020-turkish.zip` komutunu çalıştırabilirsiniz. Açılmış sürüm dosyaları için `--source-dir KLASÖR` kullanılır. Betik eğitim metinlerini ve etiketlerini yayımlanmış TSV'den okur. Test etiketleri ayrı dosyadadır ve test mesajlarıyla kimlik üzerinden birleştirilir.

Bu başlangıç çalışmasında yayımlanmış üç TSV'ye GitHub'daki kopyadan erişildi. Sürüm `e94b0bc16cce5b91a4bd0a2823c818d4ed58e606`. [Kullanılan dosyalar ve sürüm README'si](https://github.com/vidhur2k/Multilngual-Hate-Speech/tree/e94b0bc16cce5b91a4bd0a2823c818d4ed58e606/data/raw/turkish)

Yayımlanmış 31.756 eğitim satırından 2'si testte de yer alıyordu. Eğitim içindeki 9 tekrar da çıkarıldı. Bu karşılaştırma Unicode NFC ve boşluk eşitlemesiyle yapılır. CSV'ye özgün mesaj yazılır. Kalan eğitim örneklerinin yaklaşık %10'u her sınıftan ayrı ayrı doğrulamaya ayrılır. Aynı dosyalarda aynı ayrımı üretmek için örnekler sabit tohum (42) ve kayıt kimliğinin SHA-256 özetiyle sıralanır.

| Küme | Normal (0) | Saldırgan (1) | Toplam |
| --- | ---: | ---: | ---: |
| Eğitim | 23.053 | 5.518 | 28.571 |
| Doğrulama | 2.561 | 613 | 3.174 |
| Resmî test | 2.812 | 716 | 3.528 |

Çıktılar `data/processed/train.csv`, `validation.csv` ve `test.csv` dosyalarında UTF-8 olarak yazılır. Her biri `text,label` sütunlarını içerir. `split_summary.json` kaynak ve çıktı dosyalarının SHA-256 özetlerini, sınıf sayılarını ve çıkarılan tekrarları kaydeder.
