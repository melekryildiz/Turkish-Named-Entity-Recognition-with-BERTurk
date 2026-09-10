# Turkish Named Entity Recognition (NER) with BERTurk

This project was developed by fine-tuning the `dbmdz/bert-base-turkish-cased` (BERTurk) model using the Hugging Face Transformers library to detect person (PERSON), location (LOCATION), and organization (ORGANIZATION) entities within Turkish texts.

## Technologies
*   Python, PyTorch
*   Hugging Face Transformers & Datasets
*   BERTurk (`dbmdz/bert-base-turkish-cased`)
*   WikiANN Turkish Dataset
*   Seqeval (Evaluation Metrics)

## Model Performance (Test Set Results)

The model was evaluated on the unseen **Test Dataset**, achieving the following final metrics:

*   **Precision:** 91.44% (0.914363)
*   **Recall:** 92.26% (0.922573)
*   **F1 Score:** 91.85% (0.918450)
*   **Accuracy:** 97.11% (0.971069)
*   **Test Loss:** 0.123274

### Training Progress
The model's progression on the validation set during the 3-epoch fine-tuning process is as follows:

| Epoch | Training Loss | Validation Loss | Precision | Recall | F1 Score | Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.1569 | 0.1200 | 0.8936 | 0.9098 | 0.9016 | 0.9660 |
| **2** | 0.0938 | 0.1149 | 0.9106 | 0.9225 | 0.9165 | 0.9703 |
| **3** | 0.0608 | 0.1223 | 0.9145 | 0.9278 | 0.9211 | 0.9713 |

---

## Manual Test Sentences (20 Custom Examples)

The trained model was tested via the inference `pipeline` on 20 different sentences designed to challenge Turkish morphology and entity ambiguity:

1. Mustafa Kemal Atatürk Ankara'da yaşamıştır.
2. Çankaya Üniversitesi, Ankara Büyükşehir Belediyesi ile ortak bir proje yürütüyor.
3. RKSOFT şirketinde derin öğrenme modelleri ve bilgisayarlı görü algoritmaları geliştiriliyor.
4. TEKNOFEST yarışması bu yıl Adana'da büyük bir coşkuyla düzenlenecek.
5. IEEE öğrenci kolu, mühendislik fakültesinde yeni bir toplantı organize etti.
6. Hasan Kalyoncu Üniversitesi'ne kayıt yaptırmak için Gaziantep'e gittim.
7. Elon Musk, SpaceX ve Tesla'nın operasyon merkezini Teksas'a taşıma kararı aldı.
8. Prof. Dr. İlber Ortaylı, Topkapı Sarayı'nda Osmanlı tarihi üzerine bir konferans verecek.
9. Mark Zuckerberg, Meta'nın gelecekteki sanal gerçeklik yatırımları hakkında konuştu.
10. Sağlık Bakanlığı ile Milli Eğitim Bakanlığı pandemiden sonra yeni bir genelge yayınladı.
11. Birleşmiş Milletler, New York'taki genel merkezinde acil bir güvenlik zirvesi topladı.
12. Anadolu Ajansı'nın son dakika haberine göre, Japonya'nın başkenti Tokyo'da deprem oldu.
13. Ali, Ayşe ile birlikte akşam Kızılay Meydanı'nda buluşup kahve içecek.
14. Fatih Sultan Mehmet, 1453 yılında İstanbul'u fethederek bir çağı kapattı.
15. Galatasaray, UEFA Şampiyonlar Ligi grup maçında Bayern Münih ile karşılaşacak.
16. Tarkan'ın yeni albümü sadece Türkiye'de değil, Avrupa'da da büyük ilgi gördü.
17. Karadeniz Bölgesi'nde yaz aylarında başlayan çay hasadı sonbahara kadar sürer.
18. Ufuk, yarın sabah erkenden Türk Hava Yolları uçağıyla İzmir'e uçacak.
19. Boğaziçi Üniversitesi'ndeki araştırmacılar, TÜBİTAK destekli projelerini tamamladı.
20. Hafta sonu Kadıköy'den vapura binip Beşiktaş'a geçtik ve Dolmabahçe'yi gezdik.

---

## Error Analysis (5 Examples)

5 specific errors made by the model on challenging Turkish structures and their technical reasons are analyzed below:

**1. Subword Tokenization and Label Conflict**
*   **Input:** "Hafta sonu Kadıköy'den vapura binip Beşiktaş'a geçtik ve Dolmabahçe'yi gezdik."
*   **Expected:** Dolmabahçe `(LOC)`
*   **Model's Prediction:** Dolma `(LOC)`, ##bahçe `(ORG)`
*   **Reason:** The BERTurk tokenizer split the word "Dolmabahçe" into `["Dolma", "##bahçe"]`. Because the model classified the first piece as a location and the second as an organization, the aggregation strategy failed to merge them, splitting the word into two separate entities.

**2. Missing B-ORG (Beginning) Tag**
*   **Input:** "TEKNOFEST yarışması bu yıl Adana'da büyük bir coşkuyla düzenlenecek."
*   **Expected:** TEKNOFEST `(ORG)`
*   **Model's Prediction:** ##OFE `(ORG)`
*   **Reason:** The model failed to recognize the beginning of the subword-tokenized word "TEKNOFEST" (`TEKN`) as an entity. It only assigned a confidence score to the middle subword `##OFE`. Since the beginning tag (`B-ORG`) was missed, the entity's overall integrity was broken.

**3. OOV (Out of Vocabulary) and Missed Organization Name**
*   **Input:** "RKSOFT şirketinde derin öğrenme modelleri..."
*   **Expected:** RKSOFT `(ORG)`
*   **Model's Prediction:** None (Not found)
*   **Reason:** Although "RKSOFT" is written entirely in uppercase, it is a highly specific company name likely absent from the WikiANN dataset. The model failed to deduce its ORG status purely from the context.

**4. Incorrect Context Association (False Positive)**
*   **Input:** "...bilgisayarlı görü algoritmaları geliştiriliyor."
*   **Expected:** görü `(O - Ordinary Word)`
*   **Model's Prediction:** görü `(PER)`
*   **Reason:** The model lost the semantic context of the sentence. With a low confidence score, it hallucinated by tagging the technical term "görü" (vision) as a person's name (PER).

**5. Apostrophe and Suffix Confusion**
*   **Input:** "Mark Zuckerberg, Meta'nın gelecekteki..."
*   **Expected:** Meta `(ORG)`
*   **Model's Prediction:** None (Not found)
*   **Reason:** Being a relatively new and specific entity name not frequently seen in the training data, combined with the apostrophe and the genitive suffix (`'nın`), caused the model to miscalculate the word boundaries and assign it an 'O' tag.