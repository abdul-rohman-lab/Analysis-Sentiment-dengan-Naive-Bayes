# Analysis-Sentiment-dengan-Naive-Bayes

---
jupyter:
  colab:
    name: 17. Analysis Sentiment - Naive Bayes.ipynb
  kernelspec:
    display_name: Python 3
    name: python3
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .markdown id="_UFOZtDjCHQF"}
# **Analysis Sentiment - Naive Bayes**

Sekarang kita akan mencoba untuk melakukan Analsis sentiment menggunakan
salah satu metode Supervised Learning yaitu Naive Bayes. Seperti yang
kita tahu bahwa pada supervised learning kita membutuhkan dataset agar
mesin kita dapat belajar jadi pastika kita mempunyai dataset sudah
memiliki dataset tersebut. Bisa dengan cara melakukan analisis sentiment
terlebih dahulu dengan menggunakan perbandingan antara kata
negatif-positif seperti kemarin, atau dengan mencari dataset yang sudah
diberi label di internet.

Pada kesempatan ini kalian bisa menggunakan dataset tweet di bawah,
untuk memperingan kinerja pada kesempatan pertama ini maka diberikan
dataset dengan ukuran kecil
:::

::: {.cell .markdown id="lRp75YehCKae"}
ini untuk dataset tweet yang diguakan

[dataset_tweet_2](https://blog.sanbercode.com/wp-content/uploads/2020/09/dataset_tweet_2.csv)
:::

::: {.cell .markdown id="yJ_p9GEaCT9Y"}
Mari kita coba baca dataset kita lalu membaginya kedalam dua atribut,
yaitu tweet dan label.
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="mqn-CIMqBaH1" outputId="92906fc0-39a2-4f74-b425-9cf1dda2a4a6"}
``` python
import pandas as pd
import csv

file = 'dataset_tweet_2.csv'

token_data = open(file)
tokens = csv.reader(token_data, delimiter=';')
tweets = []
label = []
for row in tokens:
    tweets.append(row[0])
    label.append(int(row[1].replace(',','')))

df = pd.DataFrame(columns=['tweets','label'])
df['tweets'] = tweets
df['label'] = label

print (df)
```

::: {.output .stream .stdout}
                                                   tweets  label
    0   rt @napqilla: no 1, 3 ambisinya menguasai raky...      1
    1   rt @pandji: nah gue pikir sentimen petahana ok...      1
    2   rt @pandji: urutan pertama best moment #debat2...      1
    3   rt @pandji: ini artikel yg menjelaskan ternyat...      1
    4   rt @mrtampi: agus makin santai.\nahok makin sa...      0
    ..                                                ...    ...
    76  rt @pandji: nah gue pikir sentimen petahana ok...      0
    77  rt @josua_tm: ibu sylvi adalah contoh bahwa wa...      1
    78  besok saya ajak kesana saja, saya udah survei ...      1
    79  benerr bgt.. dan tidak mengajak penonton ikut ...      1
    80  rt @gandy_koz: pak anis,kl pas libur lebaran i...      1

    [81 rows x 2 columns]
:::
:::

::: {.cell .markdown id="xA51-iW0D01d"}
lalu selanjutnya kita akan melakukan pembersihan tweet di atas, kita
akan memanfaatkan modul stemming pada sastrawi, jadi jangan lupa untuk
menambahkan import pada bagian library. pertama kita akan melakukan case
folding lalu kita akan melakukan stemming.
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="jktA9wogkowi" outputId="4fd95c5f-2037-41cd-d2fe-789430fcee1b"}
``` python
pip install sastrawi
```

::: {.output .stream .stdout}
    Collecting sastrawi
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="Pcel_O2BD2jM" outputId="d3887ea1-e04a-4401-c867-10986e26c795"}
``` python
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


import re,string

clean_tweets = []
for tweet in tweets:
    def hapus_tanda(tweet): 
        tanda_baca = set(string.punctuation)
        tweet = ''.join(ch for ch in tweet if ch not in tanda_baca)
        return tweet
    
    tweet=tweet.lower()
    tweet = re.sub(r'\\u\w\w\w\w', '', tweet)
    tweet=re.sub(r'http\S+','',tweet)
    #hapus @username
    tweet=re.sub('@[^\s]+','',tweet)
    #hapus #tagger 
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #hapus tanda baca
    tweet=hapus_tanda(tweet)
    #hapus angka dan angka yang berada dalam string 
    tweet=re.sub(r'\w*\d\w*', '',tweet).strip()
    
    #stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    tweet = stemmer.stem(tweet)
    clean_tweets.append(tweet)

df['clean'] = clean_tweets
print(df.head())
```

::: {.output .stream .stdout}
                                                  tweets  ...                                              clean
    0  rt @napqilla: no 1, 3 ambisinya menguasai raky...  ...  rt no ambisi kuasa rakyat ambisi layan rakyat ...
    1  rt @pandji: nah gue pikir sentimen petahana ok...  ...  rt nah gue pikir sentimen tahana oke di malam ...
    2  rt @pandji: urutan pertama best moment #debat2...  ...  rt urut pertama best moment pak basuki misahin...
    3  rt @pandji: ini artikel yg menjelaskan ternyat...  ...  rt ini artikel yg jelas nyata di yg dapet resp...
    4  rt @mrtampi: agus makin santai.\nahok makin sa...  ...  rt agus makin santainahok makin santunnanies m...

    [5 rows x 3 columns]
:::
:::

::: {.cell .markdown id="6Hw7zS-Mlg7J"}
# **Machine Learning & Data Teks** {#machine-learning--data-teks}

Seperti yang kita tahu untuk dapat menggunakan sebuah model machine
learning maka kita membutuhkan data dengan tipe numerik atau integer,
akan tetapi salah satu satu feature yang kita miliki merupakan data
teks. Nah pada kesempatan ini kita akan memanfaatkan metode perubahan
data teks - matriks yang sudah kita pelajari sebelumnya, yaitu TF-IDF

Mari kita mulai, kita akan memanfaatkan metode TfidVectorizer pada
library sklearn dan gaussian Naive Bayes.
:::

::: {.cell .code id="1oQwYf5-ljKR"}
``` python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB

vectorizer = TfidfVectorizer (max_features=2500)
model_g = GaussianNB()
```
:::

::: {.cell .markdown id="x69vY59hl7gP"}
Lalu kita ubah data clean_tweet kita ke dalam bentuk TFIDF Vectorizer
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="TfF9myyQl4z5" outputId="b4d9fbd7-6405-4c37-f86e-22fc0d2f8ddf"}
``` python
v_data = vectorizer.fit_transform(df['clean']).toarray()

print(v_data)
```

::: {.output .stream .stdout}
    [[0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     ...
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]]
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="nW-Np2YXmHh0" outputId="cf98b904-ac9d-47d5-9537-351140e7ef6a"}
``` python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(v_data, df['label'], test_size=0.2, random_state=0)
model_g.fit(X_train,y_train)
```

::: {.output .execute_result execution_count="14"}
    GaussianNB(priors=None, var_smoothing=1e-09)
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="LOPW3Wvckdeb" outputId="22f76c75-8e8d-41f5-bd73-75badc10b951"}
``` python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_preds = model_g.predict(X_test)

print(confusion_matrix(y_test,y_preds))
print(classification_report(y_test,y_preds))
print('nilai akurasinya adalah ',accuracy_score(y_test, y_preds))
```

::: {.output .stream .stdout}
    [[8 4]
     [1 4]]
                  precision    recall  f1-score   support

               0       0.89      0.67      0.76        12
               1       0.50      0.80      0.62         5

        accuracy                           0.71        17
       macro avg       0.69      0.73      0.69        17
    weighted avg       0.77      0.71      0.72        17

    nilai akurasinya adalah  0.7058823529411765
:::
:::

::: {.cell .markdown id="s0JrFYhXnvcS"}
Untuk melakukan prediksi jangan lupa untuk mengubah 0 dan 1 menjadi
sebuah parameter string pada akhir langkah.
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="DVxWJVMfpaZn" outputId="5777efa7-99ca-46b9-a6f8-61e21d498849"}
``` python
y_preds
```

::: {.output .execute_result execution_count="17"}
    array([1])
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="rOjHhav0nwUp" outputId="1b9d6930-c796-4076-e796-2ce8361abb77"}
``` python
tweet = ''
v_data = vectorizer.transform([tweet]).toarray()
y_preds = model_g.predict(v_data)

# dengan asumsi bahwa 1 merupan label positif
if y_preds == 1:
  print('positif')
else:
  print('negatif')
```

::: {.output .stream .stdout}
    positif
:::
:::
