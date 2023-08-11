import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_diabetes

class ComplaintClassifier:
   def __init__(self, excel_file):
       self.excel_file = excel_file
       self.load_data()
       self.preprocess_data()
       #self.preprocess_data()
       self.encode_categories()
       self.train_model()
       self.analyze_common_words()
       self.outlier_detection()
   def load_data(self):
       excel_data = pd.read_excel(self.excel_file, sheet_name='Sheet1', usecols='B,G,H,I')
       self.df = excel_data.copy()


   def preprocess_data(self):
       sınıflarRemove = ['BÜRO ŞIKAYETI', 'ÖDEME PLANI-SMS', 'PARA İADESI', 'DIĞER', 'TEMSILCI ŞIKAYETI-ULAŞILAMAMASI']
       word_count = lambda text: len(text.split())
       self.df = self.df[self.df['Şikayet'].apply(word_count) >= 20]
       total = self.df['Şikayet'].notnull().sum()  # print(excel_data)

       xr = round((total / len(self.df) * 100), 1)
       print(xr)
       self.df = self.df.dropna()
       self.df['Kategori'] = self.df['Kategori'].str.upper()
       self.df = self.df[~self.df['Kategori'].isin(sınıflarRemove)]
       self.df= self.df[self.df['Hesap No'].apply(lambda x: x not in ['Evet', 'Hayır'])]



   def correlation_Words(self):
       # finding the most correlated terms with each category
       N = 3

       for Kategori, category_id in sorted(self.category_id_df.items()):
           features_chi2 = chi2(self.features, self.labels == category_id)
           indices = np.argsort(features_chi2[0])
           feature_names = np.array(self.tfidf.get_feature_names_out())[indices]
           unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
           bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
           trigrams = [v for v in feature_names if len(v.split(' ')) == 3]

           print("\n ==> %s:" % (Kategori))
           print("  * Most Correlated Unigrams are: %s" % (', '.join(unigrams[-N:])))
           print("  * Most Correlated Bigrams are: %s " % (', '.join(bigrams[-N:])))
           print("  * Most Correlated Trigrams are: %s " % (', '.join(trigrams[-N:])))

   def analyze_common_words(self):
       self.common_words_per_category = {}
       for category, category_id  in sorted(self.category_id_df.items()):
           features_chi2 = chi2(self.features, self.labels == category_id)
           indices = np.argsort(features_chi2[0])
           feature_names = np.array(self.tfidf.get_feature_names_out())[indices]
           unigrams = [v for v in feature_names if len(v.split(' ')) == 1]

           common_words = {'most_common': [], 'least_common': []}

           for i, idx in enumerate(indices):
               word = feature_names[i]
               count = int(features_chi2[0][idx])
               common_words['most_common'].append((word, count))
               common_words['least_common'].append((word, count))

           self.common_words_per_category[category] = common_words

       return self.common_words_per_category


   def encode_categories(self):
       # finding unique category values
       print(self.df.Kategori.unique())

       unique_categories = self.df['Kategori'].unique()

       # encoding categories
       self.df['category_id'] = self.df['Kategori'].factorize()[0]
       self.category_id_df = self.df[['Kategori', 'category_id']].drop_duplicates()

       self.category_id_df = dict( self.category_id_df.values)
       self.id_to_category = dict( zip(self.category_id_df.values(), self.category_id_df.keys()))

       # print(category_id_df)
       print(self.df.head())

   def outlier_detection(self):
       print("OLD SHAPE: ", self.df.shape)
       q1 = self.df.quantile(0.30)
       q3 = self.df.quantile(0.70)
       IQR = q3 - q1
       lower_bound = q1 - 1.5 * IQR
       upper_bound = q3 - 1.5 * IQR

       outliers = self.df[(self.df < lower_bound) | (self.df > upper_bound)]

       outliers_indices = set(outliers.dropna().index)
       rare_words_indices = []
       print(31)
       for category, common_words in self.common_words_per_category.items():
           for word, count in common_words['least_common']:
               if count < 3:
                   word_index = self.tfidf.vocabulary_[word]
                   rare_words_indices.append(word_index)

       indices_to_remove = list(set(rare_words_indices + list(outliers_indices)))
       self.df = self.df[~self.df.index.isin(indices_to_remove)]

       print("NEW SHAPE: ", self.df.shape)
   def cross_validation(self):
       CV = 15
       models = [LinearSVC()]
       self.cv_df = pd.DataFrame(index=range(CV * len(models)))

       entries = []
       for model in models:
           model_name = model.__class__.__name__
           accuracies = cross_val_score(model, self.features, self.labels, scoring='accuracy', cv=CV)
           for fold_idx, accuracy in enumerate(accuracies):
               entries.append((model_name, fold_idx, accuracy))

       self.cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

       mean_accuracy = self.cv_df.groupby('model_name').accuracy.mean()
       std_accuracy = self.cv_df.groupby('model_name').accuracy.std()
       acc = pd.concat([mean_accuracy, std_accuracy], axis=1, ignore_index=True)
       acc.columns = ['Mean Accuracy', 'Standard deviation']
       print(acc)
       #return self.cv_df

   def train_model(self):
       # Model training steps go here
       self.turkish_stop_words = ['bir', 'fakat', 'lakin', 'ancak', 'acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri',
                             'birkaç', 'birşey', 'biz', 'bu', 'çok',
                             'çünkü', 'da', 'daha', 'de', 'defa', 'diye', 'eğer', 'en', 'gibi', 'hem', 'hep', 'hepsi',
                             'her', 'hiç', 'için',
                             'ile', 'ise', 'kez', 'ki', 'kim', 'mı', 'mu', 'mü', 'nasıl', 'ne', 'neden', 'nerde',
                             'nerede', 'nereye', 'niçin', 'niye',
                             'o', 'sanki', 'şey', 'siz', 'şu', 'tüm', 've', 'veya', 'ya', 'yani', 'mail', 'com', 'cc',
                             'merhaba', 'didem', 'hanım', 'ad',
                             'soyad', 'T.C.K.N.', 'Email', 'Tel', '.', ',', '#', '+', '@']
       self.tfidf = TfidfVectorizer(sublinear_tf=True, min_df=4, ngram_range=(1, 3), stop_words=self.turkish_stop_words)
       self.features = self.tfidf.fit_transform(self.df.Şikayet).toarray()
       self.labels = self.df['category_id']  #self.df['category_id']
       #self.labels = self.df.category_id
       self.model = LinearSVC()
       self.model.fit(self.features, self.labels)

       self.encode_categories()
       self.correlation_Words()

   def evaluate_model(self):
       X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=0.25, random_state=1)
       y_pred = self.model.predict(X_test)

       test_accuracy = accuracy_score(y_test, y_pred)
       train_accuracy = self.model.score(X_train, y_train)

       print('Test Accuracy : {:.3f}'.format(test_accuracy))
       print('Train Accuracy : {:.3f}'.format(train_accuracy))

       model_l1 = LinearSVC(penalty='l1', dual=False, C=1.0, random_state=1)
       model_l1.fit(X_train, y_train)

       # L2 Düzenlileştirme ile LinearSVC modeli oluşturma
       model_l2 = LinearSVC(penalty='l2', C=1.0, random_state=1)
       model_l2.fit(X_train, y_train)

       # Test verisi üzerinde tahminler yapma
       y_pred_l1 = model_l1.predict(X_test)
       y_pred_l2 = model_l2.predict(X_test)

       # Model performansını değerlendirme
       accuracy_l1 = accuracy_score(y_test, y_pred_l1)
       accuracy_l2 = accuracy_score(y_test, y_pred_l2)

       print("L1 Düzenlileştirme ile Model Doğruluk: {:.3f}".format(accuracy_l1))
       print("L2 Düzenlileştirme ile Model Doğruluk: {:.3f}".format(accuracy_l2))

   def classify_complaint(self, complaint_text):
       # Classification steps go here
       complaint_vector = self.tfidf.transform([complaint_text]).toarray()
       predicted_category_id = self.model.predict(complaint_vector)[0]
       predicted_category = self.id_to_category[predicted_category_id]
       return predicted_category

if __name__ == "__main__":
   excel_file_path = 'hepsi.xlsx'
   complaint_classifier = ComplaintClassifier(excel_file_path)
   complaint_classifier.outlier_detection()
   complaint_classifier.cross_validation()
   complaint_classifier.preprocess_data()
   complaint_classifier.train_model()
   complaint_classifier.evaluate_model()





   new_complaint = ""
   predicted_category = complaint_classifier.classify_complaint(new_complaint)
   print(f"Predicted Category: {predicted_category}")

   common_words_per_category = complaint_classifier.analyze_common_words()
   for category, common_words in common_words_per_category.items():
       print(f"\nKategori:{category}")
       print("En çok geçen kelimeler:")
       for word, count in common_words['most_common']:
           print(f"{word}:{count} kez")

       """print("En Az Geçen Kelimeler:")
       for word, count in common_words['least_common']:
           print(f"{word}: {count} kez")"""