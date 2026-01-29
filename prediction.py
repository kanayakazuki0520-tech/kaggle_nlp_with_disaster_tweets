import pandas as pd
import numpy as np

train_df = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')
submission = pd.read_csv('input/sample_submission.csv')
print(train_df.columns)
print(train_df.info())
print(test_df.columns)
print(test_df.info())
# print(train_df['target'].value_counts())

# # sample 
# print('"NLP with kaggle"の文字数:')
# print(len("NLP with kaggle")) # 空白文字hもカウントされる
# print('単語数:')
# print(len("NLP with kaggle".split())) # split()で空白文字で分割してリスト化し、その長さを取得

train_df['text_len'] = train_df['text'].str.len()
train_df['word_count'] = train_df['text'].str.split().str.len() # str.split()で各テキストを単語のリストに変換し、str.len()でそのリストの長さを取得
print(train_df[['text_len', 'word_count']].describe())
test_df['text_len'] = test_df['text'].str.len()
test_df['word_count'] = test_df['text'].str.split().str.len()
print(test_df[['text_len', 'word_count']].describe())

import matplotlib.pyplot as plt

min_len = min(train_df['text_len'].min(), test_df['text_len'].min())
max_len = max(train_df['text_len'].max(), test_df['text_len'].max())
bins_len = 15 # 図の棒の数

min_words = min(train_df['word_count'].min(), test_df['word_count'].min())
max_words = max(train_df['word_count'].max(), test_df['word_count'].max())
bins_words = 15 # ???

plt.figure(figsize=(14, 8))

# 図１
plt.subplot(2, 2, 1)
# plt.hist(train_df[train_df['target'] == 0]['text_len'], bins=bins_len, range=(min_len, max_len), alpha=0.5, label='Not Disaster', color='blue', edgecolor='black')
plt.hist(train_df.loc[train_df['target']==0, 'text_len'], bins=bins_len, range=(min_len, max_len), alpha=0.5, label='Not Disaster', color='blue', edgecolor='black')
plt.hist(train_df.loc[train_df['target']==1, 'text_len'], bins=bins_len, range=(min_len, max_len), alpha=0.5, label='Disaster', color='orange', edgecolor='black')
plt.title('Text Length Distribution by Target (Character Count)')
plt.xlabel('Number of Characters')
plt.ylabel('Frequency')
plt.legend()

# 図２
plt.subplot(2, 2, 2)
plt.hist(train_df.loc[train_df['target'] == 0, 'word_count'], bins=bins_words, range=(min_words, max_words), alpha=0.5, label='Not Disaster', color='blue', edgecolor='black')
plt.hist(train_df.loc[train_df['target'] == 1, 'word_count'], bins=bins_words, range=(min_words, max_words), alpha=0.5, label='Disaster', color='orange', edgecolor='black')
plt.title('Word Count Distribution by Target')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.legend()

# 図３
plt.subplot(2, 2, 3)
plt.hist(test_df.loc[:,'text_len'], bins=bins_len, range=(min_len, max_len), alpha=0.5, color='green', edgecolor='black')
plt.title('Text Length Distribution in Test Set (Character Count)')
plt.xlabel('Number of Characters')
plt.ylabel('Frequency')

# 図４
plt.subplot(2, 2, 4)
plt.hist(test_df['word_count'], bins=bins_words, range=(min_words, max_words), alpha=0.5, color='green', edgecolor='black')
plt.title('Word Count Distribution in Test Set')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# count_vectorizer(単語の出現回数をカウント)
from sklearn.feature_extraction.text import CountVectorizer

sample_data = {
    'text': [
        "I love machine learning. I love coding.",
        "Machine learning is amazing",
        "I love Kaggle",
        "Kaggle is great for machine learning"    
    ]
}
# sample_df = pd.DataFrame(sample_data)
# print(sample_df)
# sample_vectornizes = CountVectorizer()
# vectorized_text = sample_vectornizes.fit_transform(sample_df['text'])
# print(vectorized_text.toarray())
# print(sample_vectornizes.get_feature_names_out())
# sample_features = pd.DataFrame(vectorized_text.toarray(), columns=sample_vectornizes.get_feature_names_out())
# sample_df_with_features = pd.concat([sample_df, sample_features], axis=1)
# print("特徴量を追加したデータフレーム:")
# print(sample_df_with_features)

train_vectorizer = CountVectorizer(max_features = 100)
train_df_vectorized_text = train_vectorizer.fit_transform(train_df['text'])
train_df_features = pd.DataFrame(train_df_vectorized_text.toarray(), columns = train_vectorizer.get_feature_names_out())
print(train_df_features.columns)
train_df_with_features = pd.concat([train_df, train_df_features], axis=1)
print("特徴量を追加した学習データフレーム:")
print(train_df_with_features.head())
test_df_vectorized_text = train_vectorizer.transform(test_df['text'])
test_df_features = pd.DataFrame(test_df_vectorized_text.toarray(), columns = train_vectorizer.get_feature_names_out())
test_df_with_features = pd.concat([test_df, test_df_features], axis=1)
print("特徴量を追加したテストデータフレーム:")
print(test_df_with_features.head())

# n-gram(連続したn個の単語の組み合わせ)
# sample_df = pd.DataFrame(sample_data)
# print(sample_df)
# sample_vectornizes = CountVectorizer(ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b') # バイグラム
# vectorizezd_text = sample_vectornizes.fit_transform(sample_df['text'])
# print(vectorizezd_text.toarray())
# print(sample_vectornizes.get_feature_names_out())
# sample_vectornizes_features = pd.DataFrame(vectorizezd_text.toarray(), columns = sample_vectornizes.get_feature_names_out())
# sample_df_with_features = pd.concat([sample_df, sample_vectornizes_features], axis=1)
# print("特徴量を追加したデータフレーム:")
# print(sample_df_with_features)  

train_df = pd.DataFrame(train_df)
train_vectorizes = CountVectorizer(ngram_range=(1, 2), max_features=100)
train_vectorized_text = train_vectorizes.fit_transform(train_df['text'])
df_train_vectorized_features = pd.DataFrame(train_vectorized_text.toarray(), columns = train_vectorizes.get_feature_names_out())
print(df_train_vectorized_features.columns)
train_df_with_features = pd.concat([train_df, df_train_vectorized_features], axis=1)
print("特徴量を追加した学習データフレーム:")
print(train_df_with_features.head())
test_df = pd.DataFrame(test_df)
test_vectorized_text = train_vectorizes.transform(test_df['text'])
test_df_vectorized_features = pd.DataFrame(test_vectorized_text.toarray(), columns = train_vectorizes.get_feature_names_out())
test_df_with_features = pd.concat([test_df, test_df_vectorized_features], axis=1)
print("特徴量を追加したテストデータフレーム:")
print(test_df_with_features.head())

# TF-IDF(単語の重要度を考慮したベクトル化) 
from sklearn.feature_extraction.text import TfidfVectorizer
# sample_tfidf_vectorizer = TfidfVectorizer(max_features=100, token_pattern=r'(?u)\b\w+\b')
# sample_tfidf = sample_tfidf_vectorizer.fit_transform(sample_df['text'])
# sample_tfidf_features = pd.DataFrame(sample_tfidf.toarray(), columns = sample_tfidf_vectorizer.get_feature_names_out())
# print(sample_tfidf_features.columns)
# print(sample_tfidf_features)
# sample_df_with_tfidf = pd.concat([sample_df, sample_tfidf_features], axis=1)
# print("特徴量を追加したデータフレーム(TF-IDF):")
# print(sample_df_with_tfidf)

tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['text'])
df_train_tfidf_features = pd.DataFrame(X_train_tfidf.toarray(), columns = tfidf_vectorizer.get_feature_names_out())
print(df_train_tfidf_features.columns)
train_df_with_tfidf = pd.concat([train_df, df_train_tfidf_features], axis=1)
print("特徴量を追加した学習データフレーム(TF-IDF):")
print(train_df_with_tfidf.head())
test_tfidf = tfidf_vectorizer.transform(test_df['text'])
test_df_tfidf_features = pd.DataFrame(test_tfidf.toarray(), columns = tfidf_vectorizer.get_feature_names_out())
test_df_with_tfidf = pd.concat([test_df, test_df_tfidf_features], axis=1)
print("特徴量を追加したテストデータフレーム(TF-IDF):")
print(test_df_with_tfidf.head())

#csv出力
train_df_with_tfidf.to_csv('output/train_df_with_tfidf.csv', index=False)
test_df_with_tfidf.to_csv('output/test_df_with_tfidf.csv', index=False)
