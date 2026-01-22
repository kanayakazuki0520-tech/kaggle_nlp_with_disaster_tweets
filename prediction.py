import pandas as pd
import numpy as np

train_df = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')
submission = pd.read_csv('input/sample_submission.csv')
# print(train_df.columns)
# print(train_df.info())
# print(test_df.columns)
# print(train_df['target'].value_counts())

# # sample 
# print('"NLP with kaggle"の文字数:')
# print(len("NLP with kaggle")) # 空白文字hもカウントされる
# print('単語数:')
# print(len("NLP with kaggle".split())) # split()で空白文字で分割してリスト化し、その長さを取得

train_df['text_len'] = train_df['text'].str.len()
train_df['word_count'] = train_df['text'].str.split().str.len() # ???(.str.len())? 
print(train_df[['text_len', 'word_count']].describe())

import matplotlib.pyplot as plt

min_len = train_df['text_len'].min()
max_len = train_df['text_len'].max()
bins_len = 15 # ???

min_words = train_df['word_count'].min()
max_words = train_df['word_count'].max()
bins_words = 15 # ???

print(train_df[train_df['target']==0]['text_len'].head())
plt.figure(figsize=(12, 5))

# 図１
plt.subplot(2, 2, 1)
plt.hist(train_df[train_df['target'] == 0]['text_len'], bins=bins_len, range=(min_len, max_len), alpha=0.5, label='Not Disaster', color='blue', edgecolor='black')
plt.hist(train_df[train_df['target'] == 1]['text_len'], bins=bins_len, range=(min_len, max_len), alpha=0.5, label='Disaster', color='orange', edgecolor='black')
plt.title('Text Length Distribution by Target (Character Count)')
plt.xlabel('Number of Characters')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(2, 2, 2)
plt.hist(test_df[test_df['target']== 1][train_df['word_count']])
plt.show()
