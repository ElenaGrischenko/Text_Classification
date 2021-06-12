import tensorflow
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
import re

total_categories = len(df_train['Category'].unique())
descriptions = df_train['ClearText']
categories = df_train['Category']
# создаем единый словарь (слово -> число) для преобразования
tokenizer = Tokenizer()
tokenizer.fit_on_texts(descriptions.tolist())
#print(tokenizer.word_index)
# Преобразуем все описания в числовые последовательности, заменяя слова на числа по словарю.
X_train = tokenizer.texts_to_sequences(descriptions.tolist())
#количество слов в словаре
total_words = len(tokenizer.word_index)
print('В словаре {} слов'.format(total_words))

X_test = tokenizer.texts_to_sequences(df_test['ClearText'].tolist())
import keras
# количество наиболее часто используемых слов
num_words = 2000
print(u'Преобразуем описания заявок в векторы чисел...')
tokenizer = Tokenizer(num_words=num_words)
X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')
print('Размерность X_train:', X_train.shape)
print('Размерность X_test:', X_test.shape)
print(u'Преобразуем категории в матрицу двоичных чисел '
      u'(для использования categorical_crossentropy)')
y_train = keras.utils.to_categorical(df_train['Category']-1, 3)
y_test = keras.utils.to_categorical(df_test['Category']-1, 3)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
epochs = 10
#print(u'Збираємо модель...')
model = Sequential()
model.add(Dense(10, input_shape=(num_words,), activation="relu", name="layer1"))
model.add(Dense(15, input_shape=(num_words,), activation="sigmoid", name="layer2"))
model.add(Dense(8, input_shape=(num_words,), activation="relu", name="layer3"))
model.add(Dense(3,  activation="softmax", name="layer4"))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history_cnn = model.fit(X_train, y_train,
                batch_size=32,
                epochs=epochs,
                verbose=1,
                validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test,
                       batch_size=32, verbose=1)
print(score)
print(u'Оценка теста: {}'.format(score[0]))
print(u'Оценка точности модели: {}'.format(score[1]))

