from math import log, ceil 
import mmh3
from bitarray import bitarray
import matplotlib.pyplot as plt

def bloom_init(m, k, words_list):
    #створюємо масив бітів та заповнюємо 0
    arr = bitarray(m)
    arr.setall(0)
    
    for word in words_list:
        for ki in range(0, k, 1):
            digest = mmh3.hash(word, k) % m
            #міняємо значення 0 на 1 в масиві бітів 
            #на отриманих позиціях після хешування
            arr[digest] = 1
    return arr
def Bloom_Filter(df_train, p):
    categories = []
    for key,value in enumerate(df_train['Category'].unique()):
        categories.append(value)
    print(categories)
    total_categories = len(df_train['Category'].unique())
    print('Всего категорий: {}'.format(total_categories))
    #create train dict
    train_dict = {}
    for i in categories:
        train_dict[i] = set()
    #print(train_dict)
    for row in df_train.iterrows():
        #print(train_dict[row[1]['Category']])
        #print(set(list(row[1]['ClearText'])))
        #train_dict[row[1]['Category']] += row[1]['ClearText']
        train_dict[row[1]['Category']] = set.union(train_dict[row[1]['Category']], set(row[1]['ClearText']))
    #max number elements in all categories
    max_num_words = 0
    for category in  train_dict:
        #print(category, len(train_dict[category]))
        if max_num_words < len(train_dict[category]):
            max_num_words = len(train_dict[category])
    print('max num el in cat', max_num_words)
    #calculate m - size bit array
    m = round(- max_num_words * log(p) / pow(log(2), 2))
    #calculate hash func k
    k = ceil(m / max_num_words * log(2))
    blomm_filter_dict = {}
    for category in train_dict:
        blomm_filter_dict[category] = bloom_init(m, k, train_dict[category])
    return blomm_filter_dict, k, m
#функція для визначення приналежності слова до категорії
#якщо слово належить до категорії, то функція певерне True
#якщо ж нф, то False
def word_in_bloom(word, number_hash_functions, size, arr):
    result = 1
    for K in range(0, number_hash_functions, 1):
        #print(K)
        #if arr[hash(word, K, size)] == 0:
        d = mmh3.hash(word,K) % size
        #print(d, mmh3.hash(word,K))
        #print(arr[d] == 0)
        '''if arr[d] == 0:
            return False
            break
    return True'''
        result = result * arr[d]
    return result

def bloom_predict(bloom_arr_dict, df_test, k, m):
#прогнозування приналежності текстів до категорії
    list_predict = []
    #print(bloom_arr_dict['sport'])
    for row in df_test.iterrows():
        #print(row[1]['ClearText'])
        predict = []
        #ймовірність приналежності тексту до кожної категорії
        for category in bloom_arr_dict.keys():
            count_words_in_category = 0
            #рахуємо кількість слів, що 100% входять до категорії
            for word in row[1]['ClearText']:
                if word_in_bloom(word, k, m, bloom_arr_dict[category]) == 1:
                    count_words_in_category = count_words_in_category + 1
            #зберігаємо ймовірність в словник
            predict.append(round(count_words_in_category/len(row[1]['ClearText']), 5))
        #знаходимо максимальну ймовірність та відповідну категорію   
        #print(predict_dict)
        
        max_predict = max(predict)
        if predict.count(max_predict) == 1:
            list_predict.append(list(bloom_arr_dict.keys())[predict.index(max_predict)])
        elif predict.count(max_predict) == len(predict):
            list_predict.append(list(bloom_arr_dict.keys()))
        else:
            idx_start = 0
            list_temp = []
            for i in range(0, predict.count(max_predict), 1):
                idx = predict.index(max_predict, idx_start)
                list_temp.append(list(bloom_arr_dict.keys())[idx])
                idx_start = idx + 1
            list_predict.append(list_temp)
    return list_predict

bloom_arr_dict, k, m = Bloom_Filter(df_train, i)
list_predict = bloom_predict(bloom_arr_dict, df_test, k, m)

list_real = list(df_test['Category'])
correct_predict = 0
incorrect_predict = 0
for i in range(0, len(list_real), 1):
    if list_real[i] == list_predict[i]:
        correct_predict += 1
accuracy = round(correct_predict/len(list_predict), 4)*100
