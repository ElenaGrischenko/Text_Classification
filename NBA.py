import pandas as pd
from math import log 

def NBA_init(df_train):
    categories = []
    for key,value in enumerate(df_train['Category'].unique()):
        categories.append(value)
    print(categories)
    # Запишем в новую колонку числовое обозначение категории 
    total_categories = len(df_train['Category'].unique())
    print('Всего категорий: {}'.format(total_categories))
    dict_words = {}
    for i in categories:
        dict_words[i] = []    
    set_words = set()
    all_words = []
    for row in df_train.iterrows():
        dict_words[row[1]['Category']] += row[1]['ClearText']
        set_words = set.union(set_words, set(row[1]['ClearText']))
        all_words += row[1]['ClearText']
   
    #створюємо датафрейм, де колонки - категорії, а індекси - унікальні слова
    df_words = pd.DataFrame(
        columns = categories,
        index = set_words)
    #заповнюємо структуру 0
    df_words = df_words.fillna(0.0)
    print('df creating')
    #рахуємо приналежності слова до класу
    for row in df_words.iterrows():
        #для кожної категорії
        for category in df_words.columns:
            #чисельник
            num = dict_words[category].count(row[0]) + 1
            #знаменник
            denom = len(set(all_words)) + len(dict_words[category])
            df_words[category][row[0]] = round(log(num / denom, 2), 4)  
    print('df text creating')        
    #підрахунок ймовірності категорії
    #групуємо тестову вибірку по категоріям
    df_train_group = df_train.groupby(by=["Category"]).count()
    dict_category = {}

    #зберігаємо значення кількості текстів для кажної категорї
    for row in df_train.groupby(by=["Category"]).count().iterrows():
        dict_category.update({row[0]: row[1]['Text']})
    return df_words, dict_category

#функція для прогнозування приналежності тексту до категорії
def predict_bayesian(df_test, dict_category, df_words):
    #прогнозування приналежності текстів до категорії
    list_predict = []
    for row in df_test.iterrows():
        predict_dict = {}
        for category in dict_category:
            p_text = log(dict_category[category] / sum(dict_category.values()), 2)
            for word in row[1]['ClearText']:
                if word in df_words.index:
                    p_text += df_words[category][word]
            predict_dict.update({category: p_text})
        max_predict = max(predict_dict.values())
        list_predict.append(list(predict_dict.keys())[list(predict_dict.values()).index(max_predict)])
    return list_predict

df_words, dict_category = NBA_init(df)
list_predict = predict_bayesian(df_test_r, dict_category, df_words)
list_real = list(df_test_r['Category'])
correct_predict = 0
incorrect_predict = 0
for i in range(0, len(list_predict), 1):
    if list_real[i] == list_predict[i]:
        correct_predict += 1
    else:
        incorrect_predict += 1
accuracy = round(correct_predict/(correct_predict + incorrect_predict), 4)*100
