import pandas as pd

# Загрузка данных
data = pd.read_csv('BTC-USD.csv')

# Просмотр первых нескольких строк данных
print(data.head())

import talib

# Создание признаков на основе скользящих средних
data['SMA_10'] = talib.SMA(data['Close'], timeperiod=10)
data['SMA_20'] = talib.SMA(data['Close'], timeperiod=20)

# Просмотр первых нескольких строк данных
print(data.head())

# Определение целевой переменной
data['Target'] = (data['Close'] > data['Open']).astype(int)

# Просмотр первых нескольких строк данных
print(data.head())

# Разделение данных на обучающую и тестовую выборки
train_size = int(len(data) * 0.8)
train= data.iloc[:train_size]
test = data.iloc[train_size:]

# Разделение на признаки и целевую переменную
X_train = train[['SMA_10', 'SMA_20']]
y_train = train['Target']
X_test = test[['SMA_10', 'SMA_20']]
y_test = test['Target']

from sklearn.linear_model import LogisticRegression

# Создание модели
model = LogisticRegression()

# Обучение модели
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

# Предсказание на тестовых данных
y_pred = model.predict(X_test)

# Оценка производительности модели
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ', accuracy)

# Вывод важности признаков
importance = pd.DataFrame({'feature': X_train.columns, 'importance': model.coef_[0]})
print(importance)
