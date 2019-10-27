import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from os import path

def normalize_data(data, attribute):
	attribute_list = data[attribute].tolist()
	min_value = data[attribute].min()
	max_value = data[attribute].max()

	if min_value < 0:
		min_value = 0
	
	for i in range(len(data)):
		if np.isnan(data[attribute][i]):
			continue
		if data[attribute][i] >= 0:
			data.at[i, attribute] = (data[attribute][i] - min_value) / (max_value - min_value)

	data.loc[:, attribute] = data.loc[:, attribute].apply(lambda x: "%.5f" % x)

	return data


data = pd.read_csv("previsao.csv", sep = ',', encoding='utf-8')

died = data.loc[lambda data: data['Survived'] == 0]
survived = data.loc[lambda data: data['Survived'] == 1]

print('\n\nMortos:', len(died), '\nSobreviventes:', len(survived))

print('\n\nMortos:')
print('\nMedia e mediana da idade:', died['Age'].mean(), ', ', died['Age'].median())
print('\nClasses:')
print(died["Pclass"].value_counts())
print('\nSexo:')
print(died["Sex"].value_counts())
print('\nMedia e mediana da tarifa:', died['Fare'].mean(), ', ', died['Fare'].median())

print('\n\nSobreviventes:')
print('\nMedia e mediana da idade:', survived['Age'].mean(), ', ',survived['Age'].median())
print('\nClasses:')
print(survived["Pclass"].value_counts())
print('\nSexo:')
print(survived["Sex"].value_counts())
print('\nMedia e mediana da tarifa:', survived['Fare'].mean(), ', ',survived['Fare'].median())


print('\n\n')

