import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from os import path


def processAge(data, cut_points, labels):
	data['Age'] = data['Age'].fillna(-0.5)
	data['Age_categories'] = pd.cut(data['Age'], cut_points, labels = labels)
	return data

data = pd.read_csv(path.join("bases/train.csv"), sep = ',', encoding='utf-8')

print(data.head())

dropcolumns = ['PassengerId', 'Name', 'Ticket', 'Cabin']
data = data.drop(dropcolumns, axis = 1)

cut_points = [-1, 0, 5, 12, 18, 35, 60, 100]
labels = ['Faltando', '0-4 anos', '5-12 anos', '12-18 anos', '18-35 anos', '35-60 anos', '60+ anos']
data = processAge(data, cut_points, labels)

print(data["Sex"].value_counts())
print(data["Pclass"].value_counts())
print(data["SibSp"].value_counts())

columns = ['Age', 'Pclass', 'SibSp', 'Fare']

for col in columns:
	boxplot = data.boxplot(column = col)

media = data['Age'].mean()

print('Media de idade:', media)

corr = data.corr()
corr.style.background_gradient(cmap = 'coolwarm')
print('\n\nCorrelacao dos atributos:\n')
print(corr)


pclass_pivot = data.pivot_table(index = 'Pclass', values = 'Survived')
pclass_pivot.plot.bar(rot = 0)

age_pivot = data.pivot_table(index = 'Age_categories', values = 'Survived')
age_pivot.plot.bar(rot = 0)

sex_pivot = data.pivot_table(index = 'Sex', values = 'Survived')
sex_pivot.plot.bar(rot = 0)

#plt.show()


died = data.loc[lambda data: data['Survived'] == 0]
survived = data.loc[lambda data: data['Survived'] == 1]

#print('\n\nMortos:', len(died), '\nSobreviventes:', len(survived))

print('\n\nMortos:', len(died))
print('Media e mediana da idade:', died['Age'].mean(), ', ', died['Age'].median())
print('Classes:')
print(died["Pclass"].value_counts())
print('\nSexo:')
print(died["Sex"].value_counts())
print('\nPorcentagem de mulheres:', (died["Sex"].value_counts()[1]/data["Sex"].value_counts()[1])*100)
print('Porcentagem de homens', (died["Sex"].value_counts()[0]/data["Sex"].value_counts()[0])*100)
print('\nEmbarque:')
print(died["Embarked"].value_counts())
print('\nMedia e mediana da tarifa:', died['Fare'].mean(), ', ', died['Fare'].median())

print('\n\nSobreviventes:', len(survived))
print('Media e mediana da idade:', survived['Age'].mean(), ', ',survived['Age'].median())
print('Classes:')
print(survived["Pclass"].value_counts())
print('\nSexo:')
print(survived["Sex"].value_counts())
print('\nPorcentagem de mulheres:', (survived["Sex"].value_counts()[0]/data["Sex"].value_counts()[1])*100)
print('Porcentagem de homens', (survived["Sex"].value_counts()[1]/data["Sex"].value_counts()[0])*100)
print('\nEmbarque:')
print(survived["Embarked"].value_counts())
print('\nMedia e mediana da tarifa:', survived['Fare'].mean(), ', ',survived['Fare'].median())
print('\n\n')

