import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.utils import shuffle
from keras.utils import to_categorical
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from keras.models import load_model
from os import path

##Funcoes auxiliares

def myFactorize(data, column, mapping):
	for i in range(len(data)):
		data.at[i, column] = mapping.tolist().index(data[column][i])
	return data

def normalize_data(data, attribute):
	attribute_list = data[attribute].tolist()
	min_value = float(data[attribute].min())
	max_value = float(data[attribute].max())

	if min_value < 0:
		min_value = 0
	
	for i in range(len(data)):
		if float(data[attribute][i]) >= 0:
			data.at[i, attribute] = (float(data[attribute][i]) - min_value) / (max_value - min_value)

	data.loc[:, attribute] = data.loc[:, attribute].apply(lambda x: "%.5f" % x)

	return data

def processAge(data, cut_points, labels):
	data['Age'] = data['Age'].fillna(-0.5)
	data['Age_categories'] = pd.cut(data['Age'], cut_points, labels = labels)
	return data


##Inicio do codigo

data = pd.read_csv(path.join("bases/train.csv"), sep = ',', encoding='utf-8')

print(data.head())

##Preparando os dados para treino

dropcolumns = ['PassengerId', 'Name', 'Ticket', 'Cabin']
data = data.drop(dropcolumns, axis = 1)

data['Embarked'], emb_mapping = data['Embarked'].factorize(sort = True)

data['Sex'], sex_mapping = data['Sex'].factorize(sort = True)

cut_points = [-1, 0, 5, 12, 18, 35, 60, 100]
labels = ['Faltando', 'Bebe', 'Crianca', 'Adolescente', 'Jovem Adulto', 'Adulto', 'Idoso']
data = processAge(data, cut_points, labels)

data['Age_categories'], age_mapping = data['Age_categories'].factorize(sort = True)
data['Fare'].fillna(data['Fare'].dropna().median(), inplace = True)

data = data.dropna()
data = data.reset_index()

y = data['Survived']
x = data.drop(['Survived', 'index', 'Age'], axis = 1)

for col in x.columns:
	x = normalize_data(x, col)

print(x.head())
print(x.columns)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state = 83)

y_train = y_train.values
x_train = x_train.values
x_val = x_val.values
y_val = y_val.values


print('No train:')
counter = 0
for i in tqdm(range(len(y_train))):
	if not np.isnan(y_train[i]):
		if y_train[i] == 0:
			counter = counter + 1
    
print("mortos: ", counter)
print("sobreviventes: ", len(x_train) - counter)

print('Na validacao:')
counter = 0
for i in tqdm(range(len(y_val))):
	if not np.isnan(y_train[i]):
		if y_val[i] == 0:
			counter = counter + 1
    
print("mortos: ", counter)
print("sobreviventes: ", len(x_val) - counter)


##Treino

##Random Forest
random_forest = RandomForestClassifier(n_estimators = 100)
random_forest.fit(x_train, y_train)
random_forest.score(x_train, y_train)
acc_random_forest = round(random_forest.score(x_train, y_train) * 100, 2)
acc_random_forest_teste = round(random_forest.score(x_val, y_val) * 100, 2)
print('\n\nAcuracia random forest treino:', acc_random_forest)
print('Acuracia random forest validacao:', acc_random_forest_teste)


##Arvore de Decisao
decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)
acc_decision_tree_teste = round(decision_tree.score(x_val, y_val) * 100, 2)
print('\n\nAcuracia decision tree treino:', acc_decision_tree)
print('Acuracia decision tree validacao:', acc_decision_tree_teste)

##SGD
sgd = SGDClassifier()
sgd.fit(x_train, y_train)
acc_sgd = round(sgd.score(x_train, y_train) * 100, 2)
acc_sgd_teste = round(sgd.score(x_val, y_val) * 100, 2)
print('\n\nAcuracia SGD treino:', acc_sgd)
print('Acuracia SGD validacao:', acc_sgd_teste)

##kNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
acc_knn = round(knn.score(x_train, y_train) * 100, 2)
acc_knn_teste = round(knn.score(x_val, y_val) * 100, 2)
print('\n\nAcuracia kNN treino:', acc_knn)
print('Acuracia kNN validacao:', acc_knn_teste)

##Treinamento da rede neural

y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)
y_train = to_categorical(y_train, num_classes = 2)
y_val = to_categorical(y_val, num_classes = 2)

'''class_weight = {0: 1.,
                1: 1.5}

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units = 512, activation = 'relu', input_dim = 7),
    tf.keras.layers.Dropout(rate = 0.4),    
    tf.keras.layers.Dense(units = 512, activation = 'relu'),
    tf.keras.layers.Dense(units = 256, activation = 'relu'),
    tf.keras.layers.Dropout(rate = 0.2),
    tf.keras.layers.Dense(units = 128, activation = 'relu'),
    tf.keras.layers.Dense(units = 2, activation = 'softmax')
])

model.summary()

list_callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'acc', patience = 50, verbose = 1),
                 tf.keras.callbacks.ModelCheckpoint('modelos/melhor_modelo_ann2.hdf5', monitor = 'val_acc', verbose = 1, save_best_only = True)]

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size = 32, epochs = 200, class_weight = class_weight, 
							validation_data = (x_val, y_val), callbacks = list_callbacks)

print(model.evaluate(x_val, y_val))
model.save('modelos/modelo_ann2.h5')'''


##Apresentando resultado dos modelos

melhor_modelo = load_model('modelos/melhor_modelo_ann.hdf5')
print('\n\nAcuracia rede neural treino:', melhor_modelo.evaluate(x_train, y_train))
print('Acuracia rede neural validacao:', melhor_modelo.evaluate(x_val, y_val))
print('\n\nAcuracia random forest treino:', acc_random_forest)
print('Acuracia random forest validacao:', acc_random_forest_teste)
print('\n\nAcuracia decision tree treino:', acc_decision_tree)
print('Acuracia decision tree validacao:', acc_decision_tree_teste)
print('\n\nAcuracia SGD treino:', acc_sgd)
print('Acuracia SGD validacao:', acc_sgd_teste)
print('\n\nAcuracia kNN treino:', acc_knn)
print('Acuracia kNN validacao:', acc_knn_teste)


##Aplicando o modelo na base de testes

test_df = pd.read_csv(path.join("bases/test.csv"), sep = ',', encoding='utf-8')

dropcolumns = ['PassengerId', 'Name', 'Ticket', 'Cabin']
x_test  = test_df.copy()
dropcolumns = ['PassengerId', 'Name', 'Ticket', 'Cabin']
x_test = x_test.drop(dropcolumns, axis = 1)

x_test = myFactorize(x_test, 'Embarked', emb_mapping)
x_test = myFactorize(x_test, 'Sex', sex_mapping)

cut_points = [-1, 0, 5, 12, 18, 35, 60, 100]
labels = ['Faltando', 'Bebe', 'Crianca', 'Adolescente', 'Jovem Adulto', 'Adulto', 'Idoso']
x_test = processAge(x_test, cut_points, labels)

x_test['Age_categories'] = x_test['Age_categories'].factorize(sort = True)[0]
index = x_test['Fare'].index[x_test['Fare'].apply(np.isnan)]
x_test.at[index, 'Fare'] = x_test['Fare'].median()

x_test = x_test.drop(['Age'], axis = 1)

for col in x_test.columns:
	x_test = normalize_data(x_test, col)

##Previsoes

Y_pred = []
Y_pred.append(random_forest.predict(x_test))
Y_pred.append(decision_tree.predict(x_test))
Y_pred.append(sgd.predict(x_test))
Y_pred.append(knn.predict(x_test))

predictions = [acc_random_forest_teste, acc_decision_tree_teste, acc_sgd_teste, acc_knn_teste]
best_pred = max(predictions)
best_Y_pred = Y_pred[predictions.index(best_pred)]

previsao = pd.DataFrame({'PassengerId': test_df['PassengerId'],
						'Name': test_df['Name'],
						'Sex': test_df['Sex'],
						'Age': test_df['Age'],
						'Pclass': test_df['Pclass'],
						'Fare': test_df['Fare'],
						'Survived': best_Y_pred})

previsao.to_csv('previsao.csv', index = False)