import matplotlib.pyplot as plt
import pandas as pd
from os import path


data_order = pd.read_csv(path.join("bases/order_products__train.csv"), sep = ',', encoding='utf-8')

data_products = pd.read_csv(path.join("bases/products.csv"), sep = ',', encoding='utf-8')



##Item a

produtos_vendidos = data_order['product_id'].value_counts()

#print(produtos_vendidos)

print('\nItem a - Produtos mais vendidos')
for i in range (10):
	produto = data_products.loc[data_products['product_id'] == produtos_vendidos.keys()[i]]
	print(i + 1, 'º\nproduct_id:', produtos_vendidos.keys()[i], 
			'\nproduct_name:', produto['product_name'].iloc[0], '\nvendas:',  produtos_vendidos.iloc[i],'\n')




##Item b

primeiro_carrinho = data_order.loc[lambda data: data['add_to_cart_order'] == 1]

primeiro_vendidos = primeiro_carrinho['product_id'].value_counts()

#print(primeiro_vendidos)

print('\n\nItem b - Produtos mais vendidos dos que sao primeiro introduzidos no carrinho')

for i in range (10):
	produto = data_products.loc[data_products['product_id'] == primeiro_vendidos.keys()[i]]
	#produto = dict_products[primeiro_vendidos.keys()[i]]
	print(i + 1, 'º\nproduct_id:', primeiro_vendidos.keys()[i], 
			'\nproduct_name:', produto['product_name'].iloc[0], '\nvendas:',  primeiro_vendidos.iloc[i],'\n')




##Item c

print('\n\nItem c - Media de produtos por compra')
compras = data_order['order_id'].unique()
print('\nNumero de compras:', len(compras), '\nTotal de produtos vendidos:', len(data_order['product_id']))
print('Media de produtos por compra:', round((len(data_order['product_id'])/len(compras)), 2),'\n\n')
