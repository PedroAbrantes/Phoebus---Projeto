def Bubblesort(array):
	ordenado = False
	while not ordenado:
		ordenado = True

		for i in range(len(array) - 1):
			if array[i] > array[i + 1]:
				array[i], array[i + 1] = array[i + 1], array[i]
				ordenado = False        
				print(array)

	return array


array = [5, 9, 54, 12, 7, 5, 6, 1]

print('\n\narray antes da ordenacao:', array, '\n')

array_ordenado = Bubblesort(array)

print('\narray apos ordenacao:', array_ordenado, '\n\n')