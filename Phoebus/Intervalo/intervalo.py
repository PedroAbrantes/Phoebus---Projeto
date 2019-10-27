class Intervalo:
	def __init__(self, id : int, xmin : int, xmax : int):
		self.__id = id
		self.__xmin = xmin
		self.__xmax = xmax

	def getID(self):
		return self.__id

	def getxMin(self):
		return self.__xmin

	def getxMax(self):
		return self.__xmax

	def intersecao(self, intervalo):
		intervalo1 = list(range(self.__xmin, self.__xmax + 1))
		intervalo2 = list(range(intervalo.getxMin(), intervalo.getxMax() + 1))

		intersecao = [value for value in intervalo1 if value in intervalo2] 

		return intersecao

	def elo(self, intervalo):
		elo = None

		intersecao = self.intersecao(intervalo)
		#Existe interseção
		if intersecao:
			elo = len(intersecao) - 1

		#Não existe interseção
		else:
			if self.__xmax < intervalo.getxMin():
				elo = self.__xmax - intervalo.getxMin()
			else:
				elo = intervalo.getxMax() - self.__xmin

		print(elo, '\n')
		return elo


#Rodando os testes

if __name__ == "__main__":
	a = Intervalo(id = 1, xmin = 2, xmax = 3)
	b = Intervalo(id = 2, xmin = 3, xmax = 4)


	print('\n\n')
	elo1 = a.elo(b)
	elo2 = b.elo(Intervalo(id = 4, xmin = 8, xmax = 15))
	elo3 = a.elo(Intervalo(id = 8, xmin = 2, xmax = 4))
	print('\n')
