class No:
    def __init__(self, valor):
        self.__esquerdo = None
        self.__direito = None
        self.__valor = valor

    def getFilhoEsquerdo(self):
        return self.__esquerdo

    def getFilhoDireito(self):
        return self.__direito

    def getValor(self):
        return self.__valor

    def setFilhoEsquerdo(self, no_filho):
        self.__esquerdo = no_filho

    def setFilhoDireito(self, no_filho):
        self.__direito = no_filho


class BinaryTree:
    def __init__(self):
        self.__raiz = None

    def addNo(self, no, valor):
        if(no == None):
            self.__raiz = No(valor)
        else:
            if(valor < no.getValor()):
                if(no.getFilhoEsquerdo() == None):
                    no.setFilhoEsquerdo(No(valor))
                else:
                    self.addNo(no.getFilhoEsquerdo(), valor)
            else:
                if(no.getFilhoDireito() == None):
                    no.setFilhoDireito(No(valor))
                else:
                    self.addNo(no.getFilhoDireito(), valor)

    def getRaiz(self):
        return self.__raiz

    def printOrdenado(self, no):
        if(no != None):
            self.printOrdenado(no.getFilhoEsquerdo())
            print(no.getValor())
            self.printOrdenado(no.getFilhoDireito())

    def printNo(self, no):
        if(no != None):
            self.printNo(no.getFilhoEsquerdo())
            print('No:', no.getValor())
            if no.getFilhoEsquerdo() != None:
                print('Filho esquerdo:', no.getFilhoEsquerdo().getValor())
            else:
                print('Filho esquerdo:', None)

            if no.getFilhoDireito() != None:
                print('Filho direito:', no.getFilhoDireito().getValor(), '\n')
            else:
                print('Filho direito:', None)
            self.printNo(no.getFilhoDireito())


#Rodando os testes

if __name__ == "__main__":
    array = [23, 11, 7, 21, 59, 83, 38]

    arvore = BinaryTree()

    for elemento in array:
        arvore.addNo(arvore.getRaiz(), elemento)


    print('\n\nArvore ordenada:')
    arvore.printOrdenado(arvore.getRaiz())
    print('\n\n')
    arvore.printNo(arvore.getRaiz())
    print('\n\n')


