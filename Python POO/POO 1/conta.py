#%%
class Conta:

    def __init__(self, numero, titular, saldo, limite = 1000):
        print('Construindo objeto...{}'.format(self))
        self.__numero = numero
        self.__titular = titular
        self.__saldo = saldo
        self.__limite = limite

    def extrato(self):
        print('Saldo {} do titular {}'.format(self.__saldo,self.__titular))

    def deposita(self, valor):
        self.__saldo += valor

    def __pode_sacar(self, valor_a_sacar):
        valor_disponivel_a_sacar = (self.__saldo + self.__limite)
        return valor_a_sacar <= valor_disponivel_a_sacar

    def saca(self, valor):
        if self.__pode_sacar(valor):
            self.__saldo -= valor
        else:
            print("O valor {} passou o limite".format(valor))

    def transfere(self, valor, destino):
        self.saca(valor)
        destino.deposita(valor)

    @staticmethod
    def codigos_bancos():
        return {'BB': '001', 'Caixa': '104', 'Bradesco': '237'}

    @property   
    def saldo(self):
        return self.__saldo

    @property
    def titular(self):
        return self.__titular

    @property
    def limite(self):
        return self.__limite

    @limite.setter 
    def limite(self, limite):
        self.__limite = limite

#%%
class Data:

    def __init__(self, dia, mes, ano):
        print('Construindo objeto...{}'.format(self))
        self.dia = dia
        self.mes = mes
        self.ano = ano

    def formatada(self):
        print('{}/{}/{}'.format(self.dia, self.mes, self.ano))

# %%
conta = Conta(123, "Nico", 55.5, 1000.0)
conta2 = Conta(321, "Marco", 100.0, 1000.0)


# %%
