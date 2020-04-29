#%%
class Cliente:

    def __init__(self, nome):
        self.__nome = nome

    @property
    def nome(self):
        print('chamando @propety nome()')
        return self.__nome.title()

    @nome.setter
    def nome(self, nome):
        print('chamando @propety nome()')
        self.__nome = nome

# %%
cliente = Cliente('gabs')

# %%
