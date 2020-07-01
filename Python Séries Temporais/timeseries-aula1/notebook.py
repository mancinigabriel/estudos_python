# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Aula 1
# %% [markdown]
# ## Vídeo 1.2

# %%
import pandas as pd


# %%
carbonico = pd.read_csv('co2.csv',sep='\t')


# %%
carbonico.head()


# %%
import matplotlib.pyplot as plt


# %%
plt.plot(carbonico['data'],carbonico['media'])
plt.ylabel('CO2')
plt.xlabel('Data')


# %%
nasc = pd.read_csv('nascimentos.csv')


# %%
nasc.head()


# %%
plt.plot(nasc['data'],nasc['n_nasc'])
plt.ylabel("Número de nascimentos")
plt.xlabel("Data")


# %%
nasc["data"] = pd.to_datetime(nasc["data"])

# %% [markdown]
# ## Data Convertida

# %%
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# %%
plt.figure(figsize = (8,4));
plt.plot(nasc["data"],nasc['n_nasc'])
plt.ylabel("Número de nascimentos")
plt.xlabel("Data")

# %% [markdown]
# ## Vídeo 1.3

# %%
import numpy as np


# %%
media_carb = np.mean(carbonico["media"])


# %%
dt_carb = carbonico["data"][1]-carbonico["data"][0]


# %%
plt.plot(carbonico["data"],carbonico['media'])
plt.plot(carbonico.iloc[-1,0]+dt_carb,media_carb, '*')
plt.ylabel("CO2")
plt.xlabel("Data")


# %%
media_nasc = np.mean(nasc["n_nasc"])


# %%
dt_nasc = nasc["data"][1]-nasc["data"][0]


# %%
plt.plot(nasc["data"],nasc["n_nasc"])
plt.plot(nasc.iloc[-1,0]+dt_nasc,media_nasc, '*')
plt.ylabel("Número de nascimentos")
plt.xlabel("Data")


# %%
carbonico.shape


# %%
media_tres_pontos_c = np.mean(carbonico["media"][len(carbonico["media"])-3:])


# %%
plt.plot(carbonico["data"],carbonico['media'])
plt.plot(carbonico.iloc[-1,0]+dt_carb,media_tres_pontos_c, '*')
plt.ylabel("CO2")
plt.xlabel("Data")


# %%
nasc.shape


# %%
media_tres_pontos_n = np.mean(nasc["n_nasc"][len(nasc["n_nasc"])-3:])


# %%
plt.plot(nasc["data"],nasc["n_nasc"])
plt.plot(nasc.iloc[-1,0]+dt_nasc,media_tres_pontos_n, '*')
plt.ylabel("Número de nascimentos")
plt.xlabel("Data")


# %%
media_movel_c = carbonico.rolling(5).mean()


# %%
plt.plot(carbonico['data'],carbonico['media'])
plt.plot(media_movel_c['data'],media_movel_c['media'])


# %%
nasc


# %%
nasc['media'] = nasc['n_nasc'].rolling(5).mean()


# %%
plt.plot(nasc['data'],nasc['n_nasc'])
plt.plot(nasc['data'],nasc['media'])


# %%
from statsmodels.tsa.seasonal import seasonal_decompose


# %%
result_c = seasonal_decompose(carbonico.set_index('data'), freq=35)


# %%
result_c.plot()


# %%
nasc_plot = nasc.drop(['media'],1)


# %%
result_n = seasonal_decompose(nasc_plot.set_index('data'), freq=35)


# %%
result_n.plot();

# %% [markdown]
# ## Verificar estacionariedade da série

# %%
carbonico['media'].hist()


# %%
divide = int(len(carbonico)/2)


# %%
x = carbonico['media'].values


# %%
c1, c2 = x[0:divide],x[divide:]


# %%
print("{} - {}".format(c1.mean().round(2),c2.mean().round(2)))


# %%
print("{} - {}".format(c1.var().round(2),c2.var().round(2)))


# %%
divide = int(len(nasc)/2)


# %%
x = nasc['n_nasc'].values


# %%
n1, n2 = x[0:divide], x[divide:]


# %%
print("{} - {}".format(n1.mean().round(2),n2.mean().round(2)))


# %%
print("{} - {}".format(n1.var().round(2),n2.var().round(2)))


# %%
from statsmodels.tsa.stattools import adfuller


# %%
resultado_c = adfuller(carbonico['media'].values)


# %%
print('Estatística ADF: {}'.format(resultado_c[0].round(2)))
print('p-valor: {}'.format(resultado_c[1]))


# %%
resultado_n = adfuller(nasc['n_nasc'].values)


# %%
print('Estatística ADF: {}'.format(resultado_n[0].round(2)))
print('p-valor: {}'.format(resultado_n[1]))

# %% [markdown]
# ## Diferenciação

# %%
serie_diferenciada = nasc['n_nasc'].diff()


# %%
plt.plot(nasc['data'],nasc['n_nasc'])
plt.ylabel('Nascimentos')
plt.xlabel('Data')


# %%
plt.plot(nasc ['data'],serie_diferenciada)
plt.ylabel('Nascimentos')
plt.xlabel('Data')

# %% [markdown]
# ## Suavização exponencial

# %%
carbonico = carbonico.set_index('data')


# %%
carbonico_treino = carbonico[1980.042:2015]


# %%
carbonico_teste = carbonico[2015:]


# %%
from statsmodels.tsa.holtwinters import SimpleExpSmoothing


# %%
modelo_ajustado = SimpleExpSmoothing(carbonico_treino).fit(smoothing_level=0.5)


# %%
carbonico_teste.shape


# %%
modelo_previsto = modelo_ajustado.forecast(57)


# %%
plt.plot(carbonico_treino)
plt.plot(carbonico_treino.index, modelo_ajustado.fittedvalues.values)
plt.plot(carbonico_teste,'g')
plt.plot(carbonico_teste.index, modelo_previsto, 'r.')


# %%
nasc_pred = nasc_plot.set_index('data')


# %%
nasc_treino = nasc_pred['1959-01-01':'1959-12-01'].astype('double')


# %%
nasc_teste = nasc_pred['1959-12-01':]


# %%
modelo_ajustado = SimpleExpSmoothing(nasc_treino).fit(smoothing_level=0.5)


# %%
nasc_teste.shape[0]


# %%
modelo_previsto = modelo_ajustado.forecast(31)


# %%
plt.plot(nasc_treino)
plt.plot(nasc_treino.index, modelo_ajustado.fittedvalues.values)
plt.plot(nasc_teste,'g')
plt.plot(nasc_teste.index, modelo_previsto, 'r.')

# %% [markdown]
# ## Método de Holt-Winters
# %% [markdown]
# Considera nível, tendência, sazonalidade e ruído do modelo, pode ser aditivo ou multiplicativo.

# %%
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# %%
modelo_ajustado = ExponentialSmoothing(carbonico.values, 
trend = 'additive', seasonal='additive', seasonal_periods = 35).fit()


# %%
carbonico.shape


# %%
carbonico_teste.shape


# %%
len(carbonico) - len(carbonico_teste)


# %%
modelo_previsto = modelo_ajustado.predict(start = len(carbonico) - len(carbonico_teste),end=len(carbonico)-1)


# %%
plt.figure(figsize=(8,4))
plt.plot(carbonico_treino)
plt.plot(carbonico_teste,'g')
plt.plot(carbonico_teste.index, modelo_previsto, 'r.')


# %%
modelo_ajustado = ExponentialSmoothing(nasc_treino.values,trend = 'multiplicative', seasonal=None).fit()


# %%
modelo_previsto_suave = modelo_ajustado.predict(start = 335,end = 365) 


# %%
plt.plot(nasc_treino)
plt.plot(nasc_teste,'g')
plt.plot(nasc_teste.index, modelo_previsto_suave,'r.')

# %% [markdown]
# ## Autocorrelação

# %%
from statsmodels.graphics.tsaplots import plot_acf


# %%
plot_acf(carbonico);


# %%
from statsmodels.graphics.tsaplots import plot_pacf


# %%
plot_pacf(carbonico, lags = 20);


# %%
plot_acf(nasc['n_nasc'], lags = 20);


# %%
plot_pacf(nasc['n_nasc'], lags = 20);


# %%
from statsmodels.tsa.ar_model import AutoReg


# %%
lista = np.linspace(1,40,40)


# %%
modelo_ajustado = AR(carbonico_treino.values, lags=lista,trend='c',seasonal=True,period = 35).fit()


# %%


