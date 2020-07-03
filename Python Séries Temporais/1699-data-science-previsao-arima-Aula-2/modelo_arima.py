# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Aula 1
# %% [markdown]
# ## Modelos AR

# %%
import pandas as pd


# %%
dado_ar1 = pd.read_csv('produto_ar1.csv')


# %%
dado_ar1.head()


# %%
import matplotlib.pyplot as plt


# %%
plt.figure(figsize=(12,8))
plt.plot(dado_ar1['Data'], dado_ar1['Preco'])
plt.xlabel('Data')
plt.ylabel('Preco')


# %%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# %%
plot_acf(dado_ar1['Preco']);


# %%
plot_pacf(dado_ar1['Preco']);


# %%
dado_ar2 = pd.read_csv('produto_ar2.csv')


# %%
plt.figure(figsize=(12,8))
plt.plot(dado_ar2['Data'], dado_ar2['Preco'])
plt.xlabel('Data')
plt.ylabel('Preco')


# %%
plot_acf(dado_ar2['Preco']);


# %%
plot_pacf(dado_ar2['Preco']);


# %%
dado_ar3 = pd.read_csv('produto_ar3.csv')


# %%
plt.figure(figsize=(12,8))
plt.plot(dado_ar3['Data'], dado_ar3['Preco'])
plt.xlabel('Data')
plt.ylabel('Preco')


# %%
plot_acf(dado_ar3['Preco']);


# %%
plot_pacf(dado_ar3['Preco']);

# %% [markdown]
# ## Ajuste dos modelos AR

# %%
dado_ar1_p1 = dado_ar1[0:500]
dado_ar1_p2 = dado_ar1[500:600]


# %%
import statsmodels.tsa.api as smtsa 


# %%
modelo_ar1 = smtsa.ARMA(dado_ar1_p1['Preco'], order=(1, 0)).fit()


# %%
modelo_previsto1 = modelo_ar1.predict(start=500,end=599)


# %%
plt.figure(figsize=(12,8))
plt.plot(dado_ar1_p1['Data'],dado_ar1_p1['Preco'])
plt.plot(dado_ar1_p2['Data'],dado_ar1_p2['Preco'])
plt.plot(dado_ar1_p2['Data'], modelo_previsto1,'r.')


# %%
dado_ar3_p1 = dado_ar3[0:500]
dado_ar3_p2 = dado_ar3[500:600]


# %%
modelo_ar3 = smtsa.ARMA(dado_ar3_p1['Preco'], order=(3, 0)).fit()


# %%
modelo_previsto3 = modelo_ar3.predict(start=500,end=599)


# %%
plt.figure(figsize=(12,8))
plt.plot(dado_ar3_p2['Data'],dado_ar3_p2['Preco'])
plt.plot(dado_ar3_p2['Data'], modelo_previsto3,'r.')

# %% [markdown]
# # Aula 2
# %% [markdown]
# ## Modelos MA

# %%
dado_ma1 = pd.read_csv('produto_ma1.csv')


# %%
plt.figure(figsize=(12,8))
plt.plot(dado_ma1['Data'],dado_ma1['Preco'])


# %%
plot_acf(dado_ma1['Preco']);


# %%
plot_pacf(dado_ma1['Preco']);


# %%
dado_ma1_p1 = dado_ma1[0:500]
dado_ma1_p2 = dado_ma1[500:600]


# %%
modelo_ma1 = smtsa.ARMA(dado_ma1_p1['Preco'], order=(0, 1)).fit()


# %%
modelo_previsto_ma1 = modelo_ma1.predict(start=500,end=599)


# %%
plt.figure(figsize=(12,8))
plt.plot(dado_ma1_p2['Data'],dado_ma1_p2['Preco'])
plt.plot(dado_ma1_p2['Data'], modelo_previsto_ma1,'.r')


# %%
dado_ma3 = pd.read_csv('produto_ma3.csv')


# %%
plt.figure(figsize=(12,8))
plt.plot(dado_ma3['Data'],dado_ma3['Preco'])


# %%
plot_acf(dado_ma3['Preco']);


# %%
dado_ma3_p1 = dado_ma3[0:500]
dado_ma3_p2 = dado_ma3[500:600]


# %%
plot_pacf(dado_ma3['Preco']);


# %%
modelo_ma3 = smtsa.ARMA(dado_ma3_p1['Preco'], order=(0, 3)).fit()


# %%
modelo_previsto_ma3 = modelo_ma3.predict(start=500,end=599)


# %%
plt.figure(figsize=(12,8))
plt.plot(dado_ma3_p2['Data'],dado_ma3_p2['Preco'])
plt.plot(dado_ma3_p2['Data'], modelo_previsto_ma3,'r.')

# %% [markdown]
# ## Modelo ARMA

# %%
dado_arma1 = pd.read_csv('produto_arma1_1.csv')


# %%
plt.figure(figsize=(12,8))
plt.plot(dado_arma1['Data'],dado_arma1['Preco'])


# %%
plot_acf(dado_arma1['Preco']);


# %%
plot_pacf(dado_arma1['Preco']);


# %%
dado_arma1_p1 = dado_arma1[0:500]
dado_arma1_p2 = dado_arma1[500:600]


# %%
modelo_arma1 = smtsa.ARMA(dado_arma1_p1['Preco'], order=(1, 1)).fit()


# %%
modelo_previsto_arma1 = modelo_arma1.predict(start=500,end=599)


# %%
plt.figure(figsize=(12,8))
plt.plot(dado_arma1_p2['Data'],dado_arma1_p2['Preco'])
plt.plot(dado_arma1_p2['Data'], modelo_previsto_arma1,'r')

# %% [markdown]
# ## Previsão e estacionariedade

# %%
cresc = pd.read_csv('produto_crescente.csv')


# %%
cresc.head()


# %%
plt.figure(figsize=(12,5))
plt.plot(cresc['Data'],cresc['Preco'])
plt.xticks(rotation=70);


# %%
plot_acf(cresc['Preco']);


# %%
plot_pacf(cresc['Preco']);


# %%
cresc.shape


# %%
cresc_p1 = cresc[:26][:]
cresc_p2 = cresc[26:][:]


# %%
modelo_cresc = smtsa.ARMA(cresc_p1['Preco'].values, order=(1, 2)).fit();


# %%
modelo_previsto_cresc = modelo_cresc.predict(start=26,end=35)


# %%
plt.figure(figsize=(12,5))
plt.plot(cresc_p1['Data'],cresc_p1['Preco'])
plt.plot(cresc_p1['Data'],modelo_cresc.fittedvalues,'.')
plt.plot(cresc_p2['Data'],cresc_p2['Preco'])
plt.plot(cresc_p2['Data'],modelo_previsto_cresc,'r.')
plt.xticks(rotation=70);


# %%
cres_d = cresc['Preco'].diff(1)


# %%
plt.figure(figsize=(10,4))
plt.plot(cresc['Data'],cres_d)
plt.xticks(rotation=70);


# %%



# %%
modelo_cresc = smtsa.ARIMA(cresc_p1['Preco'].values, order=(1,1,2)).fit()


# %%
import numpy as np


# %%
modelo_cresc.plot_predict(26,35);
plt.plot(np.linspace(0,9,10),cresc_p2['Preco'])


# %%
from pmdarima import auto_arima


# %%
modelo_busca = auto_arima(cresc_p1['Preco'].values,
    start_p=0, start_q=0, max_p=6, max_q=6,
    d=1, seasonal=False, trace=True,
    error_action='ignore', suppress_warnings=True,
    stepwise = False)


# %%
modelo_busca.aic()


# %%
modelo_busca.bic()


# %%
modelo_busca.fit(cresc_p1['Preco'].values)


# %%
valores_preditos = modelo_busca.predict(n_periods=10)


# %%
plt.plot(cresc_p2['Data'],valores_preditos)
plt.plot(cresc_p2['Data'],cresc_p2['Preco'])


# %%


