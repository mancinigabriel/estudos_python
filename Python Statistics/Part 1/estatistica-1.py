#%%
import pandas as pd
import numpy as np
import seaborn as sns

# %%
dataset = pd.read_csv('dados.csv')

# %% Variáveis Qualitativas Ordinais do Dataset
sorted(dataset['Anos de Estudo'].unique())

# %% Variáveis Qualitativas Nominais do Dataset
sorted(dataset['UF'].unique())
sorted(dataset['Sexo'].unique())
sorted(dataset['Cor'].unique())

# %%
dataset.Idade.min()

# %% Distribuição de frequência para variáveis qualitativas
frequencia = dataset['Sexo'].value_counts()

# %%
percentual = dataset['Sexo'].value_counts(normalize = True)

# %%
dist_freq_qualitativas = pd.DataFrame({'Frequência': frequencia, 'Percentual (%)': percentual* 100})

# %%
dist_freq_qualitativas.rename(index = {0: 'Masculino', 1: 'Feminino'}, inplace = True)
dist_freq_qualitativas.rename_axis('Sexo', axis = 'columns', inplace = True)

# %%
dist_freq_qualitativas

# %% Cross tab
sexo = {0: 'Masculino', 
        1: 'Feminino'}

cor = {0: 'Indígena',
       2: 'Branca',
       4: 'Preta',
       6: 'Amarelo',
       8: 'Parda',
       9: 'Sem declaração'}

# %% tabela de frequencia
frequencia = pd.crosstab(dataset.Sexo, dataset.Cor)

frequencia.rename(index = sexo, inplace = True)
frequencia.rename(columns = cor, inplace = True)
frequencia

# %%
percentual = pd.crosstab(dataset.Sexo, dataset.Cor, normalize = True) * 100

percentual.rename(index = sexo, inplace = True)
percentual.rename(columns = cor, inplace = True)
percentual

# %% Tabela de frequencia com agregação
#Essa tabela diz a renda média por tipo de agrupamento 
#Homens indígenas ganham em média 1080 por ex
agg_test = pd.crosstab(dataset.Sexo, dataset.Cor, aggfunc = 'mean', values = dataset.Renda)

agg_test.rename(index = sexo, inplace = True)
agg_test.rename(columns = cor, inplace = True)
agg_test

# %% Distribuições de Frequência para Variáveis Quantitativas (Classes Personalizadas)
# Classificar as rendas em classes A,B,C,D ou E de acordo com a quantidade de salários
# mínimos, valores de 2015
# A (> 15.760)
# B (> 7.880)
# C (> 3.152)
# D (> 1.576)
# E (< 1.576)
dataset.Renda.min()

# %%
dataset.Renda.max()

# %%
classes = [0, 1576, 3152, 7880, 15760, 200000]

# %%
labels = ['E','D','C','B','A']

# %% pandas.cut
frequencia_classes = pd.value_counts(
    pd.cut(x = dataset.Renda,
           bins = classes, 
           labels = labels, 
           include_lowest = True)
) 

percentual_classes = pd.value_counts(
    pd.cut(x = dataset.Renda,
           bins = classes, 
           labels = labels, 
           include_lowest = True),
           normalize = True
) * 100

# %%
dist_freq_quantitativas_personalizadas = pd.DataFrame(
    {'Frequência': frequencia_classes, 'Percentual (%)': percentual_classes})

# %%
dist_freq_quantitativas_personalizadas.sort_index(ascending = False)

# %% Distribuição de frequências para variáveis
# quantitativas (classes de amplitude fixa)
n = dataset.shape[0] 

# %% Regra de Sturges é uma fórmula para definir o número de classes
# se baseando no total de observações de uma variável
k = 1 + (10/3) * np.log10(n)
k = int(k.round(0))

# %%
frequencia = pd.value_counts(
    pd.cut(x = dataset.Renda,
           bins = k, 
           include_lowest = True),
           sort = False
) 

# %%
percentual = pd.value_counts(
    pd.cut(x = dataset.Renda,
           bins = k, 
           include_lowest = True),
           normalize = True,
           sort = False
) * 100

# %%
dist_freq_quantitativas_amplitude_fixa = pd.DataFrame(
    {'Frequência': frequencia, 'Percentual (%)': percentual})
dist_freq_quantitativas_amplitude_fixa.rename_axis(
    'Faixas de Renda', axis = 'columns', inplace = True)
dist_freq_quantitativas_amplitude_fixa

# %% Histogramas
ax = sns.distplot(dataset.Altura, kde = False)

ax.figure.set_size_inches(12,6)
ax.set_title('Distribuição de Frequências - Altura', fontsize = 18)
ax.set_xlabel('Metros', fontsize = 14)

# %%
ax = sns.distplot(dataset.Altura)

ax.figure.set_size_inches(12,6)
ax.set_title('Distribuição de Frequências - Altura', fontsize = 18)
ax.set_xlabel('Metros', fontsize = 14)

# %% Histograma com Pandas
dataset.Altura.hist(bins = 50, figsize = (12,6))

# %%
dist_freq_quantitativas_personalizadas['Frequência'].plot.bar(
    width = 1, alpha = 0.6, figsize = (12,6))

# %%
