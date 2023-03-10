#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Gerando Modelo de Machine Learning (Classificação)
# Preparação dos Dados e Treinamento do Modelo de Machine Learning - Parte 1.
# Autor: Jeshua Ben da Costa Ferreira


# In[9]:


#Conjunto de importacoes
import numpy as np #Biblioteca de manipulação de valores, vetor e matriz multidimensional
import pandas as pd # Biblioteca de manipulação de dados == Excel
import matplotlib.pyplot as plt # Biblioteca de visualização de dados gráficos (Gráfico pizza, barras)
from sklearn.naive_bayes import MultinomialNB 
# Sklearn -> principal biblioteca para machine learning com python. Ela contém todos os modelos de algoritmos.
from sklearn.naive_bayes import GaussianNB #Biblioteca de ML com todos os algoritmos
from sklearn.metrics import accuracy_score # Lib que mostra accuracy e precision
from sklearn.model_selection import train_test_split # Separa as bases para treino e teste


# In[13]:


#Carrega o dataset com os dados para o treinamento e validacao
dados_mamiferos_treino = pd.read_csv("dados_mamiferos_treino.csv", encoding="utf-8")


# In[17]:


# Visualizando os dados. Mostra as 5 primeiras linhas do dataframe.
dados_mamiferos_treino.head()


# In[15]:


# Visualizar as 5 últimas linhas
dados_mamiferos_treino.tail()


# In[18]:


#Definição dos atributos que deverao ser treinados para gerar o modelo de classificacao
data_treino = np.array(dados_mamiferos_treino[['sangue', 'bota_ovo', 'voa', 'mora_agua']])

#Definição do atributo de classificacao
data_classif = np.array(dados_mamiferos_treino['classificacao'])


# In[21]:


# Verificando o formato dos dados -> (linhas, colunas)
dados_mamiferos_treino.shape


# In[22]:


# Identificando a correlação entre as variáveis
# Correlação não implica causalidade
def plot_corr(dados_mamiferos_treino, size=10):
    corr = dados_mamiferos_treino.corr()    
    fig, ax = plt.subplots(figsize = (size, size))
    ax.matshow(corr)  
    plt.xticks(range(len(corr.columns)), corr.columns) 
    plt.yticks(range(len(corr.columns)), corr.columns) 


# In[23]:


# Criando o gráfico --> Matriz de correlação
plot_corr(dados_mamiferos_treino)


# In[24]:


# Splitting 70% treino 30% teste


# In[25]:


#Definição dos conjuntos de treinamento e validação
x_treino, x_val, y_treino, y_val = train_test_split(data_treino, data_classif, test_size=0.30)


# In[26]:


#Apresentacao dos dados selecionados para o conjunto de treinamento e validacao
print("-----------------------")
print("Conjunto de Treinamento")
print(x_treino)
print("Conjunto de Validacao")
print(x_val)
print("-----------------------")


# In[27]:


# Imprimindo os resultados
print("{0:0.2f}% nos dados de treino".format((len(x_treino)/len(dados_mamiferos_treino.index)) * 100))
print("{0:0.2f}% nos dados de teste".format((len(x_val)/len(dados_mamiferos_treino.index)) * 100))


# In[28]:


# - Aplicando o Algoritmo com o Naive Bayes - GaussianNB.


# In[29]:


#Treinamento do modelo com os dados atribuidos ao conjunto de treinamento
modelo_NB = GaussianNB()
modelo_NB.fit(x_treino, y_treino)


# In[30]:


#Predição e acurácia para o conjunto de treinamento
print("Predicao para o conjunto de treinamento")
y_pred_treino = modelo_NB.predict(x_treino)
print("Acuracia para o conjunto de treinamento")
print(accuracy_score(y_treino, y_pred_treino))
print("Na Base de Treinamento")


# In[31]:


#Predição e acurácia para o conjunto de validação
print("Predicao para o conjunto de validacao")
y_pred_val = modelo_NB.predict(x_val)
print("Acuracia para o conjunto de validacao")
print(accuracy_score(y_val, y_pred_val))
print("na Base de Teste e ou Validação")


# In[32]:


from sklearn import metrics


# In[33]:


# Criando uma Confusion Matrix
print("Confusion Matrix")

print("{0}".format(metrics.confusion_matrix(y_val, y_pred_val, labels = [1, 0])))
print("")

print("Classification Report")
print(metrics.classification_report(y_val, y_pred_val, labels = [1, 0]))


# In[ ]:




