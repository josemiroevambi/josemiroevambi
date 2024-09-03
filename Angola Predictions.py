# -*- coding: utf-8 -*-
"""
@author: Josemiro Evambi


Nessa ferramenta vamos aplicar a Regressão linear para prever os valores de 
IDH de Angola em relação ao PIB per capita

"""
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib_inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error



""" 
1º importar o arquivo excel de onde iremos extrair os dados de análise e treinamento 

"""

Angola_data = pd.read_excel("Angola Data.xlsx")

#Angola_data.info()


"""
2º visualizar o comportamento de todas as variáveis e ver quais têm uma relação
de linearidade
"""

#sns.pairplot(Angola_data, kind='scatter', plot_kws={'alpha': 0.4})


"""
3ª definir e preparar as variáveis(dados) de treinamento
"""

#sns.lmplot(x ='GDP per Capita (USD per Capita)', 
#           y = 'HDI', data=Angola_data, 
#           scatter_kws={'alpha':0.4})



X = Angola_data[['GDP per Capita (USD per Capita)',
                 'Government Expenditure (% of GDP)', 
                 'Inflation rate (Annual %)']]

y = Angola_data['HDI']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



"""
4º aplicar a regressão linear para treinar o modelo
"""

model = LinearRegression()

model.fit(X_train, y_train)

coef = pd.DataFrame(model.coef_, X.columns, columns=['Coeficientes'])

print(coef)


"""5º prever os resultados do modelo com os dados de teste"""

pred = model.predict(X_test)
#print(pred)

sns.scatterplot(pred)

#plt.title("Avaliação do IDH vs GDP per Capita")
#plt.xlabel("Previsões")

#print("Mean Absolute Error: ", mean_absolute_error(y_test, pred))
#print("Mean Squared Error: ", mean_squared_error(y_test, pred))
#print("RMSE: ", math.sqrt(mean_squared_error(y_test, pred)))

"""6º analisar os residuais"""

residuais = y_test - pred

sns.displot(residuais, bins=30, kde=True)

import pylab
import scipy.stats as stats

stats.probplot(residuais, dist="norm", plot=pylab)
pylab.show()








