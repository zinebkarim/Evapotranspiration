#!/usr/bin/env python
# coding: utf-8

# # 1. Prétraitement des données 

# ## 1.1 Importation des bibliothèques nécessaires : 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# ## 1.2 Importation des données sous forme de Dataframe

# In[2]:


df = pd.read_csv('dataET0.csv',delimiter=';',decimal=',')
df


# ## 1.3 Afficher le nombre de valeurs nulles

# In[3]:


print(df.isnull().sum())


# ## 1.4 Remplacement de valeurs nulles par la moyenne

# In[4]:


df = df.fillna(df.mean())
print(df.isnull().sum())


# ## 1.5 Les Type de données ( remarque : pour transformer les types de données on a paramétré  read_csv avec virgule  avec  decimal=',')

# In[5]:


print(df.dtypes)


# ## 1.6 Diviser les données en  données  test/training
# 

# In[6]:


train_df, test_df = train_test_split(df, test_size=0.2)


# In[7]:


test_df


# # 2.  Construction des modèles d'apprentissage : 

# ## - Définition de la variable dépendante et du vecteurs des variables indépendantes : 

# In[8]:



y=df['ETP quotidien [mm]']

X=df.iloc[0:, 1:18]
print(X.dtypes)
print(y.dtypes)


# # - Estimation du modèle de  Random forest  et différentes métriques : 

# In[9]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

S=[]
# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create the random forest model
rf = RandomForestRegressor()

# Fit the model to the training data
rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf.predict(X_test)

# Calculate the MAE, MSE, and R2 score
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)



plt.hist(y_test - y_pred, bins=50)
plt.xlabel("Erreur")
plt.ylabel("Fréquence")
plt.title("Histogramme des erreurs (MAE = {:.2f}, MSE = {:.2f})".format(mae, mse))
plt.show()
S.append(r2)


# ## - Estimation du modèle de  l'arbre de décision   et différentes métriques : 

# In[10]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


# Séparer les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialiser l'arbre de décision
tree = DecisionTreeRegressor()

# Entraîner l'arbre de décision sur les données d'entraînement
tree.fit(X_train, y_train)

# Prédire les valeurs cibles pour les données de test
y_pred = tree.predict(X_test)

# Calculer l'erreur absolue moyenne et l'erreur quadratique moyenne
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MAE:", mae)
print("MSE:", mse)
print("R-squared: ", r2)



# Tracer les erreurs
plt.hist(y_test - y_pred, bins=50)
plt.xlabel("Erreur")
plt.ylabel("Fréquence")
plt.title("Histogramme des erreurs (MAE = {:.2f}, MSE = {:.2f})".format(mae, mse))
plt.show()

S.append(r2)


# ## - Estimation du modèle de  SVR   et différentes métriques : 

# In[11]:


from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialiser et entraîner le modèle SVR
svr = SVR()
svr.fit(X_train, y_train)

# Prédire sur l'ensemble de test
y_pred = svr.predict(X_test)

# Calculer les métriques
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Afficher les résultats
print("Mean Squared Error: ", mse)
print("Mean Absolute Error: ", mae)
print("R-squared: ", r2)



# Tracer les erreurs
plt.hist(y_test - y_pred, bins=50)
plt.xlabel("Erreur")
plt.ylabel("Fréquence")
plt.title("Histogramme des erreurs (MAE = {:.2f}, MSE = {:.2f})".format(mae, mse))
plt.show()
S.append(r2)


# ## - Estimation du modèle de régression non linéaire /polynomiale    et différentes métriques : 

# In[12]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Ajouter des termes polynomiaux aux données d'entraînement
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Initialiser et entraîner le modèle de régression linéaire
regressor = LinearRegression()
regressor.fit(X_train_poly, y_train)

# Prédire sur l'ensemble de test
y_pred = regressor.predict(X_test_poly)

# Calculer les métriques
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Afficher les résultats
print("Mean Squared Error: ", mse)
print("Mean Absolute Error: ", mae)
print("R-squared: ", r2)


# Tracer les erreurs
plt.hist(y_test - y_pred, bins=50)
plt.xlabel("Erreur")
plt.ylabel("Fréquence")
plt.title("Histogramme des erreurs (MAE = {:.2f}, MSE = {:.2f})".format(mae, mse))
plt.show()

S.append(r2)


# # Comparaison entre les scores des 4 modèles et en déduire le plus précis

# In[13]:


mod=['rf','tree','svr','regressor']


# In[14]:


plt.bar(mod,S,label="r2")
plt.show()


# In[ ]:





# # Questions lors de la présentation

# In[15]:


X_train


# In[16]:


#Afficher la taille des données de l'entrainement 


# In[17]:


X_train.shape


# In[18]:


#Afficher la colonne de la target


# In[19]:


df["ETP quotidien [mm]"]


# In[ ]:





# In[20]:


#Moyenne du quotient d'évapotranspiration


# In[21]:


df["ETP quotidien [mm]"].mean()


# In[ ]:





# In[22]:


#Prédiction du quotient d'évapotranspiration


# In[23]:


X=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
X1=np.array(X).reshape(-1,17)
Y1=rf.predict(X1)
print(Y1)


# In[ ]:





# In[ ]:




