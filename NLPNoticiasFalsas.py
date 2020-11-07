# python 3.7.9

from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from tensorflow.keras import backend
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt 
import re 
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import seaborn as sns 
plt.style.use('ggplot')
from sklearn import datasets,linear_model
from sklearn.model_selection import train_test_split

from textblob import TextBlob
import string

from translate import Translator


app = Flask(__name__)

# El normalizado de datos o limpieza de datos
def normalize(data):
        normalized = []
        for i in data:
            i = i.lower()
            # eliminamos url
            i = re.sub('https?://\S+|www\.\S+', '', i)
            # eliminamos espacios extra
            i = re.sub('\\W', ' ', i)
            i = re.sub('\n', '', i)
            i = re.sub(' +', ' ', i)
            i = re.sub('^ ', '', i)
            i = re.sub(' $', '', i)
            normalized.append(i)
        return normalized
# antes de inicar llamamos al modelo
@app.before_first_request
def load_model_to_app():
   app.model = load_model('./static/model/model3.h5')   
# la ruta por defecto que funcionara
@app.route("/")
def index():
    return render_template('index.html', pred = 0)
# ruta donde se reailzara la prediccion
@app.route('/predict', methods=['POST'])
# prediccion
def predict():
    # recibe los datos del formulario
    data = [request.form['titulo'],
            request.form['noticia']]
    # recibimos ambas datos 
    a = data[0]
    b = data[1]
    # concatenamos el titulo con el contenido de la noticia
    b = a +b
    
    #Para tokenizar
    #Leyendo los datos
    fake_df = pd.read_csv('Fake.csv')
    real_df = pd.read_csv('True.csv')
    
    

    
    #Agregamos una nueva columna check con valores de True y False 
    real_df['check'] = 'TRUE'
    fake_df['check'] = 'FAKE'
    
    #Eliminamos columnas que creemos no necesarias(fecha,tema)
    fake_df.drop(['date', 'subject'], axis=1, inplace=True)
    real_df.drop(['date', 'subject'], axis=1, inplace=True)
    
    #Valores O noticias falsas,1 noticias verdaderas
    fake_df['class'] = 0 
    real_df['class'] = 1
    
  
    
    
    #Concatenamos ambos datasets en uno nuevo de noticias(news_df)
    news_df = pd.concat([fake_df, real_df], ignore_index=True, sort=False)
    #print(news_df)
    
    
    #juntamos dos columnas en una sola 
    news_df['text'] = news_df['title'] + news_df['text']
    news_df.drop('title', axis=1, inplace=True)
    
    
    
    #Dividimos entre train y test
    features = news_df['text']
    targets = news_df['class']
    
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.20, random_state=18)
    
    #Eliminamos espacios en blanco,url,etc.
    data = np.array([b])
    X_test = pd.Series(data)
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    
    """
    Aca se esta construyendo la posibilidad de que funcione para español
    usando el traductor... tomarlo como referencia.
    
    translate= Translator(from_lang="spanish",to_lang="english")
    b = translate.translate(b)
    print(b)
    print(TextBlob(a).translate(from_lang="es",to="en").sentiment)
      
    
    translate= Translator(from_lang="spanish",to_lang="english")
    b = translate.translate(b)
    print(b)
    
    a = normalize(a)
    print(TextBlob(a).translate(from_lang="es",to="en").sentiment)
    a = 'odio el codigo spaguetti'
    
    translator= Translator(from_lang="spanish",to_lang="english")
    translation = translator.translate(a)
    print (translation)
    
    print(TextBlob(a).translate(from_lang="es",to="en").sentiment)
    #print (type(X_test))
    """
    
  
    
    """
    === importante===
    # Nosotros llamamos de nuevo para usar el mismo valor que se les dio a las 
    # palabras a la hora de toketinzar, es por eso que llamamos la funcion de nuevo
    # Si es posible, guardar los datos de las palabras con su respectivo valor o
    # no llamar a dicha tokenizacion, se ahorra bastante tiempo a la hora de realizar
    # la prediccion, MEJORAR
    """
    max_vocab = 10000
    tokenizer = Tokenizer(num_words=max_vocab)
  
    
    #Convertimos los textos en vectores
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    
    #Tenemos la misma cantidad de largo en cada texto
    X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, padding='post', maxlen=256)
    X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, padding='post', maxlen=256)
    
    
    # Predecimos el modelo
    pred = app.model.predict(X_test)
    
    # Este modelo te da entre un rago negativo y 0.5 para noticias falsas y
    # 0.5 para adelante es verdadero, para orientarlo al eje 0. restamos 0.5
    pred = pred - 0.5
    
    # Esta prediccion nos ayuda a tener una que se podria acercar a porcentajes
    # que es lo que genera nuestra prediccion,
    if(pred > 0 ):
        pro = (98*pred)/(6.89)
        x = "noticia problablemente verdadera"
    else:
        pro = (98*pred)/(-8.6)
        x = "noticia probablemente falsa"
    print(x," con una probablidad de ",pro)
    
    # Concatenemos nuestra respuesta, y enviamos como respuesta a nuestro index
    re = x + " probalidad "+str(pro)
    return render_template('index.html',pred=re)

def main():
    
    """
    Run the app.
    localhost:8000
    debug = false
    """
    app.run(host='0.0.0.0', port=8000, debug=False)  # nosec


if __name__ == '__main__':
    main()