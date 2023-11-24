import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
#import logging
from loguru import logger as logging


class predictionModel:
    # 6. Class constructor, loads the dataset and loads the model
    # if existss. If not, calls the _train_model method and
    # saves the model
    '''
    logging.basicConfig(
        #filename='log_file_name.log',
        level=logging.DEBUG, 
        #format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        format= '[%(asctime)s] {%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )'''

    logging.add("./Logs/logsFile.log", enqueue=True)

    logging.debug('Se inician las variables de la clase')
    
    def __init__(self, id_anio, bimestre, id_tipoEmision, id_tipoUso, id_alcaldia , id_colonia, id_tipoDesarrollo, total_unidades_economicas, poblacion, indice_marginacion_normalizada ):

        self.df = pd.read_csv('./dataWater/data_integrada_c.csv')
        logging.debug('Se carga el archivo de consumo de agua')

        #self.model_fname_ = 'iris_model.pkl'
        self.id_anio = id_anio
        self.bimestre = bimestre
        self.id_tipoEmision = id_tipoEmision
        self.id_tipoUso = id_tipoUso
        self.id_alcaldia = id_alcaldia
        self.id_colonia = id_colonia
        self.id_tipoDesarrollo = id_tipoDesarrollo
        self.total_unidades_economicas = total_unidades_economicas
        self.poblacion = poblacion 
        self.indice_marginacion_normalizada = indice_marginacion_normalizada
        #self.consumo = consumo
        logging.debug('Se asignan el valor de las variables')

    def imprimirDatos(self):
        print(self.df)
        logging.debug('Se imprimen los datos')

    def cargarDatos(self):
        logging.debug('Se terminan de cargar los datos')
        return self.df
        
    def convertirDatos(self):
        logging.debug('Se transforman los datos')
        return self.df.to_numpy().tolist()
    
    def recortarDatos(self):
        logging.debug('Se recorta al 10% los datos')
        #return self.df.sample(frac = 0.05, random_state = 18)
        return self.df.sample(frac = 0.1, random_state = 18)

    def prediccion(self):

        logging.debug('Se inicia la prediccion')

        # Datos de ejemplo

        #self.imprimirDatos();

        #Sirve si son todos
        #data = np.array(self.convertirDatos())

        data = self.recortarDatos();
        #print(data);
        data = np.array(data.to_numpy().tolist());
        #print(data);

        logging.debug('Se convierten a datos numpy')

        # Dividir los datos en características (X) y etiquetas (y)
        X = data[:, :-1]
        y = data[:, -1]
        logging.debug('Se dividen los datos')

        # Escalar las características
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        logging.debug('Se escalan datos y caracteriticas')

        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.debug('Se dividen los datos en conjuntos de entrenamiento y prueba')

        # Reshape de los datos para LSTM
        X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        logging.debug('Reshape de los datos para LSTM')

        # Crear el modelo LSTM
        lstm_model = Sequential()
        lstm_model.add(LSTM(50, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
        lstm_model.add(Dense(1))
        lstm_model.compile(optimizer='adam', loss='mse')
        logging.debug('Crear el modelo LSTM')

        # Entrenar el modelo LSTM
        # lstm_model.fit(X_train_lstm, y_train, epochs=100, batch_size=1, verbose=0)
        #Se cambia el numero de epocas
        #lstm_model.fit(X_train_lstm, y_train, epochs=5, batch_size=1, verbose=0)
        lstm_model.fit(X_train_lstm, y_train, epochs=1, batch_size=1, verbose=0)
        logging.debug('Entrenar el modelo LSTM')

        # Realizar predicciones con el modelo LSTM
        y_pred_lstm = lstm_model.predict(X_test_lstm)
        logging.debug('Realizar la prediccion con el modelo LSTM')

        # Reshape de los datos para FFNN
        X_train_ffnn = X_train
        X_test_ffnn = X_test
        logging.debug('Realizar la division para FFNN')

        # Crear el modelo FFNN
        ffnn_model = Sequential()
        ffnn_model.add(Dense(50, activation='relu', input_shape=(X_train_ffnn.shape[1],)))
        ffnn_model.add(Dense(1))
        ffnn_model.compile(optimizer='adam', loss='mse')
        logging.debug('Crear el modelo FFNN')

        # Entrenar el modelo FFNN
        #ffnn_model.fit(X_train_ffnn, y_train, epochs=100, batch_size=1, verbose=0)
        #Se cambian el numero de epocas
        #ffnn_model.fit(X_train_ffnn, y_train, epochs=5, batch_size=1, verbose=0)
        ffnn_model.fit(X_train_ffnn, y_train, epochs=1, batch_size=1, verbose=0)
        logging.debug('Entrenar el modelo FFNN')

        # Realizar predicciones con el modelo FFNN
        y_pred_ffnn = ffnn_model.predict(X_test_ffnn)
        logging.debug('Realizar predicciones con el modelo FFNN')

        # Agregar el nuevo dato a los datos de prueba
        new_data = np.array([ self.id_anio, self.bimestre, self.id_tipoEmision, self.id_tipoUso, self.id_alcaldia , self.id_colonia, self.id_tipoDesarrollo, self.total_unidades_economicas, self.poblacion, self.indice_marginacion_normalizada ])  #Nuevo dato, checar
        new_data_scaled = scaler.transform(new_data.reshape(1, -1))
        logging.debug('Agregar el nuevo dato a los datos de prueba')

        # Realizar predicciones con el nuevo dato
        new_data_lstm = new_data_scaled.reshape((1, 1, new_data_scaled.shape[1]))
        new_data_ffnn = new_data_scaled.reshape((1, new_data_scaled.shape[1]))
        y_pred_lstm_new = lstm_model.predict(new_data_lstm)
        y_pred_ffnn_new = ffnn_model.predict(new_data_ffnn)
        logging.debug('Realizar predicciones con el nuevo dato')

        # Unir los datos de prueba con el nuevo dato
        X_test_all = np.concatenate((X_test, new_data_scaled))

        y_test_all = np.concatenate((y_test, [new_data[-1]]))
        y_pred_lstm_all = np.concatenate((y_pred_lstm.flatten(), y_pred_lstm_new.flatten()))
        y_pred_ffnn_all = np.concatenate((y_pred_ffnn.flatten(), y_pred_ffnn_new.flatten()))
        logging.debug('Unir los datos de prueba y el nuevo dato')

        # Calcular el error cuadrático medio con el nuevo dato
        mse_lstm_new = mean_squared_error(y_test_all, y_pred_lstm_all)
        mse_ffnn_new = mean_squared_error(y_test_all, y_pred_ffnn_all)

        logging.debug('Calculo de errores cuadraticos')

        print("Error cuadrático medio (LSTM) con el nuevo dato:", mse_lstm_new)
        print("Error cuadrático medio (FFNN) con el nuevo dato:", mse_ffnn_new)

        # Graficar las predicciones
        
        '''
        plt.plot(y_test, label='Valores reales')
        plt.plot(y_pred_lstm, label='Predicciones LSTM')
        plt.plot(y_pred_ffnn, label='Predicciones FFNN')
        plt.plot(len(y_test), y_pred_lstm_new, 'ro', label='Predicción LSTM (Nuevo dato)')
        plt.plot(len(y_test), y_pred_ffnn_new, 'go', label='Predicción FFNN (Nuevo dato)')
        plt.xlabel('Muestras')
        plt.ylabel('Consumo de agua')
        plt.legend()
        plt.show()
        '''
        

        #Solo el 20% de test son los que se predecin los demas es para que se entrene el 80%
        return y_pred_lstm_new, y_pred_ffnn_new, y_test, y_pred_lstm, y_pred_ffnn, mse_lstm_new, mse_ffnn_new


if __name__ == "__main__":

    #predictionModel(2019,2,1,3,15,1181,4,3152,50084,0.96).imprimirDatos()
    y_pred_lstm_new, y_pred_ffnn_new, y_test, y_pred_lstm, y_pred_ffnn, mse_lstm_new, mse_ffnn_new =predictionModel(2019,2,1,3,15,1181,4,3152,50084,0.96).prediccion();
    print(f'Valor de LSTM: {y_pred_lstm_new}, Valor de FFNN: {y_pred_ffnn_new}')

'''
modeloPrediccion =predictionModel().cargarDatos();
print(f'{modeloPrediccion}')
print(type(modeloPrediccion))


modeloLista =predictionModel().convertirDatos();
print(f'{modeloLista}')
print(type(modeloLista))
'''

'''
prediccion =predictionModel.prediccion();
'''

'''
data = np.array([
    [2017, 1, 1, 2, 10, 1064, 7, 54, 1, 0],
    [2017, 2, 1, 2, 10, 1688, 122, 77, 2, 47.58],
    [2017, 4, 1, 3, 10, 241, 221, 54, 1, 642.34],
    [2018, 3, 1, 3, 17, 1474, 210, 22, 2, 18.91],
    [2018, 6, 1, 1, 17, 1447, 226, 23, 1, 1058.04],
    [2017, 1, 1, 3, 7, 795, 48, 565, 2, 115.05],
    [2018, 1, 1, 1, 13, 699, 401, 372, 3, 14.16],
    [2019, 3, 2, 1, 5, 550, 115, 16, 2, 19.91],
    [2017, 4, 2, 1, 6, 718, 66, 46, 2, 234.85],
    [2018, 5, 1, 3, 17, 1447, 187, 23, 2, 73.8],
    [2018, 3, 1, 2, 17, 1474, 64, 22, 2, 61.98]
])

print(data)
print(type(data))

dataCrudo = [
    [2017, 1, 1, 2, 10, 1064, 7, 54, 1, 0],
    [2017, 2, 1, 2, 10, 1688, 122, 77, 2, 47.58],
    [2017, 4, 1, 3, 10, 241, 221, 54, 1, 642.34],
    [2018, 3, 1, 3, 17, 1474, 210, 22, 2, 18.91],
    [2018, 6, 1, 1, 17, 1447, 226, 23, 1, 1058.04],
    [2017, 1, 1, 3, 7, 795, 48, 565, 2, 115.05],
    [2018, 1, 1, 1, 13, 699, 401, 372, 3, 14.16],
    [2019, 3, 2, 1, 5, 550, 115, 16, 2, 19.91],
    [2017, 4, 2, 1, 6, 718, 66, 46, 2, 234.85],
    [2018, 5, 1, 3, 17, 1447, 187, 23, 2, 73.8],
    [2018, 3, 1, 2, 17, 1474, 64, 22, 2, 61.98]
]

print(dataCrudo)
print(type(dataCrudo));
'''
