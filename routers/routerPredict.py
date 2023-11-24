from fastapi import APIRouter, Path, Query
# 1. Library imnports

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from models.modelPredict import InputData
from modelPredict.predict_LSTM_RNNA import *
import itertools

# 2. Create the app object
#Create mapp and model objects

#app = FastAPI()
routerPredict = APIRouter()

# 3. Index route, opens automatically on http://127.0.0.1:8000
#4. Route with a single parameter, returns the parameter withina message
#3. Expose the prediction functionality, make a prediction from the passes
#JSON data and return the prediction flower species with the confidence
@routerPredict.post('/predict')
def predict_water_consumption(data: InputData):


    #######################################################
    #Se validan que los datos no vengan vacios
    #Solo estan de 1950 a 2050
    if data.id_anio not in  range(2024, 2050):
        logging.warning('Campo de entrada (Año) invalido')
        logging.warning("No se realizo la predicción con éxito")
        return JSONResponse(status_code=404, content='[ERROR] Año debe estar entre el rango [2024, 2050]')

    #Debe estar entre 0 y 1
    if data.indice_marginacion_normalizada not in np.arange(0, 1.01, 0.01):
        logging.warning('Campo de entrada (Indice de Marginacion Normalizado) invalido')
        logging.warning("No se realizo la predicción con éxito")
        return JSONResponse(status_code=404, content='[ERROR] Indice de Marginacion Normalizado debe estar entre el rango [0,1] con dos decimales')

    #######################################################

    #Transformar los datos
    data = data.dict()  #Crea un diccionario

    # Aquí iría el código para realizar las predicciones con los modelos entrenados
    # Utiliza los valores de entrada `data` y los modelos entrenados para obtener las predicciones

    y_pred_lstm_new, y_pred_ffnn_new, y_test, y_pred_lstm, y_pred_ffnn, mse_lstm_new, mse_ffnn_new =predictionModel(data['id_anio'], data['bimestre'], data['id_tipoEmision'], data['id_tipoUso'], data['id_alcaldia'], data['id_colonia'], data['id_tipoDesarrollo'], data['total_unidades_economicas'], data['poblacion'], data['indice_marginacion_normalizada']).prediccion();
    
    print(y_pred_lstm_new, y_pred_ffnn_new)
    print(y_test)
    print(y_pred_lstm, y_pred_ffnn)
    print(mse_lstm_new, mse_ffnn_new)

    return {
        'y_pred_lstm_new': y_pred_lstm_new[0][0].item(),
        'y_pred_ffnn_new': y_pred_ffnn_new[0][0].item(),
        'y_test': y_test.tolist(),
        'y_pred_lstm': list(itertools.chain( *y_pred_lstm.tolist() )),
        'y_pred_ffnn': list(itertools.chain( *y_pred_ffnn.tolist() )),
        'mse_lstm_new': mse_lstm_new,
        'mse_ffnn_new': mse_ffnn_new
    }


#5. Run the API with uvicorn

#Run the API with uvicorn
#Will run on http://127.0.0.1:8000
# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000)

#Como pasar los parametros
'''
{
    "anio": 2019,
    "bimestre": 4,
    "id_tipoEmision": 2,
    "id_tipoUso": 1,
    "id_alcaldia": 6,
    "id_colonia": 718,
    "manzana": 66,
    "region":  46,
    "id_tipoDesarrollo": 2
}
--Nueva
{
    "id_anio": 2019,
    "bimestre": 2,
    "id_tipoEmision": 1,
    "id_tipoUso": 3,
    "id_alcaldia": 15,
    "id_colonia" : 1181,
    "id_tipoDesarrollo": 4,
    "total_unidades_economicas": 3152,
    "poblacion": 50084,
    "indice_marginacion_normalizada": 0.96
}

'''