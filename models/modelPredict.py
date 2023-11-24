# 1.Library imports
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from pydantic import BaseModel
import joblib

# 2. Class which describes a single flower measurements
class InputData (BaseModel):
    id_anio: int
    bimestre: int
    id_tipoEmision: int
    id_tipoUso: int
    id_alcaldia : int
    id_colonia : int
    id_tipoDesarrollo: int
    total_unidades_economicas: int
    poblacion: int
    indice_marginacion_normalizada: float
    #consumo: float