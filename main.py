from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# FastAPI-App initialisieren
app = FastAPI()

# CORS-Einstellungen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Erlaube Anfragen von allen Ursprüngen
    allow_credentials=True,  # Erlaube Cookies in Anfragen
    allow_methods=["*"],  # Erlaube bestimmte HTTP-Methoden
    allow_headers=["*"],  # Erlaube alle Header in Anfragen
)



# import model
with open('xgb_phone_price_model.sav', 'rb') as f:
    price_model = pickle.load(f)

# import mapping
with open('mapping_data.pkl', 'rb') as f:
    mapping_data = pickle.load(f)

# Klasse für inputs
class PhoneInput(BaseModel):
    brand: str
    processor_brand: str
    processor: str
    internal_storage: int
    ram: float
    battery_capacity_mAh: float
    rar_camera_total_mp: float
    front_camera_mp: float
    disp_size_inch: float
    


# Endpoint erstellen
@app.post('/phone_price_prediction')
def phone_price_prediction(inputs: PhoneInput):
    brand_num = mapping_data['brand_mapping'].get(inputs.brand)
    process_brand_num = mapping_data['process_brand_mapping'].get(inputs.processor_brand)
    processor_num = mapping_data['processor_mapping'].get(inputs.processor)
    
    # Überprüfung
    if None in (brand_num, process_brand_num, processor_num):
        return {"error": "Ungültige Eingabe für Marke, Prozessormarke oder Prozessor."}

 
    input_data = np.array([
        brand_num,
        process_brand_num,
        processor_num,
        inputs.internal_storage,
        inputs.ram,
        inputs.battery_capacity_mAh,
        inputs.rar_camera_total_mp,
        inputs.front_camera_mp,
        inputs.disp_size_inch,
        
        
    ]).reshape(1, -1)

    # Vorhersage erstellen
    pred_price = price_model.predict(input_data)

    # zwei Dezimalstellen
    pred_price_rounded = round(pred_price[0], 2)

    # Konvertierung der Vorhersage in JSON-Format
    response = {"prediction": float(pred_price_rounded)}  # Vorhersage als float umwandeln


    return response
