from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Cargar modelo y transformadores entrenados en Sprint 2 y Sprint 3
try:
    modelo = joblib.load('modelo.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    print("✓ Modelo y preprocessor cargados correctamente")
except Exception as e:
    print(f"Error al cargar modelo: {e}")
    modelo = None
    preprocessor = None

@app.route('/', methods=['GET'])
def index():
    """Página principal con el formulario"""
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint que recibe datos y retorna predicción"""
    try:
        if modelo is None or preprocessor is None:
            return jsonify({'error': 'Modelo no cargado correctamente'}), 500
        
        # Obtener datos del formulario
        data = {
            'A1': request.form.get('A1'),
            'A2': float(request.form.get('A2')),
            'A3': float(request.form.get('A3')),
            'A4': request.form.get('A4'),
            'A5': request.form.get('A5'),
            'A6': request.form.get('A6'),
            'A7': request.form.get('A7'),
            'A8': float(request.form.get('A8')),
            'A9': request.form.get('A9'),
            'A10': request.form.get('A10'),
            'A11': float(request.form.get('A11')),
            'A12': request.form.get('A12'),
            'A13': request.form.get('A13'),
            'A14': float(request.form.get('A14')),
            'A15': float(request.form.get('A15')),
        }
        
        # Crear DataFrame con los datos
        df_input = pd.DataFrame([data])
        
        # Aplicar el mismo preprocesamiento
        X_processed = preprocessor.transform(df_input)
        
        # Realizar predicción
        prediction = modelo.predict(X_processed)[0]
        probability = modelo.predict_proba(X_processed)[0][1]
        
        resultado = "✓ APROBADO" if prediction == 1 else "✗ RECHAZADO"
        
        return jsonify({
            'resultado': resultado,
            'probabilidad': f"{probability:.2%}",
            'confianza': f"{max(probability, 1-probability):.2%}"
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
