from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Cargar el pipeline completo
try:
    modelo = joblib.load('modelo.pkl')   # Pipeline (preprocesador + modelo)
    print("✓ Modelo cargado correctamente")
except Exception as e:
    print(f"Error al cargar modelo: {e}")
    modelo = None

@app.route('/', methods=['GET'])
def index():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if modelo is None:
            return jsonify({'error': 'Modelo no cargado correctamente'}), 500

        # 1. Recoger datos del formulario
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

        df_input = pd.DataFrame([data])  # columnas A1..A15

        # 2. El propio pipeline hace preprocesado + predicción
        proba = modelo.predict_proba(df_input)[0][1]
        pred  = int(proba >= 0.5)

        resultado = "✓ APROBADO" if pred == 1 else "✗ RECHAZADO"
        confianza = max(proba, 1 - proba)

        return jsonify({
            'resultado': resultado,
            'probabilidad': f"{proba:.2%}",
            'confianza': f"{confianza:.2%}"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
