import streamlit as st
import numpy as np

# Clase Neuron
class Neuron:
    def __init__(self, weights, bias, func):
        self.weights = weights
        self.bias = bias
        self.tipo_funcion = func

    def run(self, input_data):
        y = sum(w * x for w, x in zip(self.weights, input_data)) + self.bias
        output = getattr(Neuron, f'_Neuron__{self.tipo_funcion}')(y)
        return output
    
    def change_bias(self, new_bias):
        self.bias = new_bias

    @staticmethod
    def __relu(value):
        return max(0, value)

    @staticmethod
    def __sigmoid(value):
        return 1 / (1 + np.exp(-value))

    @staticmethod
    def __tangente(value):
        return np.tanh(value)

# Cabecera
st.image('img/neurona.jpg', width=300)

st.title('Simulador de neurona')

num = st.slider('Elige números de entradas/pesos que tendrá la neurona', 1, 10, key=1)

# Apartado Pesos
st.header("Pesos")

columns_w = st.columns(num)

weights = []

for i, column_w in enumerate(columns_w):
    key = f'W{i}'
    value = column_w.number_input(key, key=key)
    weights.append(value)  

st.text(f'w = {weights}')

# Apartado Entradas
st.header("Entradas")

columns_x = st.columns(num)

inputs = []

for i, column_x in enumerate(columns_x):
    key = f'X{i}'
    value = column_x.number_input(key, key=key)
    inputs.append(value)  

st.text(f'x = {inputs}')

col1, col2 = st.columns(2)

# Apartado Sesgo
col1.header("Sesgo")
b = col1.number_input('Introduca el valor del sesgo', key='b')

# Apartado Función de activación
col2.header("Función de activación")

funciones_activacion = {
    'Sigmoide': 'sigmoid',
    'ReLU': 'relu',
    'Tangente hiperbólica': 'tangente'
}

option = col2.selectbox(
    'Elige la función de activación',
    list(funciones_activacion.keys()), key='func'
)

funcion = funciones_activacion[option]

# Resultado
if st.button("Calcular la salida", key=16):
    n1 = Neuron(weights=weights, bias=b, func=funcion)
    x = inputs

    output = n1.run(input_data=x)
    st.write('La salida de la neurona es', output)