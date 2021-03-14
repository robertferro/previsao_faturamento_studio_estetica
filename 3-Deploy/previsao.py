import numpy as np
import pickle
import streamlit as st

modelo = open('modelo_faturamento', 'rb')
new_model = pickle.load(modelo)
modelo.close()

dic_dia_semana = {'segunda':1, 'terca':2,'quarta':3, 
                  'quinta':4 ,'sexta':5, 'sabado':6 }


def preparing(dia_da_semana, qtde_clientes):
    dia_prep = dic_dia_semana[dia_da_semana.lower()]
    qtde_clientes_prep = int(qtde_clientes)
    features = np.r_[dia_prep,qtde_clientes_prep].reshape(1,-1)
    return features


def main():
    st.title("Previsão Faturamento")
    dia_da_semana = st.selectbox("Dia da semana",("segunda","terca","quarta","quinta","sexta","sabado"))
    qtde_clientes = st.text_input("Nº de agendamentos")
    pred = st.button("Predict")
    if pred:
        features = preparing(dia_da_semana = dia_da_semana, qtde_clientes = qtde_clientes)
        prediction = round(np.expm1(new_model.predict(features))[0], 2)
        output = st.success(f" a previsão de faturamento é {prediction}")

if __name__== '__main__':
    main()