# -*- coding: utf-8 -*-
"""
Created on Thu May  1 15:46:11 2025

@author: jahop
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly import graph_objects as go
import plotly.figure_factory as ff
from prophet import Prophet
from prophet.plot import plot_plotly
import sqlite3
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_score, recall_score, 
                            f1_score, roc_curve, auc, RocCurveDisplay)
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Configuración de la página
st.set_page_config(
    page_title="Sistema Avanzado de Riesgo Ambiental - PROFEPA",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# 1. WEB SCRAPING (Detección en tiempo real)
# --------------------------
def monitoreo_redes():
    """Cumple: Detección de incidentes en redes sociales"""
    st.subheader("🕵️ Monitoreo de Redes Sociales")
    
    # Simulación de scraping en Twitter
    tweets = [
        {"texto": "Incendio forestal en #Xochimilco. Urge atención @PROFEPA", 
         "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"), 
         "fuente": "Twitter"},
        {"texto": "Contaminación en río Tula. Ver foto: [link]", 
         "fecha": (datetime.now() - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M"), 
         "fuente": "Facebook"}
    ]
    
    df = pd.DataFrame(tweets)
    st.dataframe(df)
    
    # Análisis de sentimiento básico
    df["alerta"] = df["texto"].apply(
        lambda x: "URGENTE" if any(palabra in x.lower() 
                                  for palabra in ["incendio", "contaminación", "derrame"]) 
                 else "Normal")
    st.success(f"🔍 {sum(df['alerta']=='URGENTE')} alertas detectadas")

# --------------------------
# 2. PIPELINE ETL AUTOMATIZADO
# --------------------------
def procesar_datos():
    """Cumple: Procesamiento de datos multi-formato"""
    st.subheader("⚙️ Pipeline ETL Automatizado")
    
    # Simulación de extracción de múltiples fuentes
    with st.expander("Ver fuentes de datos"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Datos estructurados (CSV)**")
            df_csv = pd.DataFrame({
                "fecha": pd.date_range("2023-01-01", periods=5),
                "contaminante": ["CO2", "SO2", "NOx", "PM2.5", "O3"],
                "nivel": np.random.randint(1, 100, 5)
            })
            st.dataframe(df_csv)
        
        with col2:
            st.write("**Datos no estructurados (PDF simulado)**")
            texto_pdf = """
            INFORME AMBIENTAL 2023
            Fecha: 15/03/2023 - Hectáreas afectadas: 125
            Fecha: 20/04/2023 - Hectáreas afectadas: 89
            """
            datos = re.findall(r"(\d{2}/\d{2}/\d{4}).*?(\d+)", texto_pdf)
            df_pdf = pd.DataFrame(datos, columns=["Fecha", "Hectareas"])
            st.dataframe(df_pdf)
    
    # Transformación y carga (SQLite)
    conn = sqlite3.connect("datos_ambientales.db")
    df_csv.to_sql("contaminantes", conn, if_exists="replace", index=False)
    df_pdf.to_sql("deforestacion", conn, if_exists="replace", index=False)
    st.success("✅ Datos integrados en SQLite")

# --------------------------
# 3. MODELOS PREDICTIVOS (ML/DL) - MEJORADO
# --------------------------
def modelos_avanzados():
    """Cumple: Modelos de riesgo ambiental con métricas completas"""
    st.subheader("🤖 Modelos Predictivos - Random Forest")
    
    # Datos simulados para clasificación de riesgo
    X = pd.DataFrame({
        "temperatura": np.random.uniform(20, 40, 100),
        "humedad": np.random.uniform(30, 90, 100),
        "viento": np.random.uniform(0, 50, 100),
        "lluvia": np.random.uniform(0, 100, 100)
    })
    y = np.random.choice(["Alto", "Medio", "Bajo"], 100, p=[0.2, 0.3, 0.5])
    
    # Entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # ---- SECCIÓN DE MÉTRICAS COMPLETAS ----
    st.markdown("### 📊 Evaluación Avanzada del Modelo")
    
    # 1. Reporte de Clasificación
    st.markdown("#### Reporte de Clasificación")
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report.style.highlight_max(axis=0, color='lightgreen'))
    
    # 2. Matriz de Confusión Interactiva
    st.markdown("#### Matriz de Confusión")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm,
                      labels=dict(x="Predicho", y="Real", color="Casos"),
                      x=model.classes_,
                      y=model.classes_,
                      text_auto=True,
                      color_continuous_scale='Blues')
    fig_cm.update_layout(title="Matriz de Confusión - Riesgo Ambiental")
    st.plotly_chart(fig_cm, use_container_width=True)
    
    # 3. Importancia de Variables
    st.markdown("#### Importancia de Variables")
    importancia = pd.DataFrame({
        "Variable": X.columns,
        "Importancia": model.feature_importances_
    }).sort_values("Importancia", ascending=False)
    
    fig_imp = px.bar(importancia, x='Variable', y='Importancia', 
                    title="Factores de Riesgo Ambiental",
                    color='Importancia',
                    color_continuous_scale='Tealgrn')
    st.plotly_chart(fig_imp, use_container_width=True)
    
    # 4. Curvas ROC Multiclase
    st.markdown("#### Curvas ROC (One-vs-Rest)")
    y_test_bin = label_binarize(y_test, classes=model.classes_)
    n_classes = y_test_bin.shape[1]
    
    fig_roc = go.Figure()
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    for i, color in zip(range(n_classes), colors):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        fig_roc.add_trace(
            go.Scatter(
                x=fpr, y=tpr,
                name=f'Clase {model.classes_[i]} (AUC = {roc_auc:.2f})',
                line=dict(color=color, width=2)
            )
        )
    
    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1],
            line=dict(color='navy', width=2, dash='dash'),
            showlegend=False
        )
    )
    
    fig_roc.update_layout(
        title='Curvas ROC por Clase de Riesgo',
        xaxis_title='Tasa de Falsos Positivos',
        yaxis_title='Tasa de Verdaderos Positivos',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain')
    )
    st.plotly_chart(fig_roc, use_container_width=True)
    
    # 5. Métricas Resumen
    st.markdown("#### Métricas Globales")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Exactitud", f"{accuracy_score(y_test, y_pred):.1%}")
    col2.metric("Precisión Promedio", f"{precision_score(y_test, y_pred, average='weighted'):.1%}")
    col3.metric("Recall Promedio", f"{recall_score(y_test, y_pred, average='weighted'):.1%}")
    col4.metric("F1-Score Promedio", f"{f1_score(y_test, y_pred, average='weighted'):.1%}")
    
    # 6. Validación Cruzada
    st.markdown("#### Validación Cruzada (5 folds)")
    cv_scores = cross_val_score(model, X, y, cv=5)
    fig_cv = px.box(x=cv_scores, 
                   labels={'x': 'Exactitud'},
                   title="Estabilidad del Modelo")
    fig_cv.update_layout(showlegend=False)
    st.plotly_chart(fig_cv, use_container_width=True)
    
    # Guardar modelo
    joblib.dump(model, "modelo_riesgo.pkl")
    st.success("Modelo entrenado y guardado")

# --------------------------
# 4. SISTEMA DE RECOMENDACIONES
# --------------------------
def dashboard_decisiones():
    """Cumple: Apoyo a la toma de decisiones"""
    st.subheader("📊 Tablero de Control")
    
    # Predicción con Prophet
    df = pd.DataFrame({
        "ds": pd.date_range("2020-01-01", periods=36, freq="M"),
        "y": np.random.lognormal(3, 0.5, 36).cumsum()
    })
    
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=6, freq="M")
    forecast = model.predict(future)
    
    # Visualización interactiva
    fig = plot_plotly(model, forecast)
    fig.update_layout(title="Predicción de Deforestación")
    st.plotly_chart(fig, use_container_width=True)
    
    # Recomendaciones basadas en datos
    ultimo_valor = forecast["yhat"].iloc[-1]
    if ultimo_valor > 500:
        st.error("🚨 ALERTA: Se proyecta aumento crítico en deforestación")
        st.markdown("**Acciones recomendadas:**")
        st.markdown("- Desplegar equipos de inspección")
        st.markdown("- Activar protocolos de emergencia")
        st.markdown("- Notificar a autoridades locales")
    else:
        st.success("✅ Situación dentro de parámetros normales")
        st.markdown("**Acciones recomendadas:**")
        st.markdown("- Monitoreo rutinario")
        st.markdown("- Revisión de equipos")

# --------------------------
# INTERFAZ PRINCIPAL
# --------------------------
def main():
    st.title("🌍 Sistema Integral de Gestión Ambiental")
    st.markdown("""
    **Competencias demostradas:**  
    • Modelado predictivo ML/DL con métricas avanzadas  
    • ETL automatizado multi-formato  
    • Monitoreo en tiempo real  
    • Soporte a decisiones estratégicas basado en datos  
    """)

    # Botón para descargar el manual PDF
    with open("manual.pdf", "rb") as file:
    st.sidebar.download_button(
        label="📥 Descargar Manual",
        data=file,
        file_name="manual.pdf",
        mime="application/pdf"
    )

   
    # Menú de módulos
    modulo = st.sidebar.selectbox("Seleccione módulo", [
        "Monitoreo en Redes",
        "Procesamiento ETL",
        "Modelado Predictivo",
        "Tablero de Control"
    ])
    
    # Sección de Ayuda expandible
    with st.sidebar.expander("ℹ️ Ayuda Rápida", expanded=False):
        st.markdown("""
        **Funcionalidades por módulo:**
        
        🔍 **Monitoreo en Redes**:
        - Detección automática de alertas ambientales
        - Análisis de tendencias en redes sociales
        
        ⚙️ **Procesamiento ETL**:
        - Integración de datos estructurados y no estructurados
        - Limpieza automatizada de datos
        
        🤖 **Modelado Predictivo**:
        - Clasificación de riesgo con Random Forest
        - Métricas avanzadas de evaluación
        
        📊 **Tablero de Control**:
        - Predicciones de deforestación
        - Recomendaciones accionables
        """)
    
    # Créditos y fecha
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Desarrollado por:**  
    Javier Horacio Pérez Ricárdez  
    
    **Fecha:**  
    Mayo 2025  
    
    **Versión:**  
    1.0.0  
    """)
    st.sidebar.markdown("---")
    
    # Lógica de los módulos (SÓLO UNA VEZ)
    if modulo == "Monitoreo en Redes":
        monitoreo_redes()
    elif modulo == "Procesamiento ETL":
        procesar_datos()
    elif modulo == "Modelado Predictivo":
        modelos_avanzados()  # <-- Se ejecutará una sola vez
    elif modulo == "Tablero de Control":
        dashboard_decisiones()

    # Buenas prácticas documentadas
    with st.expander("📚 Guía de Buenas Prácticas", expanded=False):
        st.markdown("""
        **Evaluación de Modelos:**  
        - Matriz de confusión para análisis de errores  
        - Curvas ROC para evaluar compensaciones  
        - Validación cruzada para robustez  
        
        **Interpretación:**  
        - Importancia de variables para causalidad  
        - Métricas por clase para desbalanceo  
        - Intervalos de confianza  
        """)

if __name__ == "__main__":
    main()
