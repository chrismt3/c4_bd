
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Recomendador de Películas - Arquitectura Lambda", layout="wide")

# -----------------------
# Datos simulados (Capa Batch)
# -----------------------
st.title("🎬 Recomendador de Películas con Arquitectura Lambda")
st.caption("Simulación educativa: Capa Batch + Capa de Velocidad + Capa de Servicio")

movies = ["Matrix", "Avatar", "Titanic", "Inception", "Joker", "Toy Story", "Interstellar"]
users = ["Ana", "Luis", "Carlos", "María"]

ratings = pd.DataFrame(
    np.random.randint(1, 6, size=(len(users), len(movies))),
    index=users,
    columns=movies
)

st.subheader("🧩 Capa Batch - Calificaciones Históricas")
st.write("La capa batch procesa los datos históricos de los usuarios para construir un modelo base.")
st.dataframe(ratings)

# Entrenamiento batch (modelo de similitud)
similarity = pd.DataFrame(
    cosine_similarity(ratings.T),
    index=movies,
    columns=movies
)

st.subheader("🤖 Modelo Batch - Similitud entre películas")
st.write("Calculado mediante similitud coseno entre películas basadas en las calificaciones.")
st.dataframe(similarity.round(2))

# -----------------------
# Capa de Velocidad (Simulación de nuevos datos)
# -----------------------
st.header("⚡ Capa de Velocidad - Ingreso en Tiempo Real")

user = st.selectbox("Selecciona un usuario:", users)
movie = st.selectbox("Selecciona una película:", movies)
rating = st.slider("Nueva calificación", 1, 5, 3)

if st.button("➕ Agregar Calificación"):
    ratings.loc[user, movie] = rating
    st.success(f"✅ {user} calificó '{movie}' con {rating} estrellas.")
    similarity = pd.DataFrame(
        cosine_similarity(ratings.T),
        index=movies,
        columns=movies
    )

# -----------------------
# Capa de Servicio (Fusión de resultados)
# -----------------------
st.header("🛰️ Capa de Servicio - Recomendaciones Actualizadas")

selected_movie = st.selectbox("Selecciona una película base:", movies)
recommendations = similarity[selected_movie].sort_values(ascending=False)[1:4]

st.write("🔍 Películas similares a:", selected_movie)
st.dataframe(recommendations.round(2), use_container_width=True)

st.info("💡 Cada nueva calificación actualiza la matriz de similitud y las recomendaciones al instante.")
