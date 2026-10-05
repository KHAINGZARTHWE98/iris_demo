import streamlit as st
from joblib import load

st.set_page_config(
    page_title="Iris Feature Sliders",
    layout="wide"
)

st.title("🌸 Iris Flower Classification")

st.subheader("Selected Feature Values")

sepal_length = st.slider("Sepal Length", 2.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width", 2.0, 5.0, 3.0)
petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width", 0.1, 2.5, 1.0)

model_path = r"C:\KZTWorkingFile\mine\tbc\ML\saved_ML\iris_model_1.pkl"

if st.button("Classify"):

    try:
        # Load model
        model = load(model_path)

        # Prediction data
        data = [[
            
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]]

        result = model.predict(data)[0]

        flower_names = [
            "Iris Setosa",
            "Iris Versicolor",
            "Iris Virginica"
        ]

        st.success(f"Prediction: {flower_names[result]}")

        if result == 0:
            st.image(
                "setosa.jfif",
                caption=flower_names[result]
            )

        elif result == 1:
            st.image(
                "versicolor.jfif",
                caption=flower_names[result]
            )

        else:
            st.image(
                "virginica.jfif",
                caption=flower_names[result]
            )

    except Exception as e:
        st.error(f"Error loading model: {e}")