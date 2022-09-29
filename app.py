# Imports
import os
import base64
import random
import tensorflow as tf
import numpy as np
import streamlit as st
from st_clickable_images import clickable_images
from PIL import Image
from time import sleep


# General
IMG_SIZE = (224, 224)
label_dict = {1: 'food 🍖', 2: 'inside ⛩', 0: 'drink 🍹', 4: 'outside 🌅', 3: 'menu 📖'}
options = ['food', 'inside', 'drink', 'outside', 'menu']
model_dir = 'model/tf_model_exp1_0a07a63d76db48a8a632dbb3a88e505d'
st_options = ['👨🏻‍💻Backend', '🤖Frontend']

# BACKEND. Custom metrics, Model load, Predict & Image Load.

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall) / (precision+recall+K.epsilon()))

def convert_images(label):
    base_dir = 'assets/' + label
    image_list = os.listdir(base_dir)
    random.shuffle(image_list)

    images = []
    for file in image_list[:10]:
        with open(base_dir + "/" + file, "rb") as image:
            encoded = base64.b64encode(image.read()).decode()
            images.append(f"data:image/jpeg;base64,{encoded}")

    return images

def convert_and_save(img):
    with open(img, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
        converted = f"data:image/jpeg;base64,{encoded}"

    with open("img_local_info.txt", "a+") as f:
        f.write(converted)


def load():
    # Load model
    model = tf.keras.models.load_model(model_dir, custom_objects={'f1_m': f1_m,
                                                                  'recall_m': recall_m,
                                                                  'precision_m': precision_m})
    img_dict = {}
    for i in options:
        img_dict[i] = convert_images(i)

    return model, img_dict

model, img_dict = load()


def predict_label(img):
    # Resize image & Predict Label
    im_array = np.asarray(img.resize(IMG_SIZE))
    im_normalized = im_array / 225

    model_predict = model.predict(tf.expand_dims(im_normalized, 0))
    predicted_label = np.argmax(model_predict, axis=1)
    predicted_label_text = label_dict[int(predicted_label)]
    prob_score = round((model_predict[0][predicted_label][0] * 100), 2)

    return f'La categoría es: {predicted_label_text}, con una probabilidad del {prob_score} %'

# FRONTEND.

def image_showcase(images):

    clicked = clickable_images(
        images,
        titles=[f"Image #{str(i)}" for i in range(len(images))],
        div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
        img_style={"margin": "5px", "height": "200px"},
    )

    return clicked

def side_bar_info():
    st.sidebar.image("thevalley_bluetab.png", use_column_width=True)   #width=200
    st.sidebar.markdown("<h1 style='color: grey;'>MDS+ Trabajo fin de master: Sistema de clasificación de imágenes</h1>",
                        unsafe_allow_html=True)
    st.sidebar.write("Made with ❤️ by Pedro Tavares, Carlos Huguet, Carlos Cejas & Toni Vila")

def main():

    show_side_bar_options = side_bar_info()
    sidebar_option = st.sidebar.selectbox("¿Qué te gustaría ver?", st_options)


    if sidebar_option == st_options[0]:

        #@st.cache(suppress_st_warning=True)
        def backend():
            st.markdown("# 👨🏻‍💻 Backend. Etiquetado de imágenes.")
            st.info("""Esta sección simula un proceso de *backend* en el cual se cagarían en *batch* las fotos que han sido scrappeadas de distintas fuentes
                    (social media, portales de opinión, etc.) y subidas por los usuarios para su proceder a su etiquetado.""")

            st.markdown('### Cargar y etiquetar foto:')
            uploaded_file = st.file_uploader("❗️Selecciona una imagen para clasificar:")


            if uploaded_file is not None:

                col1, col2 = st.columns(2)
                with col1:
                    im = Image.open(uploaded_file)
                    st.markdown("### Input Image")
                    st.image(im)

                    #convert_and_save(uploaded_file)

                with col2:
                    st.markdown("### Predict")
                    with st.spinner("#### 🔥🔥 Espera por favor"):
                        result = st.success(f'##### {predict_label(im)}')
                    st.markdown('**¿Está la foto bien clasificada?**')
                    button_ok = st.button('✅')
                    button_ko = st.button('❌')

                    if button_ok:
                        st.checkbox('Gracias! Sube otra foto para clasificar')

                    if button_ko:
                        radio_button = st.radio("Gracias por el *feedback*! ¿Qué categoría le asignamos?", options)

        call_backend = backend()


    elif sidebar_option == st_options[1]:
    #@st.cache(suppress_st_warning=True)
        def frontend():
            st.markdown("# 🤖 Frontend: Explorador de imágenes ")
            st.info("""Esta sección simula la experiencia que podría tener el usuario final, que vería desde web / app la galería de imágenes
            y sobre la cual podría filtrar por las distintas categorías. Las imágenes se incluyen en la galería una vez han sido etiquetadas""")
            st.warning("""A efectos de la demo y para limitar los tiempos de carga, únicamente se muestran 10 imágenes de cada categoría. 
            Las imágenes se muestran de forma aleatoria en cada carga.""")

            st.markdown("---")

            option = st.selectbox("Escoge una categoría:", options)
            with st.spinner("#### No te desesperes. Estamos buscando las mejores fotos para ti. 🙃"):
                image_showcase(img_dict[option])

        call_frontend = frontend()

    else:
        pass



if __name__ == "__main__":
    main()
