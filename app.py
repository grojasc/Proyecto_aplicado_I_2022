
import streamlit as st
import pandas as pd
import numpy as np
import pickle

from PIL import Image

def main():
  st.title("Alamacenes La Pelar")
  #st.subheader("Siempre todo original")
  #img = Image.open("/content/descarga.png")
  #st.image(img,width=100,caption="La Pelar")
  st.markdown("""### _Elige los mejores productos originales de nuestras marcas adides y under amor_ :sunglasses:""",True)

  menu = ["Home","About","Login","SignUp"]
  choice = st.sidebar.selectbox('Menu',menu)
  if choice == 'Home':
    st.subheader("Inicio")	

  elif choice == "Login":
    st.subheader("Login Section")	
    username  = st.sidebar.text_input("User Name")
    password = st.sidebar.text_input("Password",type='password' )
    if st.sidebar.checkbox("Login"):
      if password == '12345':
        st.success("Logged In as {}".format(username))

        file = st.file_uploader("Subir aquí la imagen ", type=["jpg", "jpeg", "png"])
        if file is None:
          st.text("Por favor subir una imagen")
      else:
        st.warning("Incorrect UserName/Password")

  elif choice == "SignUp":
    st.subheader("Create New Account")	
    new_user = st.text_input("UserName")
    new_password = st.text_input("Password",type = 'password')
    if st.button("SignUp"):
      st.success("you have successfully created an Account")
      st.info("Go to Login Menu")


















if __name__ == '__main__':
  main()
