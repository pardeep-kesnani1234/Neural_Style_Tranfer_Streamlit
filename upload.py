import streamlit as st 
from PIL import Image
import numpy as np
import cv2
import PIL
import numpy

st.title("Create Artistic Image with Neural Style Transfer")

# Define Model name
model_name= ['original','feathers', 'candy', 'composition_vii', 'udnie', 'the_wave', 'the_scream', 'mosaic', 'la_muse', 'starry_night']

def init_style_transfer(style,image1):
    #global net
    model ='Model/'+style+".t7" 
    net = cv2.dnn.readNetFromTorch(model)
    pil_image = PIL.Image.open(image1).convert('RGB')
    open_cv_image = numpy.array(pil_image) 
    # Convert RGB to BGR 
    image = open_cv_image[:, :, ::-1].copy() 	
    #image = cv2.imread(open_cv_image)
    # Make prediction
    R,G,B = 103.939, 116.779, 123.680
    blob = cv2.dnn.blobFromImage(image, 1.0, (image.shape[1], image.shape[0]),(R,G,B), swapRB=False, crop=False)
    net.setInput(blob)
    output = net.forward()
    final_output = output.reshape((3, output.shape[2], output.shape[3])).copy()
    final_output[0] += R
    final_output[1] += G
    final_output[2] += B
    final_output = final_output.transpose(1, 2, 0)
    outmid = np.clip(final_output, 0, 255)
    styled= outmid.astype('uint8')
    return styled

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)	
    #st.image(image, caption='Uploaded Image.',width=300)
    st.write("")
    style_model = st.sidebar.radio("Select model",model_name)
    if style_model == "original":
        st.image(image,width=400,caption="origin	al image");	
    else:
        styled =  init_style_transfer(style_model,uploaded_file) 
        st.image(styled,width=400,caption='Styled Image.');
