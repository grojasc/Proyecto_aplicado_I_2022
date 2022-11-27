import streamlit as st
import io
import numpy as np
import os
import json
from torchvision import models, transforms
model = models.vgg16(pretrained=True)
import cv2
import heapq
# from google.colab.patches import cv2_imshow
import torch
from torch import optim, nn
from numpy.ma.core import argmin
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
#from keras.preprocessing import image
from PIL import ImageFile
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
#from yellowbrick.cluster import KElbowVisualizer


vgg19_featmat = np.load('features_numpy.npy', allow_pickle=True)

with open('lista_images.json') as f:
    index2id = json.load(f)

def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None

    


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
            # Extract VGG-16 Feature Layers
        self.features = list(model.features)
        self.features = nn.Sequential(*self.features)
            # Extract VGG-16 Average Pooling Layer
        self.pooling = model.avgpool
            # Convert the image into one-dimensional vector
        self.flatten = nn.Flatten()
            # Extract the first part of fully-connected layer from VGG16
        self.fc = model.classifier[0]
  
    def forward(self, x):
        # It will take the input 'x' until it returns the feature vector called 'out'
        out = self.features(x)
        out = self.pooling(out)
        out = self.flatten(out)
        out = self.fc(out) 
        return out 

def load_model():
# Initialize the model
    model = models.vgg16(pretrained=True)
    new_model = FeatureExtractor(model)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    # Change the device to GPU
    new_model = new_model.to(device)
    
    return new_model

def preprocess_image(model, image):
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    new_model = model
    transform = transforms.Compose([
    transforms.CenterCrop(512),
    transforms.Resize(448),
    transforms.ToTensor()                              
    ])
    
    # Read the file
    #img = cv2.imread(image)
    # Transform the image
    img = transform(image)
    # Reshape the image. PyTorch model reads 4-dimensional tensor
    # [batch_size, channels, width, height]
    img = img.reshape(1, 3, 448, 448)
    img = img.to(device)
    # We only extract features, so we don't need gradient
    with torch.no_grad():
      # Extract the feature from the image
      feature = new_model(img)
    # Convert to NumPy Array, Reshape it, and save it to features variable
    feature_reshape = feature.cpu().detach().numpy().reshape(-1)

    # Convert to NumPy Array
    features = np.array(feature_reshape)
    print(features.shape)
    return features
    
def plot_images(batch_images, row_size):    
    grid = st.columns(row_size)
    col = 0
    for image in batch_images:
        with grid[col]:
            image_plot = Image.open('./imagenes/images_rescaled/'+image)
            st.image(image_plot)
        col = (col + 1) % row_size
    
    
def find_similar_images(embedding_search, embedding, metrics=('euclidean',), topk=5):
    assert len(metrics) > 0
    assert topk <= 30 #No recomendar tantas imagenes
    n = embedding.shape[0]    

    # --- show retrieved images for each metric
    for metric in metrics:
        print('-------- retrieved with metric = %s -----' % metric)
        distances = pairwise_distances(embedding_search.reshape(1,-1), embedding, metric=metric)
        heap = []
        for i in range(n):            
            if len(heap) < topk:
                heapq.heappush(heap, (-distances[0][i], i))
            else:
                heapq.heappushpop(heap, (-distances[0][i], i))
        heap.sort(reverse=True)
        
        rec_ids = [index2id[i] for _,i in heap]
        plot_images(rec_ids, 5)
        

def main():
    st.title('Demo imagen')
    model = load_model()
    image = load_image()
    result = st.button('Run on image')
    if result:
        embedding_search= preprocess_image(model, image)
        find_similar_images(embedding_search, vgg19_featmat)


if __name__ == '__main__':
    main()

