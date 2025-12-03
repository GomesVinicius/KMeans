from pydantic import BaseModel
from fastapi import FastAPI
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI()

class Purchase(BaseModel):
    description: str
    value: float

purchases = []
purchases_clustereds = []
category_labels = []
glove_embeddings = {}
embedding_dim = 50

def load_glove_embeddings(glove_file):
    glove_embeddings = {}

    if not os.path.exists(glove_file):
        raise FileNotFoundError(f'Arquivo não encontrado: {glove_file}')
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            glove_embeddings[word] = vector

    print(f'Carregado {len(glove_embeddings)} vetores de palavras')
    
    return glove_embeddings

def generate_embeddings(sentence):
    words = sentence.lower().split()

    valid_vectors = [glove_embeddings[word] for word in words if word in glove_embeddings]
    
    if valid_vectors:
        return np.mean(valid_vectors, axis=0).astype(np.float64)
    else:
        return np.zeros(embedding_dim, dtype=np.float64)

kmeans_model_path = 'kmeans_model.pkl'
kmeans = joblib.load(kmeans_model_path)

glove_file = 'glove.6B.50d.txt'
glove_embeddings = load_glove_embeddings(glove_file)

categories = ['enternaiment', 'investments', 'food']

@app.post('/add_purchase/')
def add_purchase(purchase: Purchase):
    global purchases, category_labels
    description, value = purchase.description, purchase.value
    purchases.append((description, value))
    
    embedding = generate_embeddings(description)
    cluster = kmeans.predict([embedding])[0]

    category = categories[cluster]
    category_labels.append(category)

    purchases_clustereds.append({
        'Description': description,
        'Value': value,
        'Category': category
        })

    return {
        'Status': 'Criado',
        'Description': description,
        'Value': value,
        'Category': category
    }

@app.get('/purchases/')
def get_purchase():
    return purchases_clustereds

@app.get('/purchases_clustereds')
def get_clustered_purchases():
    df = pd.DataFrame(purchases_clustereds)
    print(purchases_clustereds)

    return df.groupby('Category')['Description'].apply(list)
