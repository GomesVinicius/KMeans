# 📌 Purchase Categorization API

Esta aplicação FastAPI recebe descrições de compras, gera  **embeddings GloVe** , utiliza um modelo **KMeans pré-treinado** para identificar o cluster correspondente e retorna a **categoria** da compra.

As categorias disponíveis são:

* `enternaiment`
* `investments`
* `food`

## 📂 Estrutura dos arquivos necessários

Antes de rodar a API, garanta que você possui:

* `kmeans_model.pkl` → modelo KMeans treinado
* `glove.6B.50d.txt` → arquivo de embeddings GloVe (50 dimensões)
* Baixar no link: #https://nlp.stanford.edu/data/glove.6B.zip

Ambos devem estar na mesma pasta do script.

## ▶️ Como executar

1. Instale as dependências:

``pip install fastapi uvicorn pandas numpy joblib``

2. Inicie o servidor:

``uvicorn main:app --reload``

3. Acesse a base API:

*http://127.0.0.1:8000/*

---



## 📥 Endpoint: Adicionar Compra

### **POST /add_purchase/**

Envia uma compra para categorização.

### **Body JSON**

```
{
 "description": "i eat apple",
"value": 1.0
}
```

### **Resposta**

```
{
    "Status": "Criado" ,
    "Description": "i eat apple",
    "Value":1.0,
    "Category":"food"
}
```

## 📤 Endpoint: Listar Compras

### **GET /purchases/**

Retorna todas as compras já cadastradas com suas categorias.

## 📊 Endpoint: Agrupar Compras por Categoria

### **GET /purchases_clustereds**

Retorna:

* Um DataFrame agrupado por categoria
* Lista das descrições em cada grupo

Exemplo:

{
    "enternaiment":["i go to music"],
    "food":["i eat apple", "ice cream"],
    "investments":["i invest my payroll"]
}

## 🧠 Como a categorização funciona

1. A descrição é convertida para tokens (palavras).
2. Cada palavra é transformada no seu vetor GloVe correspondente.
3. O embedding final é a média dos vetores válidos.
4. O KMeans prediz o cluster do embedding.
5. O cluster é mapeado para uma categoria na lista `categories`.


## 🔧 Possíveis Melhorias no Algoritmo

Este projeto funciona como uma prova de conceito, mas a qualidade das previsões pode ser significativamente aprimorada com algumas melhorias estruturais importantes:

### **1. Base de treinamento maior e mais variada**

O modelo KMeans depende diretamente da qualidade e diversidade dos exemplos usados no treino.

Uma base pequena ou pouco representativa reduz a precisão da categorização.

**Quanto mais frases de exemplo forem incluídas — variadas, reais e balanceadas — melhor o modelo aprende os padrões das categorias.**

### **2. Usar embeddings com mais dimensões**

Atualmente, o projeto utiliza vetores GloVe de  **50 dimensões** , o que funciona, mas limita a capacidade do modelo de capturar nuances semânticas das frases.

Trocar para embeddings maiores, como  **100d, 200d ou 300d** , pode melhorar substancialmente a precisão, pois:

* capturam mais informações semânticas;
* representam melhor diferenças sutis entre palavras;
* fornecem embeddings mais ricos para o KMeans trabalhar.
