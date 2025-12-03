from k_means import KMeans

data = [
    [1, 2],
    [1, 4],
    [1, 8],
    [100, 100],
    [95, 95],
    [90, 90],
]

kmeans = KMeans(n_clusters=2)
kmeans.fit(data)

print(f'Centroids: {kmeans.centroids}')

predicts = kmeans.predict(data)
print(f'Previsão Data: {predicts}')

new_points = [[0, 0], [101, 101]]
predicts = kmeans.predict(new_points)

print(f'Previsões novos pontos: {predicts}')
