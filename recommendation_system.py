import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Sistema Basico de Recomendacion

# Ejemplo simple de recomendación basada en similitud de usuarios (filtrado colaborativo)


# Datos de ejemplo: usuarios y sus calificaciones a productos
# Filas: usuarios, Columnas: productos
ratings = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])

def recomendar(usuario_id, ratings, top_n=2):
    # Calcular similitud de coseno entre el usuario dado y los demás

    user_ratings = ratings[usuario_id].reshape(1, -1)
    similarities = cosine_similarity(user_ratings, ratings)[0]
    # No comparar consigo mismo
    similarities[usuario_id] = 0

    # Encontrar usuarios más similares
    similar_users = similarities.argsort()[::-1]

    # Recomendar productos que los usuarios similares han calificado alto y el usuario no ha visto
    recommendations = {}
    for user in similar_users:
        for idx, rating in enumerate(ratings[user]):
            if ratings[usuario_id][idx] == 0 and rating > 3:
                recommendations[idx] = recommendations.get(idx, 0) + rating * similarities[user]

    # Ordenar recomendaciones
    recommended_items = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    return [item[0] for item in recommended_items[:top_n]]

# Ejemplo de uso
usuario = 0
recomendaciones = recomendar(usuario, ratings, top_n=2)
print(f"Recomendaciones para el usuario {usuario}: {recomendaciones}")
