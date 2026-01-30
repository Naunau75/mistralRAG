# 🧠 Construire un RAG efficace et minimaliste avec Mistral AI et MongoDB Atlas

Salut réseau ! 👋

Aujourd'hui, je vous partage un petit projet sur lequel je me suis amusé : **MissRAG**.

L'objectif ? Construire une application de **RAG (Retrieval-Augmented Generation)** qui permet de "chatter" avec ses propres documents PDF, mais en gardant une approche **simple, robuste et cloud-native**. Et bien sûr, en utilisant les excellents modèles de notre champion français **Mistral AI** ! 🇫🇷

🚀 **Sous le capot, une stack moderne et efficace :**

*   **Intelligence** : Utilisation du modèle `mistral-small-latest` pour la génération et `mistral-embed` pour la vectorisation.
*   **Mémoire** : Fini les bases de données vectorielles locales complexes à gérer. Ici, on utilise **MongoDB Atlas Vector Search** pour stocker les embeddings directement dans le cloud. Scalable par défaut ! ☁️
*   **Orchestration** : **LangChain** pour lier le tout avec fluidité.
*   **Robustesse** : Utilisation de **Pydantic** pour valider la configuration au démarrage (fini les scripts qui plantent au milieu parce qu'une variable d'env manque !).
*   **Performance** : Gestion des dépendances avec **uv**, le remplaçant ultra-rapide de pip.

✨ **La petite fonctionnalité sympa ?**
Le script gère l'**ingestion incrémentale** 🔄.
Mettez 10 PDF dans le dossier, il les indexe. Ajoutez-en un 11ème le lendemain ? Le script détecte le nouveau fichier et n'indexe *que* celui-ci. C'est un gain de temps et d'économie de crédits API. 💸

C'est une base idéale pour ceux qui veulent explorer la recherche sémantique sans se noyer dans une complexité inutile.

Le code est disponible (lien en premier commentaire 👇).

Qui a déjà testé le combo Mistral + Mongo Atlas ici ? Je suis curieux d'avoir vos retours !

#AI #RAG #MistralAI #MongoDB #Python #LangChain #Dev #OpenSource #Innovation
