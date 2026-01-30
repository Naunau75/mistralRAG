# 🧠 MissRAG

**MissRAG** est une application de **Retrieval-Augmented Generation (RAG)** minimaliste et puissante, construite avec **Mistral AI**, **LangChain** et **MongoDB Atlas**.

Elle permet de discuter avec vos propres documents PDF. L'application ingère un fichier PDF, le transforme en vecteurs (embeddings), et utilise un modèle de langage Mistral pour répondre à vos questions en se basant sur le contenu du document.

---

## ✨ Fonctionnalités

- **� Multi-PDF & Incrémental** : Charge automatiquement tous les fichiers PDF du dossier `pdf/`. Le script détecte les nouveaux fichiers et ne traite que ceux qui ne sont pas encore dans la base (économie de temps et de crédits API).
- **✂️ Découpage Intelligent** : Utilise `RecursiveCharacterTextSplitter` pour découper le texte en morceaux cohérents.
- **🔢 Embeddings Mistral** : Utilise le modèle `mistral-embed` via l'API officielle Mistral pour vectoriser le texte.
- **☁️ Base Vectorielle Cloud** : Stocke les vecteurs avec **MongoDB Atlas Vector Search** pour une architecture scalable et cloud-native.
- **🤖 Chat IA** : Utilise le LLM `mistral-small-latest` pour générer des réponses précises basées sur le contexte retrouvé.
- **🛡️ Configuration Robuste** : Utilise **Pydantic** pour la validation de la configuration et la gestion des erreurs.

---

## 🛠️ Prérequis

- **Python 3.13** (Recommandé)
- **uv** (Gestionnaire de paquets ultra-rapide)
- Une clé API **Mistral AI** (disponible sur [console.mistral.ai](https://console.mistral.ai/))
- Un cluster **MongoDB Atlas** (le tier gratuit M0 suffit)

---

## 🚀 Installation

1. **Cloner le projet** (si ce n'est pas déjà fait) :
   ```bash
   git clone <votre-repo>
   cd missrag
   ```

2. **Installer les dépendances** :
   Ce projet utilise `uv` pour la gestion des dépendances.
   ```bash
   uv sync
   ```
   *Ou manuellement si vous n'avez pas le fichier lock :*
   ```bash
   uv add langchain langchain-mistralai langchain-mongodb pymongo langchain-text-splitters pydantic python-dotenv pypdf langchain-community
   ```

---

## ⚙️ Configuration

1. **Créer le fichier `.env`** à la racine du projet :
   ```bash
   touch .env
   ```

2. **Ajouter votre clé API Mistral** dans le fichier `.env` :
   ```properties
   MISTRAL_API_KEY=votre_cle_api_mistral...
   MONGODB_ATLAS_Cluster_URI=mongodb+srv://user:pass@cluster.mongodb.net/?...
   ```

3. **Créer l'Index Vectoriel sur Atlas** (⚠️ Étape Obligatoire) :
   - Connectez-vous à votre cluster MongoDB Atlas.
   - Allez dans l'onglet **"Atlas Search"** (ou "Vector Search").
   - Créez un nouvel index nommé `vector_index`.
   - Choisissez "JSON Editor" et collez la configuration suivante :
     ```json
     {
       "fields": [
         {
           "numDimensions": 1024,
           "path": "embedding",
           "similarity": "cosine",
           "type": "vector"
         }
       ]
     }
     ```
   - *Note : Sélectionnez bien la database `missrag_db` et la collection `rag_collection`.*

**Note sur le Reset :**
Si vous souhaitez vider la base de données pour repartir de zéro, vous pouvez passer l'option `reset_db=True` dans la configuration `RagConfig` (dans `main.py`).

---

## 🏃‍♂️ Utilisation

1. **Ajouter des documents** :
   Placez vos fichiers PDF (par exemple `these.pdf`, `guide.pdf`) dans le dossier `pdf/`.
   L'application scannera ce dossier et n'ajoutera que les fichiers qui ne sont pas encore indexés.

2. **Lancer l'application** :
   ```bash
   uv run main.py
   ```
   *(Assurez-vous que votre environnement virtuel est activé si vous passez par la commande python main.py)*

3. **Poser des questions** :
   Le script exécutera des questions de test définies à la fin du fichier `main.py`. Vous pouvez modifier ces appels `ask("Votre question ?")` directement dans le code pour interroger votre base de connaissances.

---

## 📂 Structure du Projet

```text
missrag/
├── .env                # Variables d'environnement (Clé API)
├── .python-version     # Version Python fixée (3.13)
├── main.py             # Code principal de l'application
├── pdf/                # Dossier contenant vos documents sources
│   └── document.pdf
├── pyproject.toml      # Configuration du projet et dépendances
└── README.md           # Documentation
```

## 🛠️ Stack Technique

- **Langage** : Python 3.13
- **Orchestration** : [LangChain](https://www.langchain.com/)
- **LLM & Embeddings** : [Mistral AI](https://mistral.ai/)
- **Vector Store** : [MongoDB Atlas Vector Search](https://www.mongodb.com/products/platform/atlas-vector-search)
- **Validation** : [Pydantic](https://docs.pydantic.dev/)
