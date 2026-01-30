import os
from typing import List
import glob
from langchain_community.document_loaders import PyPDFLoader
# Chargement des variables d'environnement
from dotenv import load_dotenv
load_dotenv()

# Pydantic pour la validation des données et de la config
from pydantic import BaseModel, Field, SecretStr

# LangChain pour l'orchestration
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. CONFIGURATION AVEC PYDANTIC ---
# L'intérêt : Si tu oublies la clé API ou l'URI Mongo, le script s'arrête immédiatement avec une erreur claire.
class RagConfig(BaseModel):
    reset_db: bool = Field(False, description="Reset la base de données")
    mistral_api_key: str = Field(..., description="Clé API Mistral")
    mongodb_uri: str = Field(..., description="URI de connexion MongoDB Atlas")
    db_name: str = Field("missrag_db", description="Nom de la base de données")
    collection_name: str = Field("rag_collection", description="Nom de la collection")
    index_name: str = Field("vector_index", description="Nom de l'index Vector Search sur Atlas")
    model_name: str = Field("mistral-small-latest", description="Modèle pour le chat")
    embedding_model: str = Field("mistral-embed", description="Modèle pour les vecteurs")
    chunk_size: int = Field(500, description="Taille des morceaux de texte")
    chunk_overlap: int = Field(50, description="Chevauchement entre les morceaux")

try:
    config = RagConfig(
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        mongodb_uri=os.getenv("MONGODB_ATLAS_Cluster_URI"), # Assure-toi que cette variable est dans ton .env
        reset_db=False 
    )
except Exception as e:
    print(f"❌ Erreur de configuration : {e}")
    print("💡 Astuce : Vérifie que MISTRAL_API_KEY et MONGODB_ATLAS_Cluster_URI sont bien dans ton fichier .env")
    exit()

# --- 2. VECTOR STORAGE & EMBEDDINGS ---
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch

# Connexion à MongoDB
try:
    client = MongoClient(config.mongodb_uri)
    # Test de connexion
    client.admin.command('ping')
    print("✅ Connexion à MongoDB Atlas réussie !")
except Exception as e:
    print(f"❌ Impossible de se connecter à MongoDB : {e}")
    exit()

collection = client[config.db_name][config.collection_name]

# On instancie l'objet d'embedding Mistral
embeddings = MistralAIEmbeddings(
    api_key=SecretStr(config.mistral_api_key),
    model=config.embedding_model
)

# Gestion du RESET de la base
if config.reset_db:
    print(f"🗑️ Option reset_db activée : Suppression de tous les documents dans '{config.collection_name}'...")
    collection.delete_many({})

# Initialisation du VectorStore
# NOTE : Tu DOIS avoir créé un index de recherche vectorielle sur Atlas pour que cela fonctionne !
# Nom de l'index : 'vector_index' (par défaut)
# Dimensions : 1024 (pour mistral-embed)
vectorstore = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name=config.index_name,
    relevance_score_fn="cosine",
)

# --- GESTION INCREMENTALE DES PDFS ---
print("🕵️  Vérification des documents existants...")

# On récupère tous les fichiers PDF du dossier
pdf_folder = "./pdf"
pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))

if not pdf_files:
    print(f"⚠️ Aucun fichier PDF trouvé dans {pdf_folder}")
else:
    # On regarde ce qu'il y a déjà dans la base Mongo
    # On récupère tous les chemins 'source' distincts dans les métadonnées
    existing_sources = set(collection.distinct("source"))
    
    print(f"📚 {len(existing_sources)} fichier(s) déjà indexé(s) dans la base.")

    # On identifie les nouveaux à ajouter
    new_files = []
    for pdf_path in pdf_files:
        # On normalise le chemin ou on compare les noms de fichiers pour être robuste
        is_present = False
        filename = os.path.basename(pdf_path)
        
        # Vérification simple : est-ce que le chemin exact ou le nom de fichier est dans les sources ?
        for source in existing_sources:
            if source == pdf_path or source.endswith(filename):
                is_present = True
                break
        
        if not is_present:
            new_files.append(pdf_path)
        else:
            print(f"⏩ Déjà indexé : {pdf_path}")

    if not new_files:
        print("✅ Tous les fichiers sont déjà à jour.")
    else:
        print(f"🚀 {len(new_files)} nouveau(x) fichier(s) détecté(s). Traitement...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )

        for pdf_path in new_files:
            print(f"📄 Traitement de : {pdf_path}")
            try:
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()
                print(f"   ↳ {len(pages)} pages chargées.")
                
                docs = text_splitter.split_documents(pages)
                print(f"   ↳ {len(docs)} chunks générés. Indexation sur Atlas...")
                
                # Ajout incrémental à la base (Chaque doc a 'source' dans metadata via PyPDFLoader)
                vectorstore.add_documents(docs)
                print("   ✅ Ajouté avec succès.")
                
            except Exception as e:
                print(f"❌ Erreur lors du traitement de {pdf_path}: {e}")

# On transforme la base en "Retriever"
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


# --- 4. LE PIPELINE RAG (LCEL - LangChain Expression Language) ---
print("🔗 Construction du pipeline...")

# Le modèle de Chat
llm = ChatMistralAI(
    api_key=SecretStr(config.mistral_api_key),
    model=config.model_name,
    temperature=0
)

# Le Prompt Template
template = """Réponds à la question uniquement basé sur le contexte suivant :
{context}

Question : {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Définition de la fonction pour formater les docs récupérés (les coller ensemble)
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 5. EXECUTION ---

def ask(question: str):
    print(f"\n❓ Question : {question}")
    # invoke lance toute la chaîne définie au-dessus
    response = rag_chain.invoke(question)
    print(f"🤖 Réponse : {response}")

# Tests
ask("Dis moi ce que tu connais sur le village de Soubès")
