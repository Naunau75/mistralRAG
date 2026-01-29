import os
from typing import List
import shutil
import glob
from langchain_community.document_loaders import PyPDFLoader
# Chargement des variables d'environnement
from dotenv import load_dotenv
load_dotenv()

# Pydantic pour la validation des données et de la config
from pydantic import BaseModel, Field, SecretStr

# LangChain pour l'orchestration
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. CONFIGURATION AVEC PYDANTIC ---
# L'intérêt : Si tu oublies la clé API, le script s'arrête immédiatement avec une erreur claire.
class RagConfig(BaseModel):
    reset_db: bool = Field(False, description="Reset la base de données")
    mistral_api_key: str = Field(..., description="Clé API Mistral")
    model_name: str = Field("mistral-small-latest", description="Modèle pour le chat")
    embedding_model: str = Field("mistral-embed", description="Modèle pour les vecteurs")
    chunk_size: int = Field(500, description="Taille des morceaux de texte")
    chunk_overlap: int = Field(50, description="Chevauchement entre les morceaux")
    persist_directory: str = Field("./chroma_db", description="Dossier de sauvegarde BDD")

try:
    config = RagConfig(
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        reset_db=False # Mettre à True pour forcer la recréation de la BDD, puis remettre à False !
    )
except Exception as e:
    print(f"❌ Erreur de configuration : {e}")
    exit()

# --- 2. VECTOR STORAGE & EMBEDDINGS ---

# On instancie l'objet d'embedding Mistral
embeddings = MistralAIEmbeddings(
    api_key=SecretStr(config.mistral_api_key),
    model=config.embedding_model
)

# Gestion du RESET de la base
if config.reset_db and os.path.exists(config.persist_directory):
    print(f"🗑️ Option reset_db activée : Suppression de '{config.persist_directory}'...")
    shutil.rmtree(config.persist_directory)

# Initialisation du VectorStore (il se crée s'il n'existe pas, ou se charge s'il existe)
vectorstore = Chroma(
    persist_directory=config.persist_directory,
    embedding_function=embeddings
)

# --- GESTION INCREMENTALE DES PDFS ---
print("�️  Vérification des documents existants...")

# On récupère tous les fichiers PDF du dossier
pdf_folder = "./pdf"
pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))

if not pdf_files:
    print(f"⚠️ Aucun fichier PDF trouvé dans {pdf_folder}")
else:
    # On regarde ce qu'il y a déjà dans la base
    # vectorstore.get() renvoie un dict avec 'ids', 'embeddings', 'metadatas', 'documents'
    existing_data = vectorstore.get()
    existing_sources = set()
    
    # On extrait les noms de fichiers des métadonnées stockées
    if existing_data["metadatas"]:
        for metadata in existing_data["metadatas"]:
            # LangChain stocke le chemin complet dans 'source'
            if metadata and "source" in metadata:
                existing_sources.add(metadata["source"])
    
    print(f"📚 {len(existing_sources)} fichier(s) déjà indexé(s) dans la base.")

    # On identifie les nouveaux à ajouter
    new_files = []
    for pdf_path in pdf_files:
        # On normalise le chemin pour être sûr de la comparaison (relatif vs absolu)
        # Note: LangChain stocke souvent le chemin tel qu'il est passé au loader.
        # Pour être robuste, on compare juste le nom du fichier s'il y a un doute, 
        # mais ici on va comparer les chemins tels que scannés.
        if pdf_path not in existing_sources:
             # Petite subtilité: parfois le chemin est absolu stocké, parfois relatif.
             # On vérifie si l'un des existing_sources termine par notre nom de fichier
             is_present = False
             filename = os.path.basename(pdf_path)
             for source in existing_sources:
                 if source.endswith(filename):
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
                print(f"   ↳ {len(docs)} chunks générés. Indexation...")
                
                # Ajout incrémental à la base
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

# LA CHAÎNE MAGIQUE (LCEL)
# 1. On prend la question
# 2. En parallèle : on cherche le contexte (retriever) ET on garde la question (Passthrough)
# 3. On envoie tout au prompt
# 4. On envoie au LLM
# 5. On parse la sortie en string
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
