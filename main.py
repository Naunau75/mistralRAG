import os
from typing import List

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
    mistral_api_key: str = Field(..., description="Clé API Mistral")
    model_name: str = Field("mistral-small-latest", description="Modèle pour le chat")
    embedding_model: str = Field("mistral-embed", description="Modèle pour les vecteurs")
    chunk_size: int = Field(500, description="Taille des morceaux de texte")
    chunk_overlap: int = Field(50, description="Chevauchement entre les morceaux")
    persist_directory: str = Field("./chroma_db", description="Dossier de sauvegarde BDD")

try:
    config = RagConfig(
        mistral_api_key=os.getenv("MISTRAL_API_KEY")
    )
except Exception as e:
    print(f"❌ Erreur de configuration : {e}")
    exit()

# --- 2. PRÉPARATION DU TEXTE (LANGCHAIN) ---
from langchain_community.document_loaders import PyPDFLoader
import glob

print("📂 Chargement du PDF...")

# Trouver le fichier PDF dans le dossier 'pdf'
pdf_folder = "./pdf"
pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))

if not pdf_files:
    print(f"❌ Aucun fichier PDF trouvé dans {pdf_folder}")
    exit()

pdf_path = pdf_files[0] # On prend le premier PDF trouvé
print(f"📄 Lecture du fichier : {pdf_path}")

loader = PyPDFLoader(pdf_path)
pages = loader.load()
print(f"✅ {len(pages)} pages chargées.")

print("✂️ Découpage du texte...")
# LangChain gère le découpage intelligemment (ne coupe pas les mots/phrases si possible)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.chunk_size,
    chunk_overlap=config.chunk_overlap
)

# On splitte les pages chargées (qui sont déjà des Documents)
docs = text_splitter.split_documents(pages)
print(f"🧩 Nombre de chunks créés : {len(docs)}")


# --- 3. CRÉATION DU VECTOR STORE (LANGCHAIN + CHROMA) ---
print("💾 Indexation dans ChromaDB avec Mistral Embeddings...")

# On instancie l'objet d'embedding Mistral via LangChain
embeddings = MistralAIEmbeddings(
    api_key=SecretStr(config.mistral_api_key),
    model=config.embedding_model
)

# LangChain s'occupe d'appeler l'API Mistral, vectoriser et stocker dans Chroma
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=config.persist_directory
)

# On transforme la base en "Retriever" (outil de recherche)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # k=2 : on veut les 2 meilleurs morceaux


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
ask("Qui sont les fondateurs de Mistral ?")
ask("A quoi sert Pydantic ?")
ask("Quelle est la hauteur de la Tour Eiffel ?") # Doit dire qu'il ne sait pas ou répondre avec ses connaissances générales si le prompt n'est pas strict.
