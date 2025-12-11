import os
import sys
import logging
import time
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# 1. CONFIGURATION & LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    stream = sys.stdout
)
logger = logging.getLogger(__name__)

# System Constants
LLM_MODEL = "gemma3n:e4b"
EMBED_MODEL = "embeddinggemma"
DB_PATH = "./db_sigil_store"

# ==========================================
# 2. DATASET
# ==========================================
def load_knowledge_base():
    file_path = "projects.txt"
    try:
        if not os.path.exists(file_path):
            logger.error(f"CRITICAL: '{file_path}' not found in {os.getcwd()}")
            logger.info("Please create 'projects.txt' and paste your data there.")
            sys.exit(1)

        with open(file_path, "r", encoding="utf-8") as f:
            data = f.read()
            logger.info(f"Loaded {len(data)} characters from {file_path}")
            return data

    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        sys.exit(1)
RAW_DATA = load_knowledge_base()


# ==========================================
# 3. CLASS: MANUAL RAG PIPELINE
# ==========================================
class Assistant:
    def __init__(self):
        self.llm = None
        self.vector_store = None
        # Placeholder list for conversation memory
        self.chat_history = []
        self._initialize_system()

    def _initialize_system(self):
        """Sets up the Database and LLM connection."""
        start_time = time.time()
        logger.info("Initializing Stall Assistant System...")

        # 1. Setup LLM
        try:
            logger.info(f"Connecting to LLM: {LLM_MODEL}")
            self.llm = OllamaLLM(model=LLM_MODEL, temperature=0.1)
        except Exception as error:
            logger.error(f"LLM Connection Failed: {error}")
            sys.exit(1)

        # 2. Setup Embeddings & Vector Database
        logger.info(f"Loading Embeddings: {EMBED_MODEL}")
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)

        # 3. Process Data
        if os.path.exists(DB_PATH):
            logger.info(f"Loading existing database from {DB_PATH}")
            self.vector_store = Chroma(
                persist_directory=DB_PATH,
                embedding_function=embeddings
            )
        else:
            logger.info("Creating new vector index from raw data...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200
            )
            docs = text_splitter.create_documents([RAW_DATA])
            self.vector_store = Chroma.from_documents(
                docs,
                embeddings,
                persist_directory=DB_PATH
            )
            logger.info("Database creation complete.")

        elapsed = time.time() - start_time
        logger.info(f"System ready in {elapsed:.2f} seconds.")

    def generate_answer(self, user_question):
        if not user_question.strip():
            return "Please ask a question."

        # --- STEP 1: RETRIEVAL ---
        logger.info("Retrieving relevant context...")
        relevant_docs = self.vector_store.similarity_search(user_question, k=4)
        context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

        # =========================================================
        # --- MEMORY / CONTEXT ---
        # =========================================================

        history_context = ""
        if self.chat_history:
            history_context = "Previous Conversation:\n"
            recent_history = self.chat_history[-3:]
            for turn in recent_history:
                history_context += f"Visitor: {turn['user']}\nAssistant: {turn['bot']}\n"
        # =========================================================

        # --- STEP 2: AUGMENTATION ---
        # Add {history_context}

        prompt = f"""
        You are a professional technical assistant at a software stall for Lotus Valley school's annual function named Dhanak.
        You are showcasing: Sigil (Encryption), Coeus (NFC Tool), and LOTL (Game).
        Use the user's preferred language for responses.

        Instructions:
        1. Answer based ONLY on the Context provided below.
        2. If the answer is not in the Context, say "I don't have that specific information."
        3. Be enthusiastic but professional.

        Context Data:
        {context_text}
        
        Conversation history:
        {history_context}

        User Question: {user_question}

        Answer:
        """

        # --- STEP 3: GENERATION ---
        logger.info("Generating response...")
        try:
            full_response_accumulator = ""
            for chunk in self.llm.stream(prompt):
                full_response_accumulator += chunk
                yield chunk

            # =========================================================
            # --- SECTION: UPDATE MEMORY ---
            # =========================================================
            self.chat_history.append({"user": user_question, "bot": full_response_accumulator})
            # =========================================================

        except Exception as error:
            logger.error(f"Generation failed: {error}")
            return "I encountered an error generating the response."


# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    bot = Assistant()

    print("\n" + "=" * 60)
    print("ASSISTANT ONLINE (PROJECT PIPELINE)")
    print("Type 'exit' to quit.")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("Visitor >> ")

            if query.lower() in ["exit", "quit", "q"]:
                logger.info("Session closed by user.")
                break

            response_text = bot.generate_answer(query)

            print("\nAssistant >> ", end="", flush=True)
            for chunk in bot.generate_answer(query):
                print(chunk, end="", flush=True)
            print("\n")
            print("-" * 60)
        except KeyboardInterrupt:
            logger.info("Force shutdown.")
            break

# TODO:
# - Add GUI
# - Benchtest
# - Add README
# - Add details to projects.txt