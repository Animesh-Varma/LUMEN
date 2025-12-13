import os
import sys
import logging
import time
import argparse
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# 1. CONFIGURATION & LOGGING
# ==========================================
parser = argparse.ArgumentParser(description="LUMEN - School Event AI")
parser.add_argument("--verbose", action="store_true", help="Show detailed prompts and generation stats")
args, unknown = parser.parse_known_args()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

if not args.verbose:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

# --- SYSTEM CONSTANTS ---
LLM_MODEL = "qwen2.5:3b"
EMBED_MODEL = "snowflake-arctic-embed:110m"
DB_PATH = "./db_lumen_store"
DATA_DIRECTORY = "./TXT"


# ==========================================
# 2. DATASET LOADER
# ==========================================
def load_documents_from_folder(folder_path):
    """Scans a specific folder and combines all .txt files into one string."""
    combined_text = ""
    file_count = 0

    if not os.path.exists(folder_path):
        logger.error(f"CRITICAL: Directory '{folder_path}' not found.")
        logger.info(f"Please create the folder '{folder_path}' and put your text files inside.")
        sys.exit(1)

    logger.info(f"Scanning '{folder_path}' for data...")

    # Iterate through all files in the directory
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Add a header so the AI knows which file this data came from
                    combined_text += f"\n\n--- SOURCE: {filename} ---\n{content}"
                    file_count += 1
                    logger.info(f"Loaded: {filename} ({len(content)} chars)")
            except Exception as e:
                logger.warning(f"Skipped {filename}: {e}")

    if file_count == 0:
        logger.error("No .txt files found in the data directory!")
        sys.exit(1)

    logger.info(f"Total Knowledge Base: {len(combined_text)} characters from {file_count} files.")
    return combined_text


# Load data immediately
RAW_DATA = load_documents_from_folder(DATA_DIRECTORY)


# ==========================================
# 3. CLASS: LUMEN RAG PIPELINE
# ==========================================
class LumenAssistant:
    def __init__(self, verbose_mode=False):
        self.verbose = verbose_mode
        self.llm = None
        self.vector_store = None
        self.chat_history = []
        self._initialize_system()

    def _initialize_system(self):
        """Sets up the Database and LLM connection."""
        start_time = time.time()
        logger.info("Initializing LUMEN System...")

        # 1. Setup LLM
        try:
            logger.info(f"Connecting to LLM: {LLM_MODEL}")
            self.llm = OllamaLLM(
                model=LLM_MODEL,
                temperature=0.1,
                keep_alive="-1m"
            )
        except Exception as error:
            logger.error(f"LLM Connection Failed: {error}")
            sys.exit(1)

        # 2. Setup Embeddings
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
            logger.info("Indexing new data (Creating Vector Store)...")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=900,
                chunk_overlap=200
            )
            docs = text_splitter.create_documents([RAW_DATA])

            self.vector_store = Chroma.from_documents(
                docs,
                embeddings,
                persist_directory=DB_PATH
            )
            logger.info(f"Database created with {len(docs)} chunks.")

        logger.info("Warming up neural network...")
        try:
            self.llm.invoke("Ready")
        except:
            pass

        elapsed = time.time() - start_time
        logger.info(f"LUMEN Ready in {elapsed:.2f} seconds.")

    def generate_answer(self, user_question):
        if not user_question.strip():
            return "Please ask a question."

        # --- STEP 1: RETRIEVAL ---
        logger.info("Searching knowledge base...")
        if self.verbose:
            print(f"\n[VERBOSE] Querying Vector DB for: '{user_question}'")

        db_start = time.time()

        # CHANGED: Switched from similarity_search to max_marginal_relevance_search
        # k=5: Return 5 final chunks
        # fetch_k=20: Look at 20 chunks initially, then pick the 5 most UNIQUE ones
        # lambda_mult=0.5: Balance between "exact match" and "diversity"
        relevant_docs = self.vector_store.max_marginal_relevance_search(
            user_question,
            k=2,
            fetch_k=20,
            lambda_mult=0.5
        )

        db_end = time.time()

        db_duration = db_end - db_start
        context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
        context_token_count = int(len(context_text) / 4)

        # --- MEMORY ---
        history_context = ""
        #if self.chat_history:
        #    history_context = "Previous Conversation:\n"
        #    recent_history = self.chat_history[-3:]
        #    for turn in recent_history:
        #        history_context += f"User: {turn['user']}\nLUMEN: {turn['bot']}\n"

        # --- STEP 2: AUGMENTATION (PROMPT) ---
        prompt = f"""
        You are LUMEN (Lotus valley Unified Model for Engagement and Navigation), the AI Assistant for the 'Dhanak' annual function at Lotus Valley School, Mandsaur.

        Your Tasks:
        1. Help visitors find student performances, awards, and project details.
        2. Provide school information (location, contacts) if asked.
        3. Be polite, concise, and accurate.

        Instructions:
        - Answer using ONLY the Context Data below.
        - If the info is missing, say "I don't have that information in my records."
        - If listing students, use bullet points.

        Context Data:
        {context_text}

        Visitor Question: {user_question}

        Answer:
        """

        if self.verbose:
            print("\n" + "=" * 20 + " [VERBOSE] FULL PROMPT " + "=" * 20)
            # Print first 2000 chars so you can verify if General Info is there
            print(prompt[:2000] + "... [truncated]")
            print("=" * 70 + "\n")

        # --- STEP 3: GENERATION ---
        logger.info("Generating response...")
        start_time = time.time()
        first_token_time = None
        char_count = 0

        try:
            full_response_accumulator = ""
            for chunk in self.llm.stream(prompt):
                text_chunk = str(chunk)
                if first_token_time is None and text_chunk.strip():
                    first_token_time = time.time()
                char_count += len(text_chunk)
                full_response_accumulator += text_chunk
                yield text_chunk

            # Update Memory
            # self.chat_history.append({"user": user_question, "bot": full_response_accumulator})

            if self.verbose:
                end_time = time.time()
                total_duration = end_time - start_time
                ttft = (first_token_time - start_time) if first_token_time else 0
                est_tokens = int(char_count / 4)
                tps = est_tokens / total_duration if total_duration > 0 else 0

                print("\n\n" + "-" * 20 + " PERFORMANCE METRICS " + "-" * 20)
                print(f"[DB LOOKUP (MMR)]")
                print(f"• Time:           {db_duration:.4f}s")
                print(f"• Chunks:         {len(relevant_docs)} retrieved")
                print(f"\n[AI GENERATION]")
                print(f"• Total Time:     {total_duration:.2f}s")
                print(f"• Latency (TTFT): {ttft:.2f}s")
                print(f"• Speed:          {tps:.2f} tokens/sec")
                print("-" * 60)

        except Exception as error:
            logger.error(f"Generation failed: {error}")
            yield "I encountered an error accessing my neural network."


# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    bot = LumenAssistant(verbose_mode=args.verbose)

    print("\n" + "=" * 60)
    print("LUMEN ONLINE (Lotus Valley School AI)")
    if args.verbose:
        print("Status: VERBOSE MODE ENABLED")
    print("Type 'exit' to quit.")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("Visitor >> ")

            if query.lower() in ["exit", "quit", "q"]:
                logger.info("LUMEN signing off.")
                break

            print("\nLUMEN >> ", end="", flush=True)

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
# - Finetune