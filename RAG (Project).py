import os
import sys
import logging
import time

# --- Third Party Imports ---
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
RAW_DATA = """
#Sigil
About Sigil
Sigil is an open-source, advanced text encryption utility designed for zero-trust environments. It provides a cryptographic workbench for users who require defense-in-depth security strategies. Unlike standard encryption tools that rely on a single algorithm, Sigil employs a multi-layered, randomized chaining architecture to ensure data remains secure even against sophisticated cryptanalysis.

The application operates entirely offline, performing all cryptographic operations locally on the device. It features a modern, Material 3 interface that balances high-level security with professional usability. As an open-source project, Sigil prioritizes transparency and auditability in its security implementation.

Core Security Features
• Hardware-Backed Vault: Sigil now utilizes the Android Trusted Execution Environment (TEE). Master keys are generated and stored within the hardware enclave, ensuring that your saved encryption passwords never touch the disk in plaintext.
• Multi-Layer Chaining: In Auto Mode, Sigil randomizes the order of encryption algorithms for every message. A typical chain utilizes AES-256-GCM, Twofish-CBC, and Serpent-CBC in a randomized sequence, making traffic analysis significantly more difficult.
• Cryptographic Integrity: Sigil utilizes an Encrypt-then-MAC architecture. A Global HMAC-SHA256 signature is applied to the final container, ensuring that any tampering or corruption is detected and rejected before decryption is attempted.

Advanced User Toolkit
• Secure Keystore: Manage your complex keys securely. You can save, name, and recall encryption keys using the hardware-backed vault, eliminating the need to type long passwords repeatedly while maintaining security.
• Custom Cryptographic Workbench: Users have full manual control to design their own encryption chains. You can add, remove, and reorder layers, selecting from a registry of over 15 industrial and legacy standards.
• Seamless Integration: Sigil acts as a first-class citizen on your device. Share text directly from other apps (like Signal, WhatsApp, or Notes) into Sigil for instant encryption or decryption.
• System Console: A built-in logging interface provides real-time auditing of the encryption process, including timing metrics and detailed error diagnostics.

Technical Specifications
• Secure Memory Architecture: Sigil employs aggressive memory hygiene. Sensitive keys are processed using CharArrays and are immediately wiped (overwritten with zeros) from RAM after use to prevent memory-dump attacks.
• Key Derivation: Keys are derived using Argon2id with high memory hardening (64MB) combined with SHA-512 pre-hashing to resist GPU-based brute-force attacks.
• Key Separation: The application uses HKDF (HMAC-based Key Derivation Function) to generate mathematically distinct keys for every layer and header from the root secret.
• Supported Algorithms: AES (GCM/CBC), Twofish, Serpent, Camellia, CAST6, RC6, SM4, GOST, SEED, Blowfish, IDEA, CAST5, TEA, and XTEA.

Privacy, Permissions & Open Source
Sigil is a zero-knowledge application committed to transparency.
• Open Source: The complete source code is available for public audit at https://github.com/Animesh-Varma/Sigil, ensuring no hidden backdoors or vulnerabilities.
• Offline Only: It does not require internet access, does not track usage analytics, and does not store data on external servers.
• User Controlled Storage: Data exists only during the active session. Passwords are never stored permanently unless you explicitly save them to the Hardware Vault.

#Coeus
Coeus is an advanced NFC toolkit designed for developers, security researchers, and power users who require granular control and insight into Near Field Communication hardware. Built with a zero-trust philosophy and a strict Material 3 interface, Coeus provides direct access to raw tag data often abstracted away by the operating system.

v0.1 - Initial Release
This MVP release establishes the core architecture and introduces the comprehensive Reader Module.

Core Functionality
- Deep Tag Analysis: Automatically detects and parses ISO-DEP (ISO 14443-4), NDEF, Mifare Classic, and Mifare DESFire EVx technologies.
- Raw Data Inspection: View unmasked technical details including UID, ATQA, SAK, Historical Bytes, and Maximum Transceive Lengths in raw Hex format.
- Advanced DESFire Support: Specialized logic to identify DESFire EV1/EV2/EV3 cards, including protocol detection and application listing via factory default authentication sweeps.
- Smart NDEF Parsing: Identifies NDEF message types, writability status, and memory capacity, with automatic decoding of Text and URI records.
- Foreground Dispatch System: The application only engages the NFC hardware when active and focused, preventing interference with system-level payment applications or background services.

User Interface
- Material 3 Design: Built entirely with Jetpack Compose, featuring dynamic color theming and dark mode support.
- Physics-Based Interaction: Custom spring-based animations provide fluid, tactile feedback for list items and navigation.
- Clean Architecture: A focused, utility-driven interface designed for rapid analysis without visual clutter.

Privacy and Security
- Offline Operation: All tag processing occurs locally in system RAM. No tag data is transmitted to external servers.
- Transparency: Open architecture designed for verification.

Roadmap
This is the initial v0.1 Alpha release. While the Reader module is fully functional, advanced modules such as Tag Writing, Relay Mode (ADB Bridge), and the APDU Command Console are currently in development and will be introduced in subsequent updates.

#LOTL
LOTL
About this game
Reawaken India’s Intellectual Flame in Light of the Lost
Embark on a journey through time and thought in Light of the Lost, a narrative-driven educational adventure designed to bridge the gap between modern STEM concepts and India’s rich scientific history. Play as Nova, a seeker of knowledge in a futuristic world, who enters a mystical "Dream World" to recover lost wisdom.

Explore the Dream World
Navigate a stunning 3D hub filled with clouds and portals. Your journey begins in the Dream World, an interactive level selector where you choose your path to enlightenment. From here, transition into the Path, a challenging rite of passage that tests your skills before you can unlock the secrets of history.

Meet the Legends
History comes alive through cutting-edge AI. Engage in real-time, dynamic conversations with legendary figures like Master Brahmagupta in the Sabha (Council). Ask questions, seek guidance, and learn directly from the masters who defined mathematics and astronomy.

Key Features:
- Narrative-Driven Gameplay: Uncover the story of Shunya (Zero) and its origins in Ujjain. Experience a non-linear progression system where every discovery powers your journey.
- Interactive Puzzles: Challenge your logic with mini-games like Vamana’s Ledger, where you must identify accounting errors and solve spacing problems to progress.
- Flight & Exploration: Soar through the skies and navigate obstacle-filled paths to earn Shakti Powers—special abilities unlocked by completing educational milestones.
- Immersive Learning: Move beyond rote memorization. Understand complex concepts like energy conservation, geometric principles, and acoustics through hands-on gameplay and storytelling.
- AI-Powered Dialogue: Powered by advanced Natural Language Processing, our characters respond intelligently to your queries, making every interaction unique.

Why Play?
Built by students for students, Light of the Lost transforms abstract textbook theories into an immersive experience. Whether you are a student, a history buff, or a gamer, prepare to explore, question, and understand the roots of scientific discovery like never before.
"""


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
                chunk_size=500,
                chunk_overlap=50
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