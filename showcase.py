import os
import glob
import sqlite3
import time
import multiprocessing
import warnings
from docx import Document
from worker import AntiPlagiarismWorker
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
DB_PATH = "showcase.db"
WAVE1_DIR = os.path.join("test_docs", "wave1")
WAVE2_DIR = os.path.join("test_docs", "wave2")


def setup_env():
    print("setup db...")
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("""
            CREATE TABLE tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                content TEXT,
                plagiarism_percent REAL,
                ai_percent REAL,
                status TEXT DEFAULT 'PENDING'
            )
        """)

    os.makedirs(WAVE1_DIR, exist_ok=True)
    os.makedirs(WAVE2_DIR, exist_ok=True)

    print("caching models...")
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForCausalLM
    encoder = SentenceTransformer("cointegrated/rubert-tiny2", device="cpu")
    vector_size = encoder.get_sentence_embedding_dimension()  # 312

    AutoTokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
    AutoModelForCausalLM.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")

    print("setup qdrant...")
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        VectorParams, Distance, ScalarQuantization, ScalarQuantizationConfig, ScalarType
    )

    qdrant = QdrantClient(url="http://localhost:6333", prefer_grpc=False, timeout=10.0)
    collection_name = "university_docs"

    if not qdrant.collection_exists(collection_name):
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(type=ScalarType.INT8, quantile=0.99, always_ram=True)
            )
        )
        print(f"qdrant base '{collection_name}' created INT8, vec size: {vector_size})")
    else:
        print(f"qdrant base '{collection_name}' already exist")


def extract_text(filepath: str) -> str:
    doc = Document(filepath)
    return "\n".join([p.text for p in doc.paragraphs])


def insert_file_to_db(filepath: str):
    filename = os.path.basename(filepath)
    text = extract_text(filepath)
    with sqlite3.connect(DB_PATH, timeout=10.0) as conn:
        conn.execute("INSERT INTO tasks (filename, content) VALUES (?, ?)", (filename, text))


def start_worker(worker_id: int):
    worker = AntiPlagiarismWorker(sqlite_db_path=DB_PATH)
    worker.run_worker_loop(worker_id)


def wait_for_all_tasks_done():
    while True:
        with sqlite3.connect(DB_PATH, timeout=10.0) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM tasks WHERE status IN ('PENDING', 'PROCESSING')")
            if cursor.fetchone()[0] == 0:
                break
        time.sleep(1)


def print_results():
    print("\nTOTAL:")
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, filename, plagiarism_percent, ai_percent, status FROM tasks ORDER BY id")
        for row in cursor.fetchall():
            print(
                f"file: {row[1][:25]:<25} status: {row[4]:<8} plagiarism: {row[2]:>5}% AI: {row[3]:>5}%")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    setup_env()

    wave1_files = glob.glob(os.path.join(WAVE1_DIR, "*.docx"))
    wave2_files = glob.glob(os.path.join(WAVE2_DIR, "*.docx"))

    if not wave1_files or not wave2_files:
        print("\ndrop docx files in folders wave1 and wave2.")
        exit()

    print(f"found {len(wave1_files)} files in first wave adding to SQLite...")
    for filepath in wave1_files:
        insert_file_to_db(filepath)

    print("setup workers...")
    p1 = multiprocessing.Process(target=start_worker, args=(1,))
    p2 = multiprocessing.Process(target=start_worker, args=(2,))
    p1.start()
    p2.start()

    wait_for_all_tasks_done()
    print("\nfirst wave processed.")

    print(f"\nfound {len(wave2_files)} files in second wave adding to SQLite...")
    for filepath in wave2_files:
        insert_file_to_db(filepath)

    wait_for_all_tasks_done()
    print("\nsecond wave processed")
    print_results()

    p1.terminate()
    p2.terminate()
    print("showcase finished")