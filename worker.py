import os
import sqlite3
import time
import uuid
import warnings
import math
import zlib
from typing import Dict
import urllib3
import numpy as np

urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)
warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, QueryRequest
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_text_splitters import RecursiveCharacterTextSplitter


class AntiPlagiarismWorker:
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        sqlite_db_path: str = "showcase.db",
        collection_name: str = "university_docs",
    ):
        self.sqlite_db_path = sqlite_db_path
        self.device = "cpu"
        torch.set_num_threads(1)

        self.encoder = SentenceTransformer("cointegrated/rubert-tiny2", device=self.device)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
        self.ai_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.ai_model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.ai_model.eval()

        self.qdrant = QdrantClient(url=f"http://{qdrant_host}:{qdrant_port}", prefer_grpc=False, timeout=30.0)
        self.collection_name = collection_name

    def _analyze_ai_chunk(self, text: str) -> float:
        token_ids = self.ai_tokenizer.encode(text, return_tensors='pt', max_length=1024, truncation=True).to(
            self.device)
        if token_ids.shape[1] < 2:
            return 0.0

        with torch.no_grad():
            logits = self.ai_model(token_ids).logits[0, :-1, :]
            actual_next_tokens = token_ids[0, 1:].unsqueeze(-1)
            sorted_indices = torch.argsort(logits, dim=-1, descending=True)
            ranks = (sorted_indices == actual_next_tokens).nonzero(as_tuple=True)[1].cpu().numpy() + 1

        total_words = len(ranks)
        if total_words == 0: return 0.0

        # P vec
        p_ratio = np.sum(ranks <= 50) / total_words
        p_score = 1.0 / (1.0 + math.exp(-20.0 * (p_ratio - 0.85)))

        # M vec
        m_ratio = np.sum(ranks > 1000) / total_words
        m_score = 1.0 / (1.0 + math.exp(300.0 * (m_ratio - 0.01)))

        # C vec
        text_bytes = text.encode('utf-8')
        c_ratio = len(zlib.compress(text_bytes))/len(text_bytes)
        c_score = 1.0/(1.0+math.exp(15.0 *(c_ratio-0.45)))*min(1.0,len(text_bytes)/1000.0)

        return math.sqrt(0.60 * p_score + 0.30 * m_score + 0.10 * c_score * 100) / 100

    def process_text(self, text: str, filename: str, verbose: bool = True) -> Dict:
        chunks = self.text_splitter.split_text(text)
        if not chunks:
            return {"plagiarism_percent": 0.0, "ai_percent": 0.0}

        if verbose:
            print(f"      Analysis {filename[:20]}... (batches: {len(chunks)})")

        ai_scores = []
        for chunk in chunks:
            if len(chunk.strip()) > 100:
                ai_scores.append(self._analyze_ai_chunk(chunk))
        ai_percent = round((sum(ai_scores) / len(ai_scores)) * 100, 2) if ai_scores else 0.0
        vectors = self.encoder.encode(chunks, batch_size=16, show_progress_bar=False).tolist()
        requests = [QueryRequest(query=vec, limit=1, score_threshold=0.92) for vec in vectors]
        batch_search_results = self.qdrant.query_batch_points(collection_name=self.collection_name, requests=requests)

        plagiarized_chunks = 0
        points_to_insert = []

        for chunk_text, vector, qdrant_res in zip(chunks, vectors, batch_search_results):
            if len(qdrant_res.points) > 0:
                plagiarized_chunks += 1
            points_to_insert.append(
                PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"filename": filename, "text": chunk_text})
            )

        if points_to_insert:
            self.qdrant.upsert(collection_name=self.collection_name, points=points_to_insert)

        plagiarism_percent = round((plagiarized_chunks / len(chunks)) * 100, 2)

        return {"plagiarism_percent": plagiarism_percent, "ai_percent": ai_percent}

    def _fetch_next_task(self):
        for _ in range(5):
            try:
                with sqlite3.connect(self.sqlite_db_path, timeout=15.0) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE tasks SET status = 'PROCESSING' 
                        WHERE id = (SELECT id FROM tasks WHERE status = 'PENDING' LIMIT 1)
                        RETURNING id, filename, content
                    """)
                    row = cursor.fetchone()
                    conn.commit()
                    return dict(row) if row else None
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower():
                    time.sleep(0.5)
                    continue
                print(f"error in db reading: {e}")
                return None
        return None

    def _save_task_result(self, task_id: int, result: dict):
        for _ in range(5):
            try:
                with sqlite3.connect(self.sqlite_db_path, timeout=15.0) as conn:
                    conn.execute(
                        "UPDATE tasks SET status = 'DONE', plagiarism_percent = ?, ai_percent = ? WHERE id = ?",
                        (result["plagiarism_percent"], result["ai_percent"], task_id))
                    conn.commit()
                    break
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower():
                    time.sleep(0.5)
                    continue

    def run_worker_loop(self, worker_id: int):
        print(f"worker #{worker_id} launched...")
        while True:
            task = self._fetch_next_task()
            if task:
                print(f"[worker #{worker_id}] processing: {task['filename'][:25]}...")
                result = self.process_text(task['content'], task['filename'])
                self._save_task_result(task['id'], result)
                print(
                    f"[worker #{worker_id}] processed: {task['filename'][:20]}... (plag: {result['plagiarism_percent']}%, AI: {result['ai_percent']}%)")
            else:
                time.sleep(2)