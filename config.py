# 📁 디렉터리 및 모듈 구성도
# 프로젝트 구조
#
# airbnb_chatbot/
# ├── app.py                           ← ✅ Streamlit UI 실행 진입점
# ├── config.py                        ← ✅ 임베딩/데이터/모델 로드, 공통 전역 변수
# ├── chat_interface.py                ← ✅ Streamlit용 메인 응답 함수
# ├── utils/
# │   ├── __init__.py
# │   ├── map_utils.py                 ← create_map_from_results
# │   └── response_utils.py            ← generate_free_response 등
# ├── agent/
# │   ├── __init__.py
# │   ├── check_rag.py                 ← agent_check_rag_need
# │   ├── expand_query.py              ← agent_expand_and_extract_columns
# │   ├── extract_user_req.py          ← extract_user_requirements
# │   └── extract_keywords.py          ← extract_user_keywords (분리 시)
# ├── search/
# │   ├── __init__.py
# │   ├── smart_search.py              ← smart_rag_search
# │   └── filter_results.py            ← filter_results_by_requirements
# └── response/
#     ├── __init__.py
#     └── generate_response.py         ← generate_context_aware_response

# ============================
# config.py
# ============================
import pickle
import faiss
import numpy as np
from collections import deque
from sentence_transformers import SentenceTransformer
import pandas as pd
from openai import OpenAI

# --- OpenAI API Client ---
client = OpenAI(api_key="")

# --- Embedding Model ---
embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# --- Airbnb DB Load ---
with open("dataset/airbnb_data_준회.pkl", "rb") as f:
    data = pickle.load(f)
    df = data["df"]
    vectors = data["vectors"].astype("float32")

with open("dataset/airbnb_data_price_준회.pkl", "rb") as f:
    price_data = pickle.load(f)
    price_df = price_data["df"]
    price_vectors = price_data["vectors"].astype("float32")

# --- FAISS Index ---
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

price_index = faiss.IndexFlatL2(price_vectors.shape[1])
price_index.add(price_vectors)

# --- Chat History ---
MAX_HISTORY_LENGTH = 10
chat_history = deque(maxlen=MAX_HISTORY_LENGTH)

# --- Expose Globals ---
__all__ = [
    "client", "embedding_model", "df", "vectors", "index", "price_data",
    "price_df", "price_vectors", "price_index", "chat_history"
]

# ============================
# app.py (예시 구조, Streamlit)
# ============================
# from agent.check_rag import agent_check_rag_need
# from agent.expand_query import agent_expand_and_extract_columns
# from agent.extract_user_req import extract_user_requirements, extract_user_keywords
# from search.smart_search import smart_rag_search
# from search.filter_results import filter_results_by_requirements
# from response.generate_response import generate_context_aware_response
# import config

# Streamlit UI + 위 함수들을 조합하여 챗봇 인터페이스 완성
# 예: query → expand → extract req → search → filter → respond

# 각각의 py 파일에서는 config.py에서 필요한 전역 객체 import해서 사용하면 됨
# 예: from config import embedding_model, index, df 등
