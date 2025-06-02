import folium
from folium.plugins import MarkerCluster

# 기능 모듈 import
from agent.check_rag import agent_check_rag_need
from agent.expand_query import agent_expand_and_extract_columns
from agent.extract_user_req import extract_user_requirements
from search.smart_search import smart_rag_search
from search.filter_results import filter_results_by_requirements
from response.generate_response import generate_context_aware_response
from utils.map_utils import create_map_from_results
from utils.response_utils import generate_free_response

# config에서 전역 자원 import
from config import chat_history, embedding_model, index, df, price_index, price_df, client

# pip install --upgrade streamlit openai sentence-transformers faiss-gpu numpy
# pip install transformers==4.45.2
# streamlit run app.py

def chat_interface(query, history):
    """개선된 채팅 인터페이스 함수
    멀티턴 대화 지원 및 조건 충족 검사 강화

    Args:
        query: 사용자 쿼리
        history: 대화 기록

    Returns:
        업데이트된 대화 기록, 채팅 기록, 지도 HTML
    """
    history = history or []
    try:
        # RAG 필요 여부 확인
        rag_needed = agent_check_rag_need(query)

        if not rag_needed:
            # 일반 대화 응답
            bot_response = generate_free_response(query)
            chat_history.append({"role": "user", "content": query})  # ← config.chat_history 사용
            map_html = ""
        else:
            # 대화 기록 추가 후 쿼리 확장 및 컬럼 추출
            chat_history.append({"role": "user", "content": query})
            expanded_query, required_columns, db_type = agent_expand_and_extract_columns(query, chat_history)

            # 사용자 요구사항 추출
            user_requirements = extract_user_requirements(query, chat_history)

            # 개선된 검색
            search_results = smart_rag_search(
                expanded_query,
                required_columns,
                db_type,
                user_requirements
            )

            # 응답 생성
            bot_response, recommended_ids = generate_context_aware_response(
                expanded_query,
                search_results,
                required_columns,
                db_type,
                user_requirements
            )

            # 지도 생성
            map_html = create_map_from_results(search_results, highlight_ids=recommended_ids)

        # 대화 기록 업데이트
        chat_history.append({"role": "assistant", "content": bot_response})
        history.append((query, bot_response))

        return history, history, map_html

    except Exception as e:
        error_msg = f"❗ 오류 발생: {str(e)}"
        history.append((query, error_msg))
        return history, history, f"<b>🛑 오류 발생:</b><br>{str(e)}"