## agent_check_rag_need 함수 : 가장 최근 쿼리만 보고 RAG 필요 여부만 판단
## input : 사용자 쿼리 (ex. 홍대 숙소를 찾아줘 / 오늘 날씨가 어때)
## output : 예 / 아니오
from config import chat_history, embedding_model, client, embedding_model, index, df

def agent_check_rag_need(query):
    system_prompt = """너는 숙소 추천 및 매니지먼트 전문 AI야.
사용자의 현재 질문을 보고, 숙소 DB 또는 가격 DB 검색(RAG)이 필요한지 판단해.

응답 포맷은 반드시 다음과 같아야 해:

[RAG 필요 여부]
예 / 아니오
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
        max_tokens=10
    )

    output = response.choices[0].message.content.strip()
    rag_needed = "예" in output.strip()
    return rag_needed