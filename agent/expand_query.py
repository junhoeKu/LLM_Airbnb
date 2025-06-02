## agent_expand_and_extract_columns 함수 : RAG가 필요한 경우 (위 로직에서 '예'를 답했다면) 대화 히스토리를 반영해 쿼리를 확장하고 필요한 컬럼을 추출
## input : 사용자 쿼리
## output : 튜플 (확장된 쿼리, 필요 컬럼 리스트, DB타입)

from config import chat_history, embedding_model, client, embedding_model, index, df
import re

def agent_expand_and_extract_columns(query, chat_history):
    """개선된 쿼리 확장 및 컬럼 추출 함수
    이전 대화와 현재 쿼리를 통합하여 완전한 검색 쿼리를 생성합니다.

    Args:
        query: 현재 사용자 쿼리
        chat_history: 이전 대화 기록

    Returns:
        확장된 쿼리, 필요한 컬럼 리스트, DB 타입
    """
    # 이전 대화 내용 텍스트 구성
    if not chat_history:
        history_text = ""
    else:
        history_text = "\n".join([f"{item['role']}: {item['content']}" for item in chat_history])

    # 프롬프트 개선: 이전 조건을 유지하고 새 조건 추가 명시
    system_prompt = """너는 숙소 추천 및 매니지먼트 전문 AI야.
대화 흐름을 분석하여,
1. 이전 대화에서 언급된 모든 조건과 현재 사용자의 요청을 통합하여 완전한 문장으로 요약해.
   - 이전 대화에서 언급된 모든 조건(숙소 유형, 인원 수, 위치, 편의 시설, 가격 등)을 유지해야 함
   - 현재 질문이 새로운 조건을 추가하거나 기존 조건을 변경하는 경우 이를 적절히 반영해야 함
   - 모든 조건들은 사용자가 반드시 만족해야 하는 필수 조건으로 간주함
2. 숙소DB/가격DB에서 필요한 컬럼을 모두 리스트업해.
3. 숙소DB/가격DB 중 어떤 DB가 더 중요한지 판단해.

응답 포맷은 정확하게 다음과 같아야 한다:
[요약]
요약한 문장 (이전 대화의 모든 조건 + 현재 요청)

[필요한 컬럼]
- 컬럼명1
- 컬럼명2
- 컬럼명3
(없으면 '없음'이라고 작성)

[DB]
숙소DB or 가격DB

가능한 컬럼명 목록은 다음과 같다.

[숙소 DB 컬럼]
Airbnb_ListingID, Title, LocalizedNeighbourhood_ML, LocationDescription, MainDescription, Bathrooms, Bedrooms, Beds, PersonCapacity, Amenities, isSuperhost, Rating, ReviewSummary

[가격 DB 컬럼]
Airbnb_ListingID, month, Guests, Price_Per_Night, Cleaning_Fee, Airbnb_Service_Fee, total_price, Available_Ranges, total_price_mean, price_comment
"""

    # GPT 호출 (model="gpt-4o" 같은 고성능 모델 사용)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"이전 대화:\n{history_text}\n\n현재 질문:\n{query}"}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",  # 더 좋은 모델 사용
        messages=messages,
        temperature=0.2,
        max_tokens=300
    )

    output = response.choices[0].message.content.strip()

    # 출력 파싱
    summary_match = re.search(r"\[요약\](.*?)(?=\[필요한 컬럼\])", output, re.DOTALL)
    columns_match = re.findall(r"-\s*(.+)", output.split("[필요한 컬럼]")[-1].split("[DB]")[0])
    db_match = re.search(r"\[DB\]\s*(숙소DB|가격DB)", output)

    expanded_query = summary_match.group(1).strip() if summary_match else query
    required_columns = [col.strip() for col in columns_match if col.strip().lower() != "없음"]
    db_type = db_match.group(1).strip() if db_match else "알 수 없음"

    # 디버깅 로그
    print(f"확장된 쿼리: {expanded_query}")
    print(f"필요 컬럼: {required_columns}")
    print(f"DB 타입: {db_type}")

    return expanded_query, required_columns, db_type