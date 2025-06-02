
from config import chat_history, embedding_model, client, embedding_model, index, df
import re

def extract_user_requirements(query, chat_history):
    """사용자 요구사항을 상세하게 추출하는 함수

    Args:
        query: 현재 사용자 쿼리
        chat_history: 이전 대화 기록

    Returns:
        추출된 요구사항 딕셔너리
    """
    # 이전 대화 내용 텍스트 구성
    if not chat_history:
        history_text = ""
    else:
        history_text = "\n".join([f"{item['role']}: {item['content']}" for item in chat_history])

    system_prompt = """너는 숙소 추천 및 매니지먼트 전문 AI야.
대화 내용을 분석하여 사용자의 숙소 검색 요구사항을 정확하게 추출해줘.

사용자의 모든 대화(이전 대화 + 현재 질문)를 종합적으로 분석하여 다음 항목들을 추출해:
1. 위치 요구사항 (예: 홍대 근처, 지하철역 5분 이내 등)
2. 인원 (필요한 수용 인원)
3. 가격 범위 (상한선, 하한선)
4. 필수 시설/어메니티 (와이파이, 주차장, 수영장, 노래방 등)
5. 숙소 유형 (파티룸, 한옥, 호텔, 아파트 등)
6. 기타 조건 (체크인/체크아웃 시간, 슈퍼호스트 여부, 음식 반입 등)

각 항목은 반드시 사용자가 명시적으로 언급한 경우만 추출하고, 없을 경우 null로 표시해줘.
이전 대화에서 언급된 조건은 현재 질문에서 명시적으로 변경되지 않는 한 계속 유지되어야 함.

JSON 형식으로 응답해줘:
{
  "location": "string or null",
  "capacity": number or null,
  "price_min": number or null,
  "price_max": number or null,
  "required_amenities": ["string", "string", ...] or [],
  "lodging_type": "string or null",
  "other_conditions": ["string", "string", ...] or []
}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"이전 대화:\n{history_text}\n\n현재 질문:\n{query}\n\nJSON 형식으로 응답해줘."}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.1,
            max_tokens=300,
            response_format={"type": "json_object"}
        )

        # JSON 파싱
        import json
        requirements = json.loads(response.choices[0].message.content)
        return requirements
    except Exception as e:
        print(f"JSON 파싱 오류: {str(e)}")
        # 오류 발생 시 기본 딕셔너리 반환
        return {
            "location": None,
            "capacity": None,
            "price_min": None,
            "price_max": None,
            "required_amenities": [],
            "lodging_type": None,
            "other_conditions": []
        }


def extract_user_keywords(query):
    """다양한 쿼리에서 키워드 추출"""
    keywords = []

    # 키워드 카테고리와 관련 표현들
    keyword_categories = {
        # 숙소 타입
        "파티룸": ["파티", "파티룸", "모임", "행사"],
        "한옥": ["한옥", "전통", "고전"],
        "호텔": ["호텔", "모텔", "여관"],
        "풀빌라": ["풀빌라", "수영장", "pool"],

        # 시설/편의
        "와이파이": ["와이파이", "wifi", "인터넷"],
        "주차": ["주차", "parking"],
        "욕조": ["욕조", "bathtub", "bath"],
        "수영장": ["수영장", "pool", "swimming"],
        "노래방": ["노래방", "노래", "karaoke"],
        "보드게임": ["보드게임", "게임", "board"],
        "음식 반입": ["음식", "배달", "취사", "조리"],

        # 위치 관련
        "지하철 근처": ["지하철", "역", "station", "근처", "가까운"],
        "강남": ["강남", "신논현", "학동"],
        "홍대": ["홍대", "합정", "상수"],
        "이태원": ["이태원", "한남", "녹사평"],

        # 가격 관련
        "가격": ["가격", "원", "만원", "저렴한", "싼", "비싼"],

        # 인원
        "인원": ["인원", "명", "사람", "인", "수용"],

        # 평가 관련
        "별점 높은": ["별점", "평점", "좋은 평가", "만족도"],
        "슈퍼호스트": ["슈퍼호스트", "superhost"],

        # 상태 관련
        "깨끗한": ["깨끗", "청결", "청소", "위생"],
        "조용한": ["조용", "소음", "시끄럽지 않은"]
    }

    # 모든 카테고리와 키워드를 확인
    for category, terms in keyword_categories.items():
        for term in terms:
            if term in query.lower():
                # 숫자 + 키워드 패턴 찾기 (ex. "15명", "10만원")
                if category == "인원" or category == "가격":
                    pattern = rf'(\d+)\s*{term}'
                    matches = re.findall(pattern, query)
                    if matches:
                        keywords.append(f"{matches[0]}{term}")
                        break

                keywords.append(category)
                break  # 하나만 찾으면 다음 카테고리로

    return list(set(keywords))  # 중복 제거하여 반환