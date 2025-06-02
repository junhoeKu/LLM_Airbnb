
from config import chat_history, embedding_model, client
from config import embedding_model, index, df

## generate_free_response 함수 : 숙소 추천이 필요 없을 때, 자유로운 답변을 생성하는 함수
def generate_free_response(query):
    messages = [
        {"role": "system", "content": "너는 친절하고 유쾌한 여행 도우미야. 숙소 추천이 필요 없는 경우 자연스럽게 답변해줘."},
        {"role": "user", "content": query}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=200
    )
    return response.choices[0].message.content.strip()