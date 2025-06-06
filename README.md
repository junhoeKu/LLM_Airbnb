# 🏡 Airbnb 숙소 추천 LLM 프로젝트

## 🎯 1. 프로젝트 소개

### 🧑‍🤝‍🧑 **팀원 (AI빅데이터융합경영학과)**
- **팀장 : 권민지**
- **팀원 : 구준회, 이지민, 방재혁, 유찬우**

### 📅 **진행 기간**
- **2025.03.04 ~ 2025.06.10**

### 🕹️ **주제**
- **사용자 조건에 따라 적절한 Airbnb 숙소를 추천하는 LLM 기반 챗봇 개발**

### ☑️ **목적**
- 기존 Airbnb 필터는 **'감성'이나 '분위기' 같은 주관적 요소를 찾기 어려움**
  ➝ 리뷰와 설명을 기반으로 **자연어 해석(LLM)을 활용**해 숙소를 추천

- **복잡한 필터 사용에 따른 검색 피로도 증가**
  ➝ **한 문장 입력**만으로 간편한 숙소 추천 제공

### 🎯 **목표**
- LLM을 기반으로 **사용자의 다양한 요구조건를 이해**하고, Airbnb 숙소 중 **가장 적합한 후보를 추천**
- 요구 조건에 맞지 않는 숙소는 제외하고, 조건 충족 여부를 명확하게 표시
- 추천 이유와 함께 **실재감을 느낄 수 있는 자연어 응답** 생성
- **사용자 친화적 인터페이스** 구현

---

## 📊 2. 데이터 수집 및 DB 구성

- **Airbnb 데이터셋**
  - 주요 컬럼: 숙소 ID, 제목, 위치, 가격, 어메니티, 공간 구성, 최대 수용 인원, 평점, 슈퍼호스트 여부, 숙소 설명, 리뷰 요약, 분위기, 편의시설 등 
    - 추후 데이터 확보 시 확장 예정
- **전처리 기법**
  - 불용어 처리, 번역 등 일반적인 자연어 전처리
  - Aribnb_ListingID 기준으로 데이터셋 병합 후 사용
- **Faiss Library + DB 구축 (RAG)**
  - Faiss Flat 방식으로 DB 구축 후 한국어 임베딩 모델을 활용해 유사도 비교
    - [jhgan/ko-sroberta-multitask](https://huggingface.co/jhgan/ko-sroberta-multitask)

---

## 💻 3. 모델 및 파이프라인 

### 3-1. 전체 파이프라인 구성 

<img src="https://github.com/user-attachments/assets/a4efed9e-e44d-4d57-9192-340fc7f2230c" alt="Pipeline" width="700"/>

### 3-2. 모델

- **오픈소스 한국어 LLM vs GPT API**
  - 성능 및 토큰 수, Inference Time을 고려하여 GPT API 활용
  - 찾아본 한국어 LLM
  - [rtzr/ko-gemma-2-9b-it](https://huggingface.co/rtzr/ko-gemma-2-9b-it)
  - [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

- 주요 모듈 구성:
  - **Query Analyzer**: 사용자 질의에서 조건 추출
  - **Structured Filter**: 조건 충족하는 숙소만 필터링
  - **Response Generator**: 추천 이유와 함께 자연스럽고 친절한 문장 생성

### 3-3. 조건 강화 프롬프트 설계

- **사용자 조건 미충족 숙소는 절대 추천하지 않도록 유도**
- 숙소 목록 중에서만 선택하도록 **ID 기반 제한 조건** 명시
- **조건 충족 여부, 추천 사유, 리뷰 요약** 등 핵심 정보 포함
---

## 📈 4. 결과 
### 4-1. 성과 및 예시

- **다양한 조건을 조합한 테스트에서 실제 Airbnb 숙소를 정확하게 필터링**
- **진정성 있는 응답 생성 품질 향상**
- **예시**
> 사용자: 강남 근처에 4명 잘 수 있고 수영장이 있는 숙소를 추천해줘
  
> 챗봇: 사용자님, ‘**강남 / 4인 / 수영장**’ 조건을 만족하는 숙소를 찾았습니다!  
> 아래 두 숙소는 조건을 모두 충족하며, 최근 후기에서도 매우 긍정적인 반응을 얻고 있어요 😊
  
| 숙소명 | ID | 조건 충족 | 특징 | 대표 후기 |
|--------|-----|------------|--------|-------------|
| **리버뷰 레지던스 강남점** | 38492 | ✅ 모두 충족 | 수영장, 루프탑 바 | “숙소가 넓고 강 전망이 끝내줘요” |
| **모던하우스 강남역점** | 37284 | ✅ 모두 충족 | 가족 단위에게 인기 | “친절한 호스트와 깨끗한 수영장” |

### 4-2. **인터페이스**

<img src="https://github.com/user-attachments/assets/2820bb11-6558-49fa-99bc-59da24870303" alt="Interface" width="700"/>

---

## 🔍 5. 향후 계획 

- 다국어 입력에 대한 조건 분석 지원 (예: 영어 사용자)
- 세밀한 조건 우선순위 설정 기능 (예: "수영장이 가장 중요")
- Java Script를 활용한 **역동적 대화형 UI 개발**
- DB 추가 구축을 통한 정확도 향상 

---

## 📚 6. 참고 자료

- **Airbnb 데이터셋**
- **LLM 기반 조건 추론 연구 논문들**
- Hugging Face 기반 LLM:
  - [rtzr/ko-gemma-2-9b-it](https://huggingface.co/rtzr/ko-gemma-2-9b-it)
  - [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- **프로젝트 진행용 협업 툴**
  - Notion, GitHub, Slack

