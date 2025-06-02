import streamlit as st
from streamlit.components.v1 import html
from chat_interface import chat_interface
import time  # ⏰ 시간 측정용

# ===== 페이지 설정 =====
st.set_page_config(page_title="AI 숙소 추천", layout="wide")

# ===== 상태 관리 =====
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_state" not in st.session_state:
    st.session_state.chat_state = None
if "map_html" not in st.session_state:
    st.session_state.map_html = ""

# ===== GPT 스타일 챗봇 UI =====
st.title("🏡 AI 숙소 추천 챗봇")

st.markdown("""
채팅창에 원하시는 숙소 조건을 입력해주세요.  
예: *강남 2인용 감성 숙소 찾아줘*, *주차 가능하고 반려동물 입장되는 숙소 있어?*
""")

# ===== 좌측: 채팅 / 우측: 지도 =====
chat_col, map_col = st.columns([2, 1])

with chat_col:
    # === 대화 내역 ===
    for idx, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_msg)
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(bot_msg)

    # === 입력창 ===
    st.divider()
    with st.chat_message("user"):
        user_query = st.text_input(
            "원하시는 숙소 조건을 입력해주세요",
            key="user_input",
            label_visibility="collapsed"
        )

    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("🔍 검색하기", use_container_width=True):
            if user_query.strip():
                start_time = time.time()  # 시작 시간
                with st.spinner("숙소를 찾고 있어요...🛌"):
                    history, _, map_html = chat_interface(user_query, st.session_state.chat_state)
                    end_time = time.time()  # 끝난 시간
                    elapsed = round(end_time - start_time, 2)  # 경과시간 계산
                    st.success(f"✅ 답변 생성 완료! ⏱️ 소요 시간: {elapsed}초")
                    
                    st.session_state.chat_history = history
                    st.session_state.chat_state = history
                    st.session_state.map_html = map_html  # 지도 저장

                st.experimental_set_query_params(reset="true")
                st.rerun()

    # === 초기화 버튼 ===
    if st.button("🧹 대화 초기화"):
        st.session_state.chat_history = []
        st.session_state.chat_state = None
        st.session_state.map_html = ""
        st.experimental_set_query_params(reset="true")
        st.rerun()

with map_col:
    st.markdown("#### 🗺️ 추천 숙소 지도")
    if st.session_state.map_html:
        html(st.session_state.map_html, height=500)
    else:
        st.info("숙소 검색 시 지도 정보가 여기에 표시됩니다.")