from config import chat_history, embedding_model, client, embedding_model, index, df
import re
import pandas as pd
import numpy as np

def generate_context_aware_response(query, results, required_columns, db_type, user_requirements):
    """개선된 컨텍스트 기반 응답 생성 함수

    Args:
        query: 사용자 쿼리
        results: 검색 결과 DataFrame
        required_columns: 필요한 컬럼 리스트
        db_type: DB 타입
        user_requirements: 추출된 사용자 요구사항

    Returns:
        생성된 응답 문자열
    """
    # 고정 컬럼 설정
    fixed_columns = ["Airbnb_ListingID", "Title", "LocalizedNeighbourhood_ML", "ReviewSummary",
                    "StarRating", "BasicNightPrice", "BasicNightPrice_Clean", "PersonCapacity",
                    "Host_isSuperhost", "isSuperhost", "CheckIn_Start", "CheckIn_End", "CheckOut_End",
                    "ReviewScore", "ReviewCount", "Amenities", "RoomType", "SpaceType", "Beds", "Bedrooms", "Bathrooms"]

    fixed_columns_price = ["Airbnb_ListingID", "Title", "LocalizedNeighbourhood_ML", "ReviewSummary",
                         "total_price", "price_comment", "Guests", "PersonCapacity", "Price_Per_Night",
                         "Cleaning_Fee", "Airbnb_Service_Fee", "Taxes", "Stay_Checkin", "Stay_Checkout"]

    # 필요한 컬럼 설정
    if db_type == '숙소DB':
        all_columns = list(set(fixed_columns + required_columns))
    else:
        all_columns = list(set(fixed_columns_price + required_columns))

    # 필요한 컬럼만 추출 (없는 컬럼은 "정보 없음" 처리)
    for col in all_columns:
        if col not in results.columns:
            results[col] = "정보 없음"

    # 충족 조건과 미충족 조건을 구분하여 컨텍스트 생성
    context_lines = []

    # 결과 처리
    for idx, row in results.iterrows():
        if idx >= 5:  # 상위 5개만 처리
            break

        candidate_info = [f"▶ 후보 {idx+1}"]

        # 기본 정보 추출
        candidate_info.append(f"- 숙소명: {row.get('Title', '정보 없음')} [ID: {row.get('Airbnb_ListingID', '정보 없음')}]")
        candidate_info.append(f"- 위치: {row.get('LocalizedNeighbourhood_ML', '정보 없음')}")

        # 가격 정보
        price = "정보 없음"
        price_fields = ['Price_Per_Night', 'total_price', 'total_price_mean', 'price',
                        'BasicNightPrice_Clean', 'BasicNightPrice', 'Price', 'Price_Verbose']

        # 접미사 추가
        suffixes = ['', '_lodging', '_price']
        all_price_fields = []
        for field in price_fields:
            for suffix in suffixes:
                all_price_fields.append(f"{field}{suffix}")

        # 실제로 존재하는 컬럼만 필터링
        existing_price_fields = [field for field in all_price_fields if field in row.index]

        # 가격 필드에서 값 추출
        for price_field in existing_price_fields:
            if pd.notna(row[price_field]) and str(row[price_field]) != "정보 없음" and str(row[price_field]) != "nan":
                try:
                    price_value = float(row[price_field])
                    price = f"{int(price_value):,}원"
                    break
                except:
                    if isinstance(row[price_field], str) and row[price_field].strip():
                        num_match = re.search(r'(\d[\d,.]+)', row[price_field])
                        if num_match:
                            try:
                                num_value = float(num_match.group(1).replace(',', ''))
                                price = f"{int(num_value):,}원"
                                break
                            except:
                                pass

        candidate_info.append(f"- 가격: {price}")

        # 수용 인원
        person_capacity = "정보 없음"
        for capacity_field in ['PersonCapacity', 'Guests', 'max_guests']:
            if capacity_field in row and pd.notna(row[capacity_field]) and str(row[capacity_field]) != "정보 없음" and str(row[capacity_field]) != "nan":
                try:
                    person_capacity = f"{int(float(row[capacity_field]))}명"
                    break
                except:
                    if isinstance(row[capacity_field], str) and row[capacity_field].strip():
                        person_capacity = row[capacity_field]
                        break

        candidate_info.append(f"- 수용 인원: {person_capacity}")

        # 별점 정보
        star_rating = "정보 없음"
        for rating_field in ['StarRating', 'ReviewScore', 'Rating', 'GuestSatisfactionOverall']:
            if rating_field in row and pd.notna(row[rating_field]) and str(row[rating_field]) != "정보 없음" and str(row[rating_field]) != "nan":
                try:
                    rating_value = float(row[rating_field])
                    if 0 <= rating_value <= 5:
                        star_rating = f"{rating_value:.1f}"
                        break
                    elif 0 <= rating_value <= 100:
                        star_rating = f"{rating_value/20:.1f}"
                        break
                except:
                    if isinstance(row[rating_field], str) and row[rating_field].strip():
                        star_rating = row[rating_field]
                        break

        candidate_info.append(f"- 별점: {star_rating}")

        # 슈퍼호스트 여부
        is_superhost = "아니오"
        for superhost_field in ['Host_isSuperhost', 'isSuperhost', 'host_is_superhost']:
            if superhost_field in row:
                field_value = row[superhost_field]
                if pd.notna(field_value) and field_value:
                    if isinstance(field_value, bool) and field_value:
                        is_superhost = "예"
                        break
                    elif isinstance(field_value, (int, float)) and field_value > 0:
                        is_superhost = "예"
                        break
                    elif isinstance(field_value, str) and field_value.lower() in ['true', 'yes', '1', 't', 'y', '예']:
                        is_superhost = "예"
                        break

        candidate_info.append(f"- 슈퍼호스트 여부: {is_superhost}")

        # 사용자 요구사항 충족 여부 체크
        fulfill_checks = []

        # 위치 체크
        if user_requirements["location"] is not None:
            location_fields = ['LocalizedNeighbourhood_ML', 'LocationDescription', 'Location', 'Title']
            location_fulfilled = False
            details = ""

            for field in location_fields:
                if field in row and isinstance(row[field], str):
                    if user_requirements["location"].lower() in row[field].lower():
                        location_fulfilled = True
                        details = row[field]
                        break

            fulfill_checks.append({
                "name": f"위치: {user_requirements['location']}",
                "fulfilled": location_fulfilled,
                "details": details
            })

        # 수용 인원 체크
        if user_requirements["capacity"] is not None:
            capacity_fields = ['PersonCapacity', 'Guests', 'max_guests']
            capacity_fulfilled = False
            details = ""

            for field in capacity_fields:
                if field in row and pd.notna(row[field]):
                    try:
                        capacity = int(float(row[field]))
                        if capacity >= user_requirements["capacity"]:
                            capacity_fulfilled = True
                            details = f"{capacity}명 수용 가능"
                            break
                    except:
                        pass

            fulfill_checks.append({
                "name": f"수용 인원: {user_requirements['capacity']}명 이상",
                "fulfilled": capacity_fulfilled,
                "details": details
            })

        # 가격 범위 체크
        if user_requirements["price_max"] is not None:
            price_fields = ['Price_Per_Night', 'total_price', 'total_price_mean', 'BasicNightPrice', 'BasicNightPrice_Clean']
            price_fulfilled = False
            details = ""

            for field in price_fields:
                if field in row and pd.notna(row[field]):
                    try:
                        price_value = float(row[field])
                        if price_value <= user_requirements["price_max"]:
                            price_fulfilled = True
                            details = f"{int(price_value):,}원"
                            break
                    except:
                        pass

            fulfill_checks.append({
                "name": f"가격: {user_requirements['price_max']:,}원 이하",
                "fulfilled": price_fulfilled,
                "details": details
            })

        # 필수 어메니티 체크
        if user_requirements["required_amenities"]:
            for amenity in user_requirements["required_amenities"]:
                amenity_fulfilled = False
                details = ""

                # 여러 필드에서 어메니티 검색
                text_fields = ['Amenities', 'Title', 'MainDescription', 'SpaceDescription']
                for field in text_fields:
                    if field in row and isinstance(row[field], str):
                        if amenity.lower() in row[field].lower():
                            amenity_fulfilled = True
                            details = amenity
                            break

                fulfill_checks.append({
                    "name": f"어메니티: {amenity}",
                    "fulfilled": amenity_fulfilled,
                    "details": details
                })

        # 숙소 유형 체크
        if user_requirements["lodging_type"] is not None:
            type_fields = ['RoomType', 'SpaceType', 'Title', 'MainDescription']
            type_fulfilled = False
            details = ""

            for field in type_fields:
                if field in row and isinstance(row[field], str):
                    if user_requirements["lodging_type"].lower() in row[field].lower():
                        type_fulfilled = True
                        details = row[field]
                        break

            fulfill_checks.append({
                "name": f"숙소 유형: {user_requirements['lodging_type']}",
                "fulfilled": type_fulfilled,
                "details": details
            })

        # 기타 조건 체크
        if user_requirements["other_conditions"]:
            for condition in user_requirements["other_conditions"]:
                condition_fulfilled = False
                details = ""

                text_fields = ['Title', 'MainDescription', 'SpaceDescription', 'Amenities', 'ReviewSummary']
                for field in text_fields:
                    if field in row and isinstance(row[field], str):
                        if condition.lower() in row[field].lower():
                            condition_fulfilled = True
                            details = condition
                            break

                fulfill_checks.append({
                    "name": f"조건: {condition}",
                    "fulfilled": condition_fulfilled,
                    "details": details
                })

        # 충족한 조건과 충족하지 못한 조건 분리
        fulfilled_conditions = [check for check in fulfill_checks if check["fulfilled"]]
        unfulfilled_conditions = [check for check in fulfill_checks if not check["fulfilled"]]

        # 충족 조건 목록 추가
        if fulfilled_conditions:
            candidate_info.append("\n※ 요청하신 조건 충족 내역:")
            for check in fulfilled_conditions:
                candidate_info.append(f"- {check['name']}: 충족 ({check['details']})")

        # 미충족 조건 목록 추가 (있는 경우)
        if unfulfilled_conditions:
            candidate_info.append("\n※ 미충족 조건:")
            for check in unfulfilled_conditions:
                candidate_info.append(f"- {check['name']}: 미충족")

        # 리뷰 요약
        review_summary = row.get('ReviewSummary', '정보 없음')
        if review_summary != '정보 없음' and pd.notna(review_summary):
            review_summary = str(review_summary)
            review_summary = review_summary[:200] + "..." if len(review_summary) > 200 else review_summary
            candidate_info.append(f"\n- 리뷰 요약: {review_summary}")
        else:
            candidate_info.append("\n- 리뷰 요약: 정보 없음")

        context_lines.append("\n".join(candidate_info))

    context = "\n\n".join(context_lines)

    # 프롬프트 개선
    system_content = f"""## 역할
당신은 서울 숙소 추천 전문가입니다. 검색 결과 중에서 사용자 요청에 가장 적합한 숙소 3개를 선별하여 추천해주세요.

## 검색 결과
{context}

## 사용자 요구사항
{query}

## 요구사항 세부 내역
- 위치: {user_requirements.get("location", "특별히 지정하지 않음")}
- 수용 인원: {f"{user_requirements.get('capacity')}명 이상" if user_requirements.get("capacity") else "특별히 지정하지 않음"}
- 가격 범위: {f"{user_requirements.get('price_max'):,}원 이하" if user_requirements.get("price_max") else "특별히 지정하지 않음"}
- 필수 어메니티: {', '.join(user_requirements.get("required_amenities", [])) or "특별히 지정하지 않음"}
- 숙소 유형: {user_requirements.get("lodging_type", "특별히 지정하지 않음")}
- 기타 조건: {', '.join(user_requirements.get("other_conditions", [])) or "특별히 지정하지 않음"}

## 답변 지침
1. 반드시 모든 필수 조건을 충족하는 숙소만 추천해야 합니다.
2. "미충족 조건"이 있는 숙소는 추천하지 마세요.
3. ID {', '.join(map(str, results['Airbnb_ListingID'].tolist()))} 중에서만 선택하세요.
4. 모든 추천 숙소에 대해 조건 충족 여부를 명확히 표시하세요.
5. 진정성 있고 자연스러운 말투로 답변하세요.
6. 각 숙소의 리뷰 요약과 추천 이유를 간략하게 제공하세요.
7. 가독성 좋게 중간중간에 적절히 줄바꿈 문자를 추가하세요.

## 숙소 추천 답변 형식
사용자님, [사용자 요구사항 요약]을 찾으셨군요!

[최종 추천 숙소 목록]

1. [ID: 숙소ID] 숙소명:
• 위치:
• 가격:
• 수용 인원:
• 별점:
• 슈퍼호스트 여부:
\n
※ 충족 조건
• [조건1]: [상세 정보] ✅
• [조건2]: [상세 정보] ✅
• [조건3]: [상세 정보] ✅
\n
리뷰 요약: [이 숙소의 리뷰 요약]
\n
추천 이유: [이 숙소를 추천하는 구체적인 이유]
\n
---
(나머지 추천 숙소도 같은 형식으로 작성)

만약 모든 필수 조건을 충족하는 숙소가 부족하다면, 그 사실을 정직하게 알리고 가장 조건에 근접한 대안을 제시하세요.
"""

    # GPT 호출
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # 더 좋은 모델 사용
            messages=[
                {"role": "system", "content": system_content}
            ],
            temperature=0.5,
            max_tokens=1000
        )
        response_text = response.choices[0].message.content.strip()
        recommended_ids = re.findall(r"\[ID:\s*(\d+)\]", response_text)
        recommended_ids = [int(rid) for rid in recommended_ids[:3]]  ## 최대 3개만 추출

        return response_text, recommended_ids
    except Exception as e:
        return f"검색 결과 처리 중 오류가 발생했습니다: {str(e)}"