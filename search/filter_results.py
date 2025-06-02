
from config import chat_history, embedding_model, client, embedding_model, index, df
import re

def filter_results_by_requirements(results, requirements):
    """사용자 요구사항에 맞게 결과 필터링
    모든 필수 조건을 충족하는 숙소만 반환

    Args:
        results: 검색 결과 DataFrame
        requirements: 추출된 사용자 요구사항

    Returns:
        필터링된 결과 DataFrame
    """
    filtered_results = results.copy()

    # 1. 인원 필터링
    if requirements["capacity"] is not None:
        # 다양한 인원 관련 컬럼 확인
        capacity_fields = ['PersonCapacity', 'Guests', 'max_guests']
        capacity_filtered = False

        for field in capacity_fields:
            if field in filtered_results.columns:
                try:
                    # 숫자로 변환 가능한 필드인 경우
                    numeric_capacity = pd.to_numeric(filtered_results[field], errors='coerce')
                    filtered_results = filtered_results[numeric_capacity >= requirements["capacity"]]
                    capacity_filtered = True
                    break
                except:
                    pass

        # 컬럼으로 필터링 안 된 경우 텍스트 기반 필터링 시도
        if not capacity_filtered and len(filtered_results) > 0:
            # 타이틀 기반 필터링
            if 'Title' in filtered_results.columns:
                def extract_capacity_from_title(title):
                    if not isinstance(title, str):
                        return 0

                    # 정규 표현식으로 숫자 + 인원 표시 패턴 추출
                    pattern = r'(\d+)\s*(?:인|명|person|people|guests)'
                    match = re.search(pattern, title, re.IGNORECASE)
                    if match:
                        return int(match.group(1))
                    return 0

                # 타이틀에서 인원 추출 후 필터링
                capacities = filtered_results['Title'].apply(extract_capacity_from_title)
                filtered_results = filtered_results[capacities >= requirements["capacity"]]

    # 2. 가격 필터링
    if requirements["price_max"] is not None:
        # 다양한 가격 관련 컬럼 확인
        price_fields = ['Price_Per_Night', 'total_price', 'total_price_mean', 'BasicNightPrice', 'BasicNightPrice_Clean']
        price_filtered = False

        for field in price_fields:
            if field in filtered_results.columns:
                try:
                    # 숫자로 변환 가능한 필드인 경우
                    numeric_price = pd.to_numeric(filtered_results[field], errors='coerce')
                    filtered_results = filtered_results[numeric_price <= requirements["price_max"]]
                    price_filtered = True
                    break
                except:
                    pass

    # 3. 필수 어메니티 필터링
    if requirements["required_amenities"] and 'Amenities' in filtered_results.columns:
        for amenity in requirements["required_amenities"]:
            # 어메니티 문자열에 해당 키워드가 포함된 경우만 남김
            filtered_results = filtered_results[
                filtered_results['Amenities'].astype(str).str.contains(amenity, case=False, na=False) |
                filtered_results['Title'].astype(str).str.contains(amenity, case=False, na=False) |
                filtered_results.get('MainDescription', '').astype(str).str.contains(amenity, case=False, na=False)
            ]

    # 4. 숙소 유형 필터링
    if requirements["lodging_type"] is not None:
        if 'RoomType' in filtered_results.columns:
            filtered_results = filtered_results[
                filtered_results['RoomType'].astype(str).str.contains(requirements["lodging_type"], case=False, na=False) |
                filtered_results['SpaceType'].astype(str).str.contains(requirements["lodging_type"], case=False, na=False) |
                filtered_results['Title'].astype(str).str.contains(requirements["lodging_type"], case=False, na=False)
            ]

    # 5. 위치 필터링
    if requirements["location"] is not None:
        location_fields = ['LocalizedNeighbourhood_ML', 'LocationDescription', 'Location']

        for field in location_fields:
            if field in filtered_results.columns:
                filtered_results = filtered_results[
                    filtered_results[field].astype(str).str.contains(requirements["location"], case=False, na=False)
                ]
                if len(filtered_results) > 0:
                    break

    # 6. 기타 조건 필터링
    if requirements["other_conditions"]:
        for condition in requirements["other_conditions"]:
            # 여러 필드에서 해당 조건이 언급되었는지 확인
            condition_pattern = r'' + re.escape(condition)
            text_fields = ['Title', 'MainDescription', 'SpaceDescription', 'Amenities', 'ReviewSummary']

            condition_mask = False
            for field in text_fields:
                if field in filtered_results.columns:
                    condition_mask = condition_mask | filtered_results[field].astype(str).str.contains(condition_pattern, case=False, na=False)

            filtered_results = filtered_results[condition_mask]

    # 최종 필터링된 결과 반환 (유사도 점수 기준 정렬)
    if 'score' in filtered_results.columns:
        filtered_results = filtered_results.sort_values('score', ascending=False)
    elif 'score_sum' in filtered_results.columns:
        filtered_results = filtered_results.sort_values('score_sum', ascending=False)

    # 필터링 결과가 너무 적으면 원본 결과의 일부 반환
    if len(filtered_results) < 3 and len(results) > 0:
        print(f"필터링 결과가 부족합니다: {len(filtered_results)}개. 원본 결과에서 상위 항목 반환")
        return results.head(10)  # 상위 10개만 반환

    return filtered_results