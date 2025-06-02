## smart_rag_search 함수 : 요구된 컬럼에 따라 숙소 DB, 가격 DB를 유동적으로 검색하는 함수
## input : 튜플 (확장된 쿼리, 필요 컬럼 리스트, DB타입)
## output : df (top-k)
import numpy as np
import pandas as pd
import faiss
from config import chat_history, embedding_model, client, embedding_model, index, price_index, price_df, df, price_data
from search.filter_results import filter_results_by_requirements

def smart_rag_search(query, required_columns, db_type, user_requirements, top_k = 50):
    """개선된 RAG 검색 함수
    사용자 요구사항을 고려하여 검색 및 필터링

    Args:
        query: 확장된 쿼리
        required_columns: 필요한 컬럼 리스트
        db_type: DB 타입
        user_requirements: 추출된 사용자 요구사항
        top_k: 반환할 최대 결과 수

    Returns:
        필터링된 검색 결과 DataFrame
    """
    # 기존 smart_rag_search 함수 로직 (변경 없음)
    scores = []
    candidates = []
    matched_lodging_ids = []

    lodging_columns = {"Airbnb_ListingID", "Title", "LocalizedNeighbourhood_ML", "LocationDescription", "MainDescription", "Bathrooms", "Bedrooms", "Beds", "PersonCapacity", "Amenities", "isSuperhost", "Rating", "ReviewSummary"}
    price_columns = {'Airbnb_ListingID', 'month', 'Guests', 'Price_Per_Night', 'Cleaning_Fee', 'Airbnb_Service_Fee', 'total_price', 'Available_Ranges', 'total_price_mean'}

    lodging_needed = any(col in lodging_columns for col in required_columns)
    price_needed = any(col in price_columns for col in required_columns)

    query_vec = embedding_model.encode(query).astype("float32")

    lodging_results = pd.DataFrame()
    price_results = pd.DataFrame()

    if lodging_needed:
        distances, indices = index.search(np.array([query_vec]), 200)
        lodging_results = df.iloc[indices[0]].copy()
        lodging_results["score"] = 1 / (1 + distances[0])
        lodging_results = lodging_results.loc[lodging_results.score > 0.01]
        candidates.append(lodging_results)

        matched_lodging_ids = lodging_results["Airbnb_ListingID"].tolist()

    if price_needed:
        if matched_lodging_ids:
            filtered_price_df = price_df[price_df["Airbnb_ListingID"].isin(matched_lodging_ids)].copy()

            if len(filtered_price_df) > 0:
                vectors = price_data["vectors"]
                filtered_vectors = vectors[filtered_price_df.index].astype("float32")
                temp_index = faiss.IndexFlatL2(filtered_vectors.shape[1])
                temp_index.add(filtered_vectors)

                distances, indices = temp_index.search(np.array([query_vec]), min(10000, len(filtered_price_df)))
                price_results = filtered_price_df.iloc[indices[0]].copy()
                price_results["score"] = 1 / (1 + distances[0])
                candidates.append(price_results)
        else:
            distances, indices = price_index.search(np.array([query_vec]), 10000)
            price_results = price_df.iloc[indices[0]].copy()
            price_results["score"] = 1 / (1 + distances[0])
            candidates.append(price_results)

    if not candidates:
        raise ValueError("필요한 컬럼이 숙소DB/가격DB 어디에도 없습니다.")

    # 결과 병합 및 정렬
    if lodging_needed and price_needed and not lodging_results.empty and not price_results.empty:
        merged_results = pd.merge(
            lodging_results,
            price_results,
            on="Airbnb_ListingID",
            suffixes=("_lodging", "_price"),
            how="inner"
        )

        if db_type == '숙소DB':
            merged_results['score_sum'] = merged_results['score_lodging'] * 5 + merged_results['score_price'] * 1
        else:
            merged_results['score_sum'] = merged_results['score_lodging'] * 1 + merged_results['score_price'] * 5

        merged_results = merged_results.sort_values(by='score_sum', ascending=False)
        search_results = merged_results.head(top_k)
    elif lodging_needed and not price_needed:
        lodging_results = lodging_results.sort_values(by="score", ascending=False)
        search_results = lodging_results.head(top_k)
    elif price_needed and not lodging_needed:
        price_results = price_results.sort_values(by="score", ascending=False)
        search_results = price_results.head(top_k)

    # 사용자 요구사항에 따라 결과 필터링
    filtered_results = filter_results_by_requirements(search_results, user_requirements)
    return filtered_results