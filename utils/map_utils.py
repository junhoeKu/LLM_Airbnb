
import folium
from folium.plugins import MarkerCluster
from config import chat_history, embedding_model, client, embedding_model, index, df

def create_map_from_results(results, highlight_ids=None):
    try:
        if 'Lat' not in results.columns or 'Lng' not in results.columns:
            results = results.merge(df[['Airbnb_ListingID', 'Lat', 'Lng']], on='Airbnb_ListingID', how='left')

        map_center = [results['Lat'].mean(), results['Lng'].mean()]
        m = folium.Map(location=map_center, zoom_start=13)
        marker_cluster = MarkerCluster().add_to(m)

        color_palette = ['red', 'orange', 'green']

        for i, (_, row) in enumerate(results.iterrows()):
            lat, lng = row['Lat'], row['Lng']
            title = row['Title']
            lodging_id = row['Airbnb_ListingID']
            popup_text = f"""
                <div style='font-size:16px; line-height:1.4;'>
                <b>{title}</b><br>ID: {lodging_id}
                </div>
                """

            if highlight_ids and lodging_id in highlight_ids:
                idx = highlight_ids.index(lodging_id)
                folium.Marker(
                    location=[lat, lng],
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=f"추천 {idx+1}: {title}",
                    icon=folium.Icon(color=color_palette[idx % len(color_palette)], icon='star')
                ).add_to(m)
            else:
                folium.Marker(
                    location=[lat, lng],
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=title
                ).add_to(marker_cluster)

        return m._repr_html_()

    except Exception as e:
        return f"<b>🛑 지도 생성 중 오류 발생:</b><br>{str(e)}"
