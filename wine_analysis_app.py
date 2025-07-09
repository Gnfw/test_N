# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import textwrap
from matplotlib import gridspec
from io import StringIO
import hashlib
import logging
import plotly.express as px
from wordcloud import WordCloud
from textblob import TextBlob
import folium
from streamlit_folium import st_folium
from datetime import datetime
import tempfile
from fpdf import FPDF
import base64
import re

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Настройки страницы Streamlit
st.set_page_config(
    page_title="Расширенный анализ вин",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Настройки графиков
pd.set_option('display.max_columns', None)
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (16, 10)
sns.set_palette("husl")

# Словари для перевода
column_translation = {
    'country': 'Страна',
    'points': 'Рейтинг',
    'price': 'Цена',
    'variety': 'Сорт',
    'winery': 'Винодельня',
    'year': 'Год',
    'province': 'Провинция',
    'region_1': 'Регион 1',
    'region_2': 'Регион 2',
    'description': 'Описание'
}

stat_translation = {
    'mean': 'Среднее',
    'median': 'Медиана',
    'std': 'Станд. отклонение',
    'count': 'Количество'
}

country_translation = {
    'Australia': 'Австралия',
    'New Zealand': 'Новая Зеландия',
    'US': 'США',
    'France': 'Франция',
    'Italy': 'Италия',
    'Spain': 'Испания',
    'Portugal': 'Португалия',
    'Chile': 'Чили',
    'Argentina': 'Аргентина'
}

class PDF(FPDF):
    def __init__(self):
        super().__init__()
        try:
            font_path = os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans.ttf")
            self.add_font("DejaVu", "", font_path, uni=True)
            self.set_font("DejaVu", size=10)
        except:
            self.set_font("helvetica", size=10)

    def safe_cell(self, w, h=0, txt="", border=0, ln=0, align="L"):
        required_width = self.get_string_width(txt) + 2
        self.cell(w if w > required_width else required_width, h, txt, border, ln, align)

def translate_data(df):
    df = df.rename(columns=column_translation)
    if 'Страна' in df.columns:
        df['Страна'] = df['Страна'].replace(country_translation)
    return df

def translate_stats(df):
    try:
        translated = df.copy()
        if isinstance(translated.index, pd.MultiIndex):
            new_index = []
            for level in translated.index.levels:
                if level.name in stat_translation:
                    new_index.append(level.map(stat_translation))
                else:
                    new_index.append(level)
            translated.index = pd.MultiIndex.from_arrays(new_index, names=translated.index.names)
        else:
            if translated.index.name in stat_translation:
                translated.index = translated.index.map(lambda x: stat_translation.get(x, x))
        return translated
    except Exception as e:
        logger.error(f"Ошибка перевода статистики: {str(e)}")
        return df

def get_file_hash(uploaded_file):
    uploaded_file.seek(0)
    return hashlib.md5(uploaded_file.read()).hexdigest()

def extract_year_from_description(description):
    try:
        if pd.isna(description):
            return None
        matches = re.findall(r'(19|20\d{2})', str(description))
        return int(matches[0]) if matches else None
    except:
        return None

def load_data(uploaded_file, use_cache=True):
    try:
        file_hash = get_file_hash(uploaded_file)
        cache_key = f"wine_data_{file_hash}"
        
        if not use_cache or cache_key not in st.session_state:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Неподдерживаемый формат файла")
                return None
                
            df = translate_data(df)
            if 'Описание' in df.columns:
                df['Год'] = df['Описание'].apply(extract_year_from_description)
            df = df.dropna(subset=['Рейтинг', 'Цена'])
            df = df.drop_duplicates()
            
            if use_cache:
                st.session_state[cache_key] = df
            return df
        return st.session_state[cache_key]
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {str(e)}")
        return None

def create_summary_plot(data, filtered_data, variety):
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.5])
    
    ax1 = plt.subplot(gs[0, 0])
    sns.boxplot(x='Страна', y='Рейтинг', data=filtered_data, ax=ax1)
    ax1.set_title(f'Распределение рейтингов {variety}')
    
    ax2 = plt.subplot(gs[0, 1])
    sns.boxplot(x='Страна', y='Цена', data=filtered_data, ax=ax2)
    ax2.set_title(f'Распределение цен {variety}')
    
    ax3 = plt.subplot(gs[1, 0])
    sns.scatterplot(x='Рейтинг', y='Цена', hue='Страна', data=filtered_data, ax=ax3, alpha=0.7)
    ax3.set_title('Зависимость цены от рейтинга')
    
    ax4 = plt.subplot(gs[1, 1])
    if 'Год' in filtered_data.columns:
        filtered_data['Год'] = pd.to_numeric(filtered_data['Год'], errors='coerce')
        filtered_data = filtered_data.dropna(subset=['Год'])
        year_counts = filtered_data.groupby(['Страна', 'Год']).size().reset_index(name='Количество')
        if not year_counts.empty:
            pivot_data = year_counts.pivot(index='Год', columns='Страна', values='Количество')
            pivot_data.plot(kind='bar', stacked=True, ax=ax4)
            ax4.set_title('Распределение по годам')
    
    ax5 = plt.subplot(gs[2, :])
    ax5.axis('off')
    plt.tight_layout()
    return fig

def create_interactive_plot(data, variety):
    fig = px.scatter(
        data,
        x='Рейтинг',
        y='Цена',
        color='Страна',
        hover_data=['Винодельня', 'Провинция', 'Регион 1'],
        title=f'Анализ {variety}',
        size_max=15
    )
    fig.update_layout(height=600, width=800)
    return fig

def create_wordcloud(data, variety):
    text = ' '.join(data['Описание'].dropna().astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    return fig

def analyze_sentiment(data):
    data['sentiment'] = data['Описание'].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity if pd.notna(x) else 0)
    fig = px.box(data, x='Страна', y='sentiment', color='Страна')
    return fig, data

def create_geographical_analysis(data):
    if 'Провинция' not in data.columns:
        return None
    
    region_stats = data.groupby(['Страна', 'Провинция', 'Регион 1']).agg({
        'Рейтинг': 'mean',
        'Цена': 'mean',
        'Сорт': 'count'
    }).reset_index()
    
    m = folium.Map(location=[20, 0], zoom_start=2)
    for _, row in region_stats.iterrows():
        folium.Marker(
            location=[np.random.uniform(-60, 70), np.random.uniform(-180, 180)],
            popup=f"{row['Регион 1']}<br>Рейтинг: {row['Рейтинг']:.1f}<br>Цена: ${row['Цена']:.1f}"
        ).add_to(m)
    return m, region_stats

def create_pdf_report(data, variety, stats):
    try:
        pdf = PDF()
        pdf.add_page()
        pdf.set_font(size=16)
        pdf.safe_cell(0, 10, f"Анализ вин: {variety}", ln=1, align="C")
        pdf.ln(10)
        
        pdf.set_font(size=12)
        pdf.safe_cell(0, 10, "Статистика по странам:", ln=1)
        for country in stats.index:
            text = f"{country}: Рейтинг {stats.loc[country, ('Рейтинг', 'mean')]:.1f}, Цена ${stats.loc[country, ('Цена', 'mean')]:.1f}"
            pdf.safe_cell(0, 10, text, ln=1)
        
        pdf.ln(5)
        pdf.safe_cell(0, 10, "Топ-5 вин:", ln=1)
        for _, row in data.nlargest(5, 'Цена')[['Винодельня', 'Страна', 'Рейтинг', 'Цена']].iterrows():
            text = f"{row['Винодельня']} ({row['Страна']}): {row['Рейтинг']} баллов, ${row['Цена']}"
            pdf.safe_cell(0, 10, text, ln=1)
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        pdf.output(temp_file.name)
        return temp_file.name
    except Exception as e:
        logger.error(f"PDF error: {str(e)}")
        return None

def analyze_wine(data, variety, countries):
    if data is None or data.empty:
        st.warning("Нет данных для анализа")
        return
    
    filtered_data = data[data['Сорт'].str.contains(variety, case=False, na=False)]
    filtered_data = filtered_data[filtered_data['Страна'].isin(countries)]
    
    if len(filtered_data) == 0:
        st.warning("Не найдено данных")
        return
    
    with st.spinner('Анализируем...'):
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Аналитика", "Графики", "Текст", "Карта", "Отчет"])
        
        with tab1:
            st.pyplot(create_summary_plot(data, filtered_data, variety))
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(filtered_data.nlargest(5, 'Цена'))
            with col2:
                st.dataframe(filtered_data.nlargest(5, 'Рейтинг'))
            
            stats = filtered_data.groupby('Страна').agg({
                'Рейтинг': ['mean', 'median', 'std'],
                'Цена': ['mean', 'median', 'std'],
                'Сорт': 'count'
            })
            st.dataframe(translate_stats(stats))
        
        with tab2:
            fig = create_interactive_plot(filtered_data, variety)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            if 'Описание' in filtered_data.columns:
                st.pyplot(create_wordcloud(filtered_data, variety))
                sentiment_fig, _ = analyze_sentiment(filtered_data)
                st.plotly_chart(sentiment_fig)
        
        with tab4:
            geo_result = create_geographical_analysis(filtered_data)
            if geo_result:
                st_folium(geo_result[0], width=800)
                st.dataframe(geo_result[1])
        
        with tab5:
            if st.button("Создать PDF отчет"):
                pdf_path = create_pdf_report(filtered_data, variety, stats)
                if pdf_path:
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="Скачать отчет",
                            data=f,
                            file_name=f"wine_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf"
                        )
                    os.unlink(pdf_path)

def main():
    st.title("🍷 Анализ вин")
    
    uploaded_file = st.sidebar.file_uploader("Загрузите данные", type=["csv", "xlsx"])
    if uploaded_file:
        data = load_data(uploaded_file)
        if data is not None:
            countries = st.sidebar.multiselect(
                "Выберите страны",
                options=data['Страна'].unique(),
                default=data['Страна'].unique()[:2]
            )
            variety = st.sidebar.selectbox(
                "Сорт вина",
                options=data['Сорт'].unique()
            )
            if st.sidebar.button("Анализировать"):
                analyze_wine(data, variety, countries)

if __name__ == "__main__":
    main()
