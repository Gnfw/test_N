# -*- coding: utf-8 -*-
# Импорт всех необходимых библиотек
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import textwrap
from matplotlib import gridspec
from io import BytesIO
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

# ============================================
# НАСТРОЙКА ПРИЛОЖЕНИЯ
# ============================================

# Конфигурация логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Настройки страницы Streamlit
st.set_page_config(
    page_title="🍷 Винный аналитик Pro",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# КОНСТАНТЫ И СЛОВАРИ
# ============================================

# Словарь перевода столбцов
COLUMN_TRANSLATION = {
    'country': 'Страна',
    'points': 'Рейтинг',
    'price': 'Цена',
    'variety': 'Сорт',
    'winery': 'Винодельня',
    'description': 'Описание',
    'province': 'Провинция',
    'region_1': 'Регион'
}

# Словарь перевода статистических терминов
STAT_TRANSLATION = {
    'mean': 'Среднее',
    'median': 'Медиана',
    'std': 'Станд. отклонение',
    'count': 'Количество'
}

# Словарь перевода стран
COUNTRY_TRANSLATION = {
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

# ============================================
# КЛАСС ДЛЯ ГЕНЕРАЦИИ PDF
# ============================================

class WinePDF(FPDF):
    """Класс для создания PDF отчетов с поддержкой кириллицы"""
    def __init__(self):
        super().__init__()
        try:
            # Попытка загрузить шрифт DejaVu
            font_path = os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans.ttf")
            self.add_font("DejaVu", "", font_path, uni=True)
            self.set_font("DejaVu", size=10)
        except Exception as e:
            logger.warning(f"Не удалось загрузить шрифт DejaVu: {e}")
            self.set_font("helvetica", size=10)  # Резервный шрифт

    def safe_cell(self, w, h=0, txt="", border=0, ln=0, align="L"):
        """Ячейка с автоматической подстройкой ширины"""
        required_width = self.get_string_width(txt) + 2  # +2 мм запаса
        effective_width = max(w, required_width)
        self.cell(effective_width, h, txt, border, ln, align)

# ============================================
# ФУНКЦИИ ДЛЯ РАБОТЫ С ДАННЫМИ
# ============================================

def translate_data(df):
    """Перевод названий столбцов и значений"""
    try:
        # Перевод названий столбцов
        df = df.rename(columns=lambda x: COLUMN_TRANSLATION.get(x.lower(), x))
        
        # Перевод названий стран
        if 'Страна' in df.columns:
            df['Страна'] = df['Страна'].map(COUNTRY_TRANSLATION).fillna(df['Страна'])
            
        return df
    except Exception as e:
        logger.error(f"Ошибка перевода данных: {e}")
        return df

def load_data(uploaded_file):
    """Загрузка и обработка данных из файла"""
    try:
        # Проверка на пустой файл
        if uploaded_file.size == 0:
            raise ValueError("Файл пустой")

        # Чтение содержимого файла
        file_content = uploaded_file.read()
        
        # Обработка CSV
        if uploaded_file.name.endswith('.csv'):
            encodings = ['utf-8', 'windows-1251', 'latin1', 'cp1252']
            for encoding in encodings:
                try:
                    df = pd.read_csv(BytesIO(file_content), encoding=encoding, on_bad_lines='warn')
                    if not df.empty:
                        break
                except Exception as e:
                    logger.warning(f"Ошибка чтения CSV с кодировкой {encoding}: {e}")
            else:
                raise ValueError("Не удалось прочитать CSV файл")
                
        # Обработка Excel
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            try:
                df = pd.read_excel(BytesIO(file_content), engine='openpyxl')
            except Exception as e:
                raise ValueError(f"Ошибка чтения Excel файла: {e}")
        else:
            raise ValueError("Поддерживаются только CSV и Excel файлы")

        # Проверка на пустые данные
        if df.empty:
            raise ValueError("Файл не содержит данных")

        # Проверка обязательных столбцов
        required_cols = {'country', 'points', 'price', 'variety'}
        available_cols = set(col.lower() for col in df.columns)
        missing_cols = required_cols - available_cols
        
        if missing_cols:
            raise ValueError(f"Отсутствуют обязательные столбцы: {missing_cols}")

        # Перевод и очистка данных
        df = translate_data(df)
        df = df.dropna(subset=['Рейтинг', 'Цена']).drop_duplicates()
        
        # Извлечение года из описания (если есть)
        if 'Описание' in df.columns:
            df['Год'] = df['Описание'].str.extract(r'(19|20\d{2})')[0]
        
        return df

    except Exception as e:
        logger.error(f"Ошибка загрузки данных: {e}")
        st.error(f"Ошибка загрузки данных: {str(e)}")
        return None

# ============================================
# ФУНКЦИИ ВИЗУАЛИЗАЦИИ
# ============================================

def create_summary_plot(data, variety):
    """Создание сводных графиков"""
    try:
        fig = plt.figure(figsize=(18, 14))
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.7])
        
        # График 1: Распределение рейтингов
        ax1 = plt.subplot(gs[0, 0])
        sns.boxplot(x='Страна', y='Рейтинг', data=data, ax=ax1)
        ax1.set_title(f'Распределение рейтингов ({variety})', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # График 2: Распределение цен
        ax2 = plt.subplot(gs[0, 1])
        sns.boxplot(x='Страна', y='Цена', data=data, ax=ax2)
        ax2.set_title(f'Распределение цен ({variety})', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        
        # График 3: Зависимость цены от рейтинга
        ax3 = plt.subplot(gs[1, 0])
        sns.scatterplot(x='Рейтинг', y='Цена', hue='Страна', data=data, ax=ax3, alpha=0.7)
        ax3.set_title('Зависимость цены от рейтинга', fontsize=12)
        
        # График 4: Распределение по годам (если есть данные)
        ax4 = plt.subplot(gs[1, 1])
        if 'Год' in data.columns:
            data['Год'] = pd.to_numeric(data['Год'], errors='coerce')
            year_counts = data.dropna(subset=['Год']).groupby(['Страна', 'Год']).size().reset_index(name='Количество')
            if not year_counts.empty:
                sns.lineplot(x='Год', y='Количество', hue='Страна', data=year_counts, ax=ax4)
                ax4.set_title('Распределение по годам', fontsize=12)
            else:
                ax4.text(0.5, 0.5, 'Нет данных по годам', ha='center', va='center')
                ax4.axis('off')
        else:
            ax4.text(0.5, 0.5, 'Нет данных о годе', ha='center', va='center')
            ax4.axis('off')
        
        # Текстовая аналитика
        ax5 = plt.subplot(gs[2, :])
        ax5.axis('off')
        
        # Собираем статистику
        stats_text = []
        for country in data['Страна'].unique():
            country_data = data[data['Страна'] == country]
            avg_rating = country_data['Рейтинг'].mean()
            avg_price = country_data['Цена'].mean()
            stats_text.append(f"{country}: Средний рейтинг {avg_rating:.1f}, Средняя цена ${avg_price:.1f}")
        
        ax5.text(0.05, 0.95, "\n".join(stats_text), 
                ha='left', va='top', fontsize=11,
                bbox={'facecolor': 'lightgray', 'alpha': 0.2, 'pad': 10})
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        logger.error(f"Ошибка создания графиков: {e}")
        st.error(f"Ошибка при создании визуализаций: {str(e)}")
        return None

def create_wordcloud(data, variety):
    """Генерация облака слов"""
    try:
        if 'Описание' not in data.columns:
            return None
            
        text = ' '.join(data['Описание'].dropna().astype(str))
        wordcloud = WordCloud(
            width=800, 
            height=400,
            background_color='white',
            stopwords=None,
            min_font_size=10
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        ax.set_title(f'Частые слова в описаниях {variety}', fontsize=12)
        return fig
    except Exception as e:
        logger.error(f"Ошибка создания облака слов: {e}")
        return None

def analyze_sentiment(data):
    """Анализ тональности описаний"""
    try:
        if 'Описание' not in data.columns:
            return None, None
            
        data['sentiment'] = data['Описание'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity if pd.notna(x) else 0)
        
        fig = px.box(
            data,
            x='Страна',
            y='sentiment',
            color='Страна',
            title='Распределение тональности описаний'
        )
        return fig, data
    except Exception as e:
        logger.error(f"Ошибка анализа тональности: {e}")
        return None, None

def create_geographical_analysis(data):
    """Создание географической карты"""
    try:
        if 'Провинция' not in data.columns or 'Регион' not in data.columns:
            return None, None
            
        region_stats = data.groupby(['Страна', 'Провинция', 'Регион']).agg({
            'Рейтинг': 'mean',
            'Цена': 'mean',
            'Сорт': 'count'
        }).reset_index()
        
        m = folium.Map(location=[20, 0], zoom_start=2)
        
        for _, row in region_stats.iterrows():
            folium.Marker(
                location=[np.random.uniform(-60, 70), np.random.uniform(-180, 180)],
                popup=f"{row['Регион']}<br>Рейтинг: {row['Рейтинг']:.1f}<br>Цена: ${row['Цена']:.1f}",
                tooltip=row['Регион']
            ).add_to(m)
            
        return m, region_stats
    except Exception as e:
        logger.error(f"Ошибка создания карты: {e}")
        return None, None

# ============================================
# PDF ОТЧЕТ
# ============================================

def generate_pdf_report(data, variety):
    """Генерация PDF отчета"""
    try:
        pdf = WinePDF()
        pdf.add_page()
        
        # Заголовок
        pdf.set_font(size=16)
        pdf.safe_cell(0, 10, f"Аналитический отчет: {variety}", ln=1, align="C")
        pdf.ln(10)
        
        # Основная статистика
        pdf.set_font(size=12)
        pdf.safe_cell(0, 10, "Основные статистики:", ln=1)
        
        stats = data.groupby('Страна').agg({
            'Рейтинг': ['mean', 'median', 'std'],
            'Цена': ['mean', 'median', 'std'],
            'Сорт': 'count'
        })
        
        for country in stats.index:
            text = (f"{country}: "
                   f"Рейтинг {stats.loc[country, ('Рейтинг', 'mean')]:.1f}±{stats.loc[country, ('Рейтинг', 'std')]:.1f}, "
                   f"Цена ${stats.loc[country, ('Цена', 'mean')]:.1f}±{stats.loc[country, ('Цена', 'std')]:.1f}, "
                   f"Найдено {int(stats.loc[country, ('Сорт', 'count')])} образцов")
            pdf.safe_cell(0, 8, text, ln=1)
        
        # Топ вин
        pdf.ln(5)
        pdf.safe_cell(0, 10, "Топ-5 самых дорогих вин:", ln=1)
        
        top_wines = data.nlargest(5, 'Цена')[['Винодельня', 'Страна', 'Рейтинг', 'Цена']]
        for _, row in top_wines.iterrows():
            text = f"{row['Винодельня']} ({row['Страна']}): {row['Рейтинг']} баллов, ${row['Цена']:.2f}"
            pdf.safe_cell(0, 8, text, ln=1)
        
        # Сохранение во временный файл
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        pdf.output(temp_file.name)
        return temp_file.name
        
    except Exception as e:
        logger.error(f"Ошибка генерации PDF: {e}")
        return None

# ============================================
# ОСНОВНОЙ ИНТЕРФЕЙС
# ============================================

def main():
    # Заголовок приложения
    st.title("🍷 Винный аналитик Pro")
    
    # Инструкция
    with st.expander("📌 Инструкция по использованию", expanded=True):
        st.markdown("""
        ### Как пользоваться приложением:
        1. **Загрузите файл** с данными о винах (CSV или Excel)
        2. **Выберите страны** и **сорт вина** для анализа
        3. **Нажмите "Анализировать"** для просмотра результатов
        4. **Сгенерируйте PDF отчет** при необходимости

        ### Требования к данным:
        - Файл должен содержать обязательные столбцы:  
          `country`, `points`, `price`, `variety`
        - Поддерживаемые форматы: CSV (UTF-8, Windows-1251), Excel (XLSX)
        """)
    
    # Загрузка данных
    uploaded_file = st.sidebar.file_uploader(
        "📤 Загрузите файл с данными",
        type=["csv", "xlsx"],
        help="Выберите CSV или Excel файл с данными о винах"
    )
    
    if uploaded_file is not None:
        with st.spinner("Загружаем и обрабатываем данные..."):
            data = load_data(uploaded_file)
        
        if data is not None:
            st.success(f"✅ Успешно загружено {len(data)} записей")
            
            # Выбор параметров анализа
            st.sidebar.subheader("Параметры анализа")
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                countries = st.multiselect(
                    "Выберите страны",
                    options=sorted(data['Страна'].unique()),
                    default=data['Страна'].unique()[:2]
                )
            with col2:
                variety = st.selectbox(
                    "Выберите сорт",
                    options=sorted(data['Сорт'].unique())
                )
            
            # Кнопка анализа
            if st.sidebar.button("🔍 Анализировать", type="primary"):
                with st.spinner("Выполняем анализ..."):
                    # Фильтрация данных
                    filtered_data = data[
                        data['Страна'].isin(countries) & 
                        data['Сорт'].str.contains(variety, case=False, na=False)
                    ]
                    
                    if len(filtered_data) == 0:
                        st.warning("⚠️ Не найдено данных для выбранных параметров")
                    else:
                        # Создание вкладок
                        tab1, tab2, tab3, tab4, tab5 = st.tabs([
                            "📊 Основная статистика", 
                            "📈 Графики", 
                            "📝 Текстовая аналитика", 
                            "🌍 География", 
                            "📄 Отчет"
                        ])
                        
                        with tab1:
                            st.header("Основная статистика")
                            stats = filtered_data.groupby('Страна').agg({
                                'Рейтинг': ['mean', 'median', 'std', 'count'],
                                'Цена': ['mean', 'median', 'std']
                            })
                            st.dataframe(stats.style.format({
                                ('Рейтинг', 'mean'): '{:.1f}',
                                ('Рейтинг', 'median'): '{:.1f}',
                                ('Рейтинг', 'std'): '{:.2f}',
                                ('Цена', 'mean'): '${:.2f}',
                                ('Цена', 'median'): '${:.2f}',
                                ('Цена', 'std'): '${:.2f}'
                            }))
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("Топ-5 по цене")
                                st.dataframe(filtered_data.nlargest(5, 'Цена')[['Винодельня', 'Страна', 'Рейтинг', 'Цена']])
                            with col2:
                                st.subheader("Топ-5 по рейтингу")
                                st.dataframe(filtered_data.nlargest(5, 'Рейтинг')[['Винодельня', 'Страна', 'Рейтинг', 'Цена']])
                        
                        with tab2:
                            st.header("Визуализация данных")
                            fig = create_summary_plot(filtered_data, variety)
                            if fig:
                                st.pyplot(fig)
                            else:
                                st.error("Не удалось создать графики")
                            
                            st.plotly_chart(px.scatter(
                                filtered_data,
                                x='Рейтинг',
                                y='Цена',
                                color='Страна',
                                hover_data=['Винодельня', 'Провинция'],
                                title=f'Интерактивный анализ {variety}',
                                size_max=15
                            ), use_container_width=True)
                        
                        with tab3:
                            st.header("Анализ текста")
                            wc_fig = create_wordcloud(filtered_data, variety)
                            if wc_fig:
                                st.pyplot(wc_fig)
                            else:
                                st.warning("Нет данных для создания облака слов")
                            
                            sentiment_fig, sentiment_data = analyze_sentiment(filtered_data)
                            if sentiment_fig:
                                st.plotly_chart(sentiment_fig)
                                st.subheader("Примеры описаний")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("Самые положительные:")
                                    positive = sentiment_data.nlargest(3, 'sentiment')['Описание']
                                    for i, desc in enumerate(positive, 1):
                                        st.write(f"{i}. {desc[:200]}...")
                                with col2:
                                    st.write("Самые отрицательные:")
                                    negative = sentiment_data.nsmallest(3, 'sentiment')['Описание']
                                    for i, desc in enumerate(negative, 1):
                                        st.write(f"{i}. {desc[:200]}...")
                            else:
                                st.warning("Невозможно проанализировать тональность")
                        
                        with tab4:
                            st.header("Географическое распределение")
                            map_data, region_stats = create_geographical_analysis(filtered_data)
                            if map_data:
                                st_folium(map_data, width=800, height=500)
                                st.dataframe(region_stats.sort_values('Рейтинг', ascending=False))
                            else:
                                st.warning("Недостаточно данных для географического анализа")
                        
                        with tab5:
                            st.header("Генерация отчета")
                            if st.button("🖨️ Создать PDF отчет"):
                                with st.spinner("Формируем отчет..."):
                                    pdf_path = generate_pdf_report(filtered_data, variety)
                                    if pdf_path:
                                        with open(pdf_path, "rb") as f:
                                            st.download_button(
                                                label="⬇️ Скачать отчет",
                                                data=f,
                                                file_name=f"wine_analysis_{variety}_{datetime.now().strftime('%Y%m%d')}.pdf",
                                                mime="application/pdf"
                                            )
                                        os.unlink(pdf_path)
                                    else:
                                        st.error("Не удалось создать PDF отчет")

# Запуск приложения
if __name__ == "__main__":
    main()
