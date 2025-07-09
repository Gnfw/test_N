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
import warnings
from PIL import Image

# Фильтрация предупреждений
warnings.filterwarnings("ignore", category=UserWarning, module="fpdf")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# ============================================
# НАСТРОЙКА ПРИЛОЖЕНИЯ
# ============================================

# Конфигурация логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Настройки страницы Streamlit
st.set_page_config(
    page_title="🍷 Лучшая аналитическая система",
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
    'Argentina': 'Аргентина',
    'Австралия': 'Австралия',
    'Новая Зеландия': 'Новая Зеландия'
}

# Путь к шрифту с поддержкой кириллицы (замените на реальный путь к файлу шрифта)
FONT_PATH = "arial.ttf"

# ============================================
# КЛАСС ДЛЯ ГЕНЕРАЦИИ PDF
# ============================================

class WinePDF(FPDF):
    """Класс для создания PDF отчетов с поддержкой кириллицы"""
    def __init__(self):
        super().__init__()
        try:
            # Добавляем шрифт с поддержкой кириллицы
            self.add_font('Arial', '', 'arial.ttf', uni=True)
            self.add_font('Arial', 'B', 'arialbd.ttf', uni=True)
            self.add_font('Arial', 'I', 'ariali.ttf', uni=True)
            self.add_font('Arial', 'BI', 'arialbi.ttf', uni=True)
            self.set_font("Arial", size=10)
        except Exception as e:
            logger.error(f"Ошибка загрузки шрифта Arial: {e}")
            try:
                # Попробуем использовать DejaVu - часто предустановлен
                self.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
                self.set_font("DejaVu", size=10)
            except:
                # Если ничего не работает, используем стандартный шрифт (но кириллица не будет отображаться)
                self.set_font("helvetica", size=10)
                logger.error("Не удалось загрузить шрифт с поддержкой кириллицы. PDF будет без русских символов.")
        
    def header(self):
        # Логотип и заголовок
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Лучшая аналитическая система (для корочки)', 0, 1, 'C')
        self.ln(5)
        
    def footer(self):
        # Номер страницы
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Страница {self.page_no()}', 0, 0, 'C')
        
    def safe_cell(self, w, h=0, txt="", border=0, ln=0, align="L"):
        """Ячейка с автоматической подстройкой ширины"""
        required_width = self.get_string_width(txt) + 2
        effective_width = max(w, required_width)
        self.cell(effective_width, h, txt, border, ln, align)
        
    def add_section_title(self, title, level=1):
        """Добавление заголовка раздела"""
        if level == 1:
            self.set_font("Arial", 'B', 14)
            self.cell(0, 10, title, ln=1)
            self.ln(2)
        elif level == 2:
            self.set_font("Arial", 'B', 12)
            self.cell(0, 8, title, ln=1)
            self.ln(1)
        else:
            self.set_font("Arial", 'B', 10)
            self.cell(0, 6, title, ln=1)
        self.set_font("Arial", size=10)
        
    def add_plot(self, fig, title=None, width=180):
        """Добавление графика в PDF"""
        if fig is None:
            return
            
        try:
            # Сохраняем график во временный файл
            temp_img = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            fig.savefig(temp_img.name, bbox_inches='tight', dpi=100)
            plt.close(fig)
            
            # Добавляем изображение в PDF
            if title:
                self.add_section_title(title, level=3)
            self.image(temp_img.name, w=width)
            self.ln(5)
            
            # Удаляем временный файл
            os.unlink(temp_img.name)
        except Exception as e:
            logger.error(f"Ошибка добавления графика в PDF: {e}")

# ============================================
# ФУНКЦИИ ДЛЯ РАБОТЫ С ДАННЫМИ
# ============================================

def translate_data(df):
    """Перевод названий столбцов и значений"""
    try:
        # Перевод названий столбцов
        df = df.rename(columns=COLUMN_TRANSLATION)
        
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
        if uploaded_file is None or uploaded_file.size == 0:
            raise ValueError("Файл пустой или не загружен")

        file_content = uploaded_file.read()
        
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
                
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            try:
                df = pd.read_excel(BytesIO(file_content), engine='openpyxl')
            except Exception as e:
                raise ValueError(f"Ошибка чтения Excel файла: {e}")
        else:
            raise ValueError("Поддерживаются только CSV и Excel файлы")

        if df.empty:
            raise ValueError("Файл не содержит данных")

        required_cols = {'country', 'points', 'price', 'variety'}
        available_cols = set(col.lower() for col in df.columns)
        missing_cols = required_cols - available_cols
        
        if missing_cols:
            raise ValueError(f"Отсутствуют обязательные столбцы: {missing_cols}")

        df = translate_data(df)
        df = df.dropna(subset=['Рейтинг', 'Цена']).drop_duplicates()
        
        if 'Описание' in df.columns:
            df['Год'] = df['Описание'].str.extract(r'(19|20\d{2})')[0]
            df['Год'] = pd.to_numeric(df['Год'], errors='coerce')
        
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
        
        ax1 = plt.subplot(gs[0, 0])
        sns.boxplot(x='Страна', y='Рейтинг', data=data, ax=ax1)
        ax1.set_title(f'Распределение рейтингов ({variety})', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        ax2 = plt.subplot(gs[0, 1])
        sns.boxplot(x='Страна', y='Цена', data=data, ax=ax2)
        ax2.set_title(f'Распределение цен ({variety})', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        
        ax3 = plt.subplot(gs[1, 0])
        sns.scatterplot(x='Рейтинг', y='Цена', hue='Страна', data=data, ax=ax3, alpha=0.7)
        ax3.set_title('Зависимость цены от рейтинга', fontsize=12)
        
        ax4 = plt.subplot(gs[1, 1])
        if 'Год' in data.columns and not data['Год'].isna().all():
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
        
        ax5 = plt.subplot(gs[2, :])
        ax5.axis('off')
        
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
        
        fig = plt.figure(figsize=(10, 6))
        sns.boxplot(x='Страна', y='sentiment', data=data)
        plt.title('Распределение тональности описаний')
        plt.xticks(rotation=45)
        plt.tight_layout()
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
        
        fig = plt.figure(figsize=(12, 8))
        sns.scatterplot(
            x='Рейтинг', 
            y='Цена',
            size='Сорт',
            hue='Страна',
            data=region_stats,
            sizes=(20, 200),
            alpha=0.7
        )
        plt.title('Географическое распределение (размер точки = количество образцов)')
        plt.grid(True)
        plt.tight_layout()
        
        return fig, region_stats
    except Exception as e:
        logger.error(f"Ошибка создания карты: {e}")
        return None, None

# ============================================
# ДОПОЛНИТЕЛЬНЫЕ ФУНКЦИИ АНАЛИЗА
# ============================================

def create_popular_varieties_plot(data):
    """График популярных сортов по регионам"""
    try:
        region_variety = data.groupby(['Страна', 'Провинция', 'Сорт']).size().reset_index(name='Количество')
        top_by_region = region_variety.sort_values(['Страна', 'Провинция', 'Количество'], ascending=[True, True, False])
        top_by_region = top_by_region.groupby(['Страна', 'Провинция']).head(3)
        
        fig = plt.figure(figsize=(12, 8))
        sns.barplot(
            x='Количество',
            y='Провинция',
            hue='Сорт',
            data=top_by_region,
            palette='viridis'
        )
        plt.title('Топ сортов по регионам')
        plt.tight_layout()
        return fig, top_by_region
    except Exception as e:
        logger.error(f"Ошибка создания графика популярных сортов: {e}")
        return None, None

def create_price_stats_plot(data):
    """График статистики цен по регионам"""
    try:
        price_stats = data.groupby(['Страна', 'Провинция'])['Цена'].agg(
            ['mean', 'median', 'min', 'max']
        ).reset_index()
        
        fig = plt.figure(figsize=(12, 8))
        price_stats_melt = price_stats.melt(
            id_vars=['Страна', 'Провинция'], 
            value_vars=['mean', 'median', 'min', 'max'],
            var_name='Метрика',
            value_name='Цена'
        )
        
        sns.barplot(
            x='Провинция',
            y='Цена',
            hue='Метрика',
            data=price_stats_melt
        )
        plt.title('Статистика цен по регионам')
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig, price_stats
    except Exception as e:
        logger.error(f"Ошибка создания графика статистики цен: {e}")
        return None, None

def create_rating_by_region_plot(data):
    """График рейтинга по регионам"""
    try:
        fig = plt.figure(figsize=(12, 8))
        sns.boxplot(
            x='Провинция',
            y='Рейтинг',
            hue='Страна',
            data=data
        )
        plt.title('Распределение рейтингов по регионам')
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"Ошибка создания графика рейтинга по регионам: {e}")
        return None

def create_correlation_plot(data):
    """График корреляции цены и рейтинга"""
    try:
        fig = plt.figure(figsize=(10, 8))
        sns.regplot(
            x='Рейтинг',
            y='Цена',
            data=data,
            scatter_kws={'alpha':0.3}
        )
        plt.title('Корреляция между ценой и рейтингом')
        plt.tight_layout()
        
        corr = data[['Рейтинг', 'Цена']].corr().iloc[0,1]
        
        return fig, corr
    except Exception as e:
        logger.error(f"Ошибка создания графика корреляции: {e}")
        return None, None

def create_price_by_region_plot(data):
    """График цен по регионам"""
    try:
        fig = plt.figure(figsize=(12, 8))
        sns.boxplot(
            x='Провинция',
            y='Цена',
            hue='Страна',
            data=data
        )
        plt.title('Распределение цен по регионам')
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"Ошибка создания графика цен по регионам: {e}")
        return None

# ============================================
# ГЕНЕРАЦИЯ PDF ОТЧЕТА
# ============================================

def generate_pdf_report(data, variety):
    """Генерация PDF отчета"""
    try:
        pdf = WinePDF()
        pdf.add_page()
        
        # Заголовок отчета
        pdf.set_font("Arial", size=16, style='B')
        pdf.cell(0, 10, f"Аналитический отчет: {variety}", ln=1, align="C")
        pdf.ln(10)
        
        # 1. Основная статистика
        pdf.add_section_title("1. Основная статистика", level=1)
        
        stats = data.groupby('Страна').agg({
            'Рейтинг': ['mean', 'median', 'std', 'count'],
            'Цена': ['mean', 'median', 'std', 'min', 'max']
        }).reset_index()
        
        stats_text = []
        for _, row in stats.iterrows():
            stats_text.append(
                f"{row['Страна']}: "
                f"Рейтинг {row[('Рейтинг', 'mean')]:.1f}±{row[('Рейтинг', 'std')]:.1f}, "
                f"Цена ${row[('Цена', 'mean')]:.1f}±${row[('Цена', 'std')]:.1f}, "
                f"Образцов: {int(row[('Рейтинг', 'count')])}"
            )
        
        for line in stats_text:
            pdf.safe_cell(0, 8, line, ln=1)
        pdf.ln(5)
        
        # Топ вин
        pdf.add_section_title("Топ-5 самых дорогих вин:", level=2)
        top_wines = data.nlargest(5, 'Цена')[['Винодельня', 'Страна', 'Рейтинг', 'Цена']]
        for _, row in top_wines.iterrows():
            text = f"{row['Винодельня']} ({row['Страна']}): {row['Рейтинг']} баллов, ${row['Цена']:.2f}"
            pdf.safe_cell(0, 8, text, ln=1)
        
        pdf.ln(5)
        pdf.add_section_title("Топ-5 с наивысшим рейтингом:", level=2)
        top_rated = data.nlargest(5, 'Рейтинг')[['Винодельня', 'Страна', 'Рейтинг', 'Цена']]
        for _, row in top_rated.iterrows():
            text = f"{row['Винодельня']} ({row['Страна']}): {row['Рейтинг']} баллов, ${row['Цена']:.2f}"
            pdf.safe_cell(0, 8, text, ln=1)
        
        # 2. Основные графики
        pdf.add_page()
        pdf.add_section_title("2. Основные графики анализа", level=1)
        
        summary_fig = create_summary_plot(data, variety)
        pdf.add_plot(summary_fig, "Сводные графики анализа")
        
        # 3. Текстовая аналитика
        pdf.add_page()
        pdf.add_section_title("3. Текстовая аналитика", level=1)
        
        wc_fig = create_wordcloud(data, variety)
        pdf.add_plot(wc_fig, "Облако слов из описаний вин")
        
        sentiment_fig, sentiment_data = analyze_sentiment(data)
        pdf.add_plot(sentiment_fig, "Анализ тональности описаний")
        
        if sentiment_data is not None:
            pdf.add_section_title("Примеры описаний:", level=2)
            
            positive = sentiment_data.nlargest(3, 'sentiment')['Описание']
            pdf.safe_cell(0, 8, "Самые положительные описания:", ln=1)
            for i, desc in enumerate(positive, 1):
                pdf.multi_cell(0, 8, f"{i}. {desc[:150]}...", ln=1)
            
            pdf.ln(2)
            
            negative = sentiment_data.nsmallest(3, 'sentiment')['Описание']
            pdf.safe_cell(0, 8, "Самые отрицательные описания:", ln=1)
            for i, desc in enumerate(negative, 1):
                pdf.multi_cell(0, 8, f"{i}. {desc[:150]}...", ln=1)
        
        # 4. Географический анализ
        pdf.add_page()
        pdf.add_section_title("4. Географический анализ", level=1)
        
        geo_fig, geo_stats = create_geographical_analysis(data)
        pdf.add_plot(geo_fig, "Географическое распределение")
        
        if geo_stats is not None:
            pdf.add_section_title("Статистика по регионам:", level=2)
            geo_stats_display = geo_stats.nlargest(10, 'Рейтинг')[['Регион', 'Страна', 'Рейтинг', 'Цена', 'Сорт']]
            geo_stats_display.columns = ['Регион', 'Страна', 'Ср. рейтинг', 'Ср. цена', 'Количество']
            
            col_widths = [50, 30, 30, 30, 30]
            headers = geo_stats_display.columns.tolist()
            data_rows = geo_stats_display.values.tolist()
            
            pdf.set_font(style='B')
            for i, header in enumerate(headers):
                pdf.cell(col_widths[i], 10, str(header), border=1)
            pdf.ln()
            
            pdf.set_font(style='')
            for row in data_rows:
                for i, item in enumerate(row):
                    if i == 3:
                        pdf.cell(col_widths[i], 10, f"${float(item):.2f}", border=1)
                    elif i == 2:
                        pdf.cell(col_widths[i], 10, f"{float(item):.1f}", border=1)
                    else:
                        pdf.cell(col_widths[i], 10, str(item), border=1)
                pdf.ln()
        
        # 5. Дополнительные анализы
        pdf.add_page()
        pdf.add_section_title("5. Дополнительные анализы", level=1)
        
        pop_fig, pop_data = create_popular_varieties_plot(data)
        pdf.add_plot(pop_fig, "Популярные сорта по регионам")
        
        price_fig, price_stats = create_price_stats_plot(data)
        pdf.add_plot(price_fig, "Статистика цен по регионам")
        
        corr_fig, corr = create_correlation_plot(data)
        pdf.add_plot(corr_fig, f"Корреляция между ценой и рейтингом (r = {corr:.2f})")
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        pdf.output(temp_file.name)
        return temp_file.name
        
    except Exception as e:
        logger.error(f"Ошибка генерации PDF: {e}")
        return None

# ============================================
# ИНТЕРФЕЙС ПОЛЬЗОВАТЕЛЯ
# ============================================

def show_basic_stats(filtered_data):
    st.header("Основная статистика")
    
    stats = filtered_data.groupby('Страна').agg({
        'Рейтинг': ['mean', 'median', 'std', 'count'],
        'Цена': ['mean', 'median', 'std', 'min', 'max']
    })
    st.dataframe(stats.style.format({
        ('Рейтинг', 'mean'): '{:.1f}',
        ('Рейтинг', 'median'): '{:.1f}',
        ('Рейтинг', 'std'): '{:.2f}',
        ('Цена', 'mean'): '${:.2f}',
        ('Цена', 'median'): '${:.2f}',
        ('Цена', 'std'): '${:.2f}',
        ('Цена', 'min'): '${:.2f}',
        ('Цена', 'max'): '${:.2f}'
    }))
    
    show_popular_varieties(filtered_data)
    show_price_stats(filtered_data)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Топ-5 по цене")
        st.dataframe(filtered_data.nlargest(5, 'Цена')[['Винодельня', 'Страна', 'Рейтинг', 'Цена']])
    with col2:
        st.subheader("Топ-5 по рейтингу")
        st.dataframe(filtered_data.nlargest(5, 'Рейтинг')[['Винодельня', 'Страна', 'Рейтинг', 'Цена']])

def show_popular_varieties(data):
    st.subheader("Популярные сорта по регионам")
    region_variety = data.groupby(['Страна', 'Провинция', 'Сорт']).size().reset_index(name='Количество')
    top_by_region = region_variety.sort_values(['Страна', 'Провинция', 'Количество'], ascending=[True, True, False])
    
    for country in top_by_region['Страна'].unique():
        st.write(f"**{country}**")
        country_data = top_by_region[top_by_region['Страна'] == country]
        st.dataframe(country_data.groupby('Провинция').first().reset_index()[['Провинция', 'Сорт', 'Количество']])

def show_price_stats(data):
    st.subheader("Детальная статистика цен по регионам")
    price_stats = data.groupby(['Страна', 'Провинция'])['Цена'].agg(
        ['mean', 'median', 'min', 'max', 'std', 'count']
    ).reset_index()
    st.dataframe(price_stats.style.format({
        'mean': '${:.2f}', 'median': '${:.2f}', 
        'min': '${:.2f}', 'max': '${:.2f}', 
        'std': '${:.2f}'
    }))

def show_visualizations(filtered_data, variety):
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
    
    show_rating_by_region(filtered_data)
    show_price_by_region(filtered_data)
    show_correlation_analysis(filtered_data)

def show_rating_by_region(data):
    st.subheader("Зависимость рейтинга от региона")
    fig = create_rating_by_region_plot(data)
    if fig:
        st.pyplot(fig)
    else:
        st.error("Не удалось создать график")

def show_price_by_region(data):
    st.subheader("Зависимость цены от региона производства")
    fig = create_price_by_region_plot(data)
    if fig:
        st.pyplot(fig)
    else:
        st.error("Не удалось создать график")

def show_correlation_analysis(data):
    st.subheader("Корреляция между ценой и рейтингом")
    fig, corr = create_correlation_plot(data)
    if fig:
        st.pyplot(fig)
        st.write(f"Коэффициент корреляции: {corr:.2f}")
        
        if abs(corr) > 0.3:
            st.write("✅ Наблюдается заметная корреляция")
        else:
            st.write("❌ Корреляция слабая или отсутствует")
    else:
        st.error("Не удалось создать график корреляции")

def show_text_analysis(filtered_data, variety):
    st.header("Анализ текста")
    
    wc_fig = create_wordcloud(filtered_data, variety)
    if wc_fig:
        st.pyplot(wc_fig)
    else:
        st.warning("Нет данных для создания облака слов")
    
    sentiment_fig, sentiment_data = analyze_sentiment(filtered_data)
    if sentiment_fig:
        st.pyplot(sentiment_fig)
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

def show_geographical_analysis(filtered_data):
    st.header("Географическое распределение")
    map_data, region_stats = create_geographical_analysis(filtered_data)
    if map_data:
        st.pyplot(map_data)
        st.dataframe(region_stats.sort_values('Рейтинг', ascending=False))
    else:
        st.warning("Недостаточно данных для географического анализа")

def show_report_generation(filtered_data, variety):
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

def main():
    st.title("🍷 Лучшая аналитическая (для получения корочки) система")
    st.markdown("### Которая анализирует то, что лучше пить, чем анализировать")
    
    # Инструкция по установке шрифта
    st.sidebar.markdown("""
    ### Для корректной работы:
    1. Выбери страны (по умолчанию - Австралия и Новая Зеландия)
    2. Выбери сорт вина. Если в данных нет вина - будет ошибка
    3. Дальше - просто смотри на результат.
    4. На последней вкладке можно создать PDF и после создания его сохранить.
    5. Пиши замечания и что тебе вообще надо. Я тут накидал всего подряд =)
    """)
    
    if 'analyzed' not in st.session_state:
        st.session_state.analyzed = False
        st.session_state.filtered_data = None
    
    with st.expander("📌 Инструкция по использованию", expanded=True):
        st.markdown("""
        ### Как пользоваться приложением:
        1. **Загрузите файл** с данными о винах (CSV или Excel)
        2. **Выберите страны** и **сорт вина** для анализа
        3. **Нажмите "Анализировать"** для просмотра результатов
        4. **Сгенерируйте PDF отчет** при необходимости
        """)
    
    uploaded_file = st.sidebar.file_uploader(
        "📤 Загрузите файл с данными",
        type=["csv", "xlsx"],
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        with st.spinner("Загружаем и обрабатываем данные..."):
            data = load_data(uploaded_file)
        
        if data is not None:
            st.success(f"✅ Успешно загружено {len(data)} записей")
            
            st.sidebar.subheader("Параметры анализа")
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                all_countries = sorted(data['Страна'].unique())
                default_countries = []
                if 'Австралия' in all_countries:
                    default_countries.append('Австралия')
                if 'Новая Зеландия' in all_countries:
                    default_countries.append('Новая Зеландия')
                
                if not default_countries and len(all_countries) >= 2:
                    default_countries = all_countries[:2]
                
                countries = st.multiselect(
                    "Выберите страны",
                    options=all_countries,
                    default=default_countries
                )

            with col2:
                variety = st.selectbox(
                    "Выберите сорт",
                    options=sorted(data['Сорт'].unique())
                )
            
            st.sidebar.subheader("Дополнительные фильтры")
            min_rating = st.sidebar.slider(
                "Минимальный рейтинг",
                min_value=80, max_value=100, value=85
            )
            max_price = st.sidebar.slider(
                "Максимальная цена ($)",
                min_value=0, max_value=1000, value=500
            )
            
            if st.sidebar.button("🔍 Анализировать", type="primary"):
                with st.spinner("Выполняем анализ..."):
                    st.session_state.filtered_data = data[
                        (data['Страна'].isin(countries)) & 
                        (data['Сорт'].str.contains(variety, case=False, na=False)) &
                        (data['Рейтинг'] >= min_rating) &
                        (data['Цена'] <= max_price)
                    ].copy()
                    st.session_state.analyzed = True
            
            if st.session_state.analyzed and st.session_state.filtered_data is not None:
                filtered_data = st.session_state.filtered_data
                
                if len(filtered_data) == 0:
                    st.warning("⚠️ Не найдено данных для выбранных параметров")
                else:
                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "📊 Основная статистика", 
                        "📈 Графики", 
                        "📝 Текстовая аналитика", 
                        "🌍 География", 
                        "📄 Отчет"
                    ])
                    
                    with tab1:
                        show_basic_stats(filtered_data)
                    
                    with tab2:
                        show_visualizations(filtered_data, variety)
                    
                    with tab3:
                        show_text_analysis(filtered_data, variety)
                    
                    with tab4:
                        show_geographical_analysis(filtered_data)
                    
                    with tab5:
                        show_report_generation(filtered_data, variety)

if __name__ == "__main__":
    # Удаляем конфликтующие библиотеки перед запуском
    os.system("pip uninstall --yes pypdf")
    os.system("pip install --upgrade fpdf2")
    
    main()
