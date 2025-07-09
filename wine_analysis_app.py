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

def translate_data(df):
    """Переводит названия столбцов в DataFrame"""
    df = df.rename(columns=column_translation)
    if 'Страна' in df.columns:
        df['Страна'] = df['Страна'].replace(country_translation)
    return df

def translate_stats(df):
    """Переводит статистическую таблицу с обработкой MultiIndex"""
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
                translated.index.name = stat_translation.get(translated.index.name, translated.index.name)
        
        if isinstance(translated.columns, pd.MultiIndex):
            new_columns = []
            for level in translated.columns.levels:
                if level.name in column_translation:
                    new_columns.append(level.map(column_translation))
                else:
                    new_columns.append(level)
            translated.columns = pd.MultiIndex.from_arrays(new_columns, names=translated.columns.names)
        else:
            if translated.columns.name in column_translation:
                translated.columns = translated.columns.map(lambda x: column_translation.get(x, x))
                translated.columns.name = column_translation.get(translated.columns.name, translated.columns.name)
        
        return translated
    except Exception as e:
        logger.error(f"Ошибка перевода статистики: {str(e)}")
        return df

def get_file_hash(uploaded_file):
    """Генерирует хеш содержимого файла"""
    uploaded_file.seek(0)
    return hashlib.md5(uploaded_file.read()).hexdigest()

def extract_year_from_description(description):
    """Извлекает год из текста описания"""
    try:
        if pd.isna(description):
            return None
        matches = re.findall(r'(19|20\d{2})', str(description))
        return int(matches[0]) if matches else None
    except:
        return None

def load_data(uploaded_file, use_cache=True):
    """Загружает и обрабатывает данные"""
    try:
        file_hash = get_file_hash(uploaded_file)
        cache_key = f"wine_data_{file_hash}"
        
        if not use_cache or cache_key not in st.session_state:
            uploaded_file.seek(0)
            
            # Автоматическое определение формата
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Неподдерживаемый формат файла. Используйте CSV или Excel.")
                return None
                
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            df = translate_data(df)
            
            # Извлечение года из описания
            if 'Описание' in df.columns:
                df['Год'] = df['Описание'].apply(extract_year_from_description)
            
            # Обработка пропущенных значений
            df = df.dropna(subset=['Рейтинг', 'Цена'])
            df = df.drop_duplicates()
            
            if use_cache:
                st.session_state[cache_key] = df
            return df
        return st.session_state[cache_key]
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {str(e)}")
        logger.exception("Ошибка при загрузке данных")
        return None

def create_summary_plot(data, filtered_data, variety):
    """Создает комплексный график с анализом"""
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.5])
    
    # График 1: Распределение рейтингов
    ax1 = plt.subplot(gs[0, 0])
    sns.boxplot(x='Страна', y='Рейтинг', data=filtered_data, ax=ax1)
    ax1.set_title(f'Распределение рейтингов {variety}')
    ax1.set_xlabel('Страна')
    ax1.set_ylabel('Рейтинг')
    
    # График 2: Распределение цен
    ax2 = plt.subplot(gs[0, 1])
    sns.boxplot(x='Страна', y='Цена', data=filtered_data, ax=ax2)
    ax2.set_title(f'Распределение цен {variety}')
    ax2.set_xlabel('Страна')
    ax2.set_ylabel('Цена ($)')
    
    # График 3: Зависимость цены от рейтинга
    ax3 = plt.subplot(gs[1, 0])
    sns.scatterplot(x='Рейтинг', y='Цена', hue='Страна', data=filtered_data, ax=ax3, alpha=0.7)
    ax3.set_title('Зависимость цены от рейтинга')
    ax3.set_xlabel('Рейтинг')
    ax3.set_ylabel('Цена ($)')
    
    # График 4: Количество образцов по годам
    ax4 = plt.subplot(gs[1, 1])
    if 'Год' in filtered_data.columns:
        filtered_data['Год'] = pd.to_numeric(filtered_data['Год'], errors='coerce')
        filtered_data = filtered_data.dropna(subset=['Год'])
        
        year_counts = filtered_data.groupby(['Страна', 'Год']).size().reset_index(name='Количество')
        
        if not year_counts.empty:
            pivot_data = year_counts.pivot(index='Год', columns='Страна', values='Количество')
            pivot_data.plot(kind='bar', stacked=True, ax=ax4)
            ax4.set_title('Распределение по годам')
            ax4.set_xlabel('Год')
            ax4.set_ylabel('Количество образцов')
        else:
            ax4.axis('off')
            ax4.text(0.5, 0.5, 'Нет данных по годам', ha='center', va='center', fontsize=12)
    else:
        ax4.axis('off')
        ax4.text(0.5, 0.5, 'Отсутствует столбец с годом', ha='center', va='center', fontsize=12)
    
    # Текстовая аналитика
    ax5 = plt.subplot(gs[2, :])
    ax5.axis('off')
    
    # Собираем аналитику
    countries = filtered_data['Страна'].unique()
    stats_dict = {}
    corr_dict = {}
    
    for country in countries:
        country_data = filtered_data[filtered_data['Страна'] == country]
        stats_dict[country] = country_data[['Рейтинг', 'Цена']].describe()
        corr_dict[country] = country_data[['Рейтинг', 'Цена']].corr().iloc[0,1]
    
    if len(countries) == 2:
        t_stat, p_value = stats.ttest_ind(
            filtered_data[filtered_data['Страна'] == countries[0]]['Рейтинг'],
            filtered_data[filtered_data['Страна'] == countries[1]]['Рейтинг'],
            equal_var=False
        )
    
    report_text = f"""
    ДЕТАЛЬНЫЙ АНАЛИЗ {variety.upper()}
    
    Общее количество образцов:
    - Всего: {len(filtered_data)}"""
    
    for country in countries:
        report_text += f"\n    - {country}: {len(filtered_data[filtered_data['Страна'] == country])}"
    
    report_text += "\n\nОсновные статистики:"
    
    for country in countries:
        report_text += f"""
    [{country}]
    - Средний рейтинг: {stats_dict[country].loc['mean', 'Рейтинг']:.1f}
    - Средняя цена: ${stats_dict[country].loc['mean', 'Цена']:.1f}
    - Корреляция цена/рейтинг: {corr_dict[country]:.2f}"""
    
    if len(countries) == 2:
        report_text += f"""
    
    Статистическая значимость различий:
    - p-значение: {p_value:.4f}
    - Заключение: {'различия значимы' if p_value < 0.05 else 'различия не значимы'}"""
    
    report_text += "\n\nВыводы:"
    
    if len(countries) == 2:
        report_text += f"""
    1. {countries[1]} {variety} в среднем {'превосходят' if stats_dict[countries[1]].loc['mean', 'Рейтинг'] > stats_dict[countries[0]].loc['mean', 'Рейтинг'] else 'уступают'} {countries[0]}
    2. Ценовая политика {'более агрессивна' if stats_dict[countries[1]].loc['mean', 'Цена'] > stats_dict[countries[0]].loc['mean', 'Цена'] else 'более консервативна'} в {countries[1]}
    3. Связь между ценой и качеством {'сильнее' if abs(corr_dict[countries[1]]) > abs(corr_dict[countries[0]]) else 'слабее'} в {countries[1]}"""
    
    ax5.text(0.05, 0.95, textwrap.dedent(report_text), 
             ha='left', va='top', fontsize=12, 
             bbox={'facecolor': 'lightgray', 'alpha': 0.2, 'pad': 10})
    
    plt.tight_layout()
    return fig

def create_interactive_plot(data, variety):
    """Создает интерактивный график Plotly"""
    fig = px.scatter(
        data,
        x='Рейтинг',
        y='Цена',
        color='Страна',
        hover_data=['Винодельня', 'Провинция', 'Регион 1'],
        title=f'Интерактивный анализ {variety}',
        size_max=15
    )
    fig.update_layout(
        hovermode='closest',
        height=600,
        width=800
    )
    return fig

def create_wordcloud(data, variety):
    """Создает облако слов из описаний"""
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
    ax.set_title(f'Частые слова в описаниях {variety}')
    return fig

def analyze_sentiment(data):
    """Анализирует тональность описаний"""
    data['sentiment'] = data['Описание'].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity if pd.notna(x) else 0
    )
    
    fig = px.box(
        data,
        x='Страна',
        y='sentiment',
        color='Страна',
        title='Распределение тональности описаний по странам'
    )
    return fig, data

def create_geographical_analysis(data):
    """Создает географический анализ"""
    if 'Провинция' not in data.columns or 'Регион 1' not in data.columns:
        return None
    
    # Группировка по регионам
    region_stats = data.groupby(['Страна', 'Провинция', 'Регион 1']).agg({
        'Рейтинг': 'mean',
        'Цена': 'mean',
        'Сорт': 'count'
    }).reset_index()
    
    # Создаем карту
    m = folium.Map(location=[20, 0], zoom_start=2)
    
    # Добавляем маркеры (упрощенный вариант)
    for idx, row in region_stats.iterrows():
        folium.Marker(
            location=[np.random.uniform(-60, 70), np.random.uniform(-180, 180)],  # Заглушка для координат
            popup=f"{row['Регион 1']}<br>Рейтинг: {row['Рейтинг']:.1f}<br>Цена: ${row['Цена']:.1f}",
            tooltip=row['Регион 1']
        ).add_to(m)
    
    return m, region_stats

def create_pdf_report(data, variety, stats):
    """Создает PDF отчет с поддержкой Unicode"""
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
    pdf.set_font('DejaVu', '', 12)
    
    # Заголовок
    pdf.cell(0, 10, f"Аналитический отчет по винам {variety}", 0, 1, 'C')
    pdf.ln(10)
    
    # Основная статистика
    pdf.cell(0, 10, "Основные статистики:", 0, 1)
    
    for country in stats.index:
        text = f"{country}: Средний рейтинг - {stats.loc[country, ('Рейтинг', 'mean')]:.1f}, Средняя цена - ${stats.loc[country, ('Цена', 'mean')]:.1f}"
        pdf.multi_cell(0, 10, text)
    
    # Топ вин
    pdf.ln(5)
    pdf.cell(0, 10, "Топ-5 самых дорогих вин:", 0, 1)
    top_expensive = data.nlargest(5, 'Цена')[['Винодельня', 'Страна', 'Рейтинг', 'Цена']]
    
    for _, row in top_expensive.iterrows():
        text = f"{row['Винодельня']} ({row['Страна']}): {row['Рейтинг']} баллов, ${row['Цена']}"
        pdf.multi_cell(0, 10, text)
    
    # Сохраняем во временный файл
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    pdf.output(temp_file.name)
    return temp_file.name

def analyze_wine(data, variety, countries):
    """Основная функция анализа данных"""
    if data is None:
        return
    
    if data.empty:
        st.warning("Загруженные данные пусты!")
        return

    required_columns = ['Страна', 'Рейтинг', 'Цена', 'Сорт']
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        st.error(f"Отсутствуют необходимые столбцы: {', '.join(missing_cols)}")
        return
    
    # Фильтрация данных
    filtered_data = data[data['Сорт'].str.contains(variety, case=False, na=False)]
    filtered_data = filtered_data[filtered_data['Страна'].isin(countries)]
    filtered_data = filtered_data.dropna(subset=['Рейтинг', 'Цена', 'Страна'])
    
    if len(filtered_data) == 0:
        st.warning(f"Не найдено данных для сорта {variety} в выбранных странах")
        return
    
    # Основной анализ
    with st.spinner('Выполняем анализ...'):
        # Вкладки для разных видов анализа
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Основная аналитика", 
            "📈 Интерактивные графики", 
            "📝 Анализ текста", 
            "🌍 Географический анализ",
            "📄 Полный отчет"
        ])
        
        with tab1:
            st.subheader(f"Основной анализ {variety}")
            st.markdown("""
            **На этой вкладке представлены основные статистики и визуализации:**
            - Графики распределения рейтингов и цен
            - Зависимость цены от рейтинга
            - Распределение по годам
            - Топ самых дорогих и высокооцененных вин
            - Подробная статистика по странам
            """)
            
            fig = create_summary_plot(data, filtered_data, variety)
            st.pyplot(fig)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Топ-5 самых дорогих")
                top_expensive = filtered_data.nlargest(5, 'Цена')[['Страна', 'Рейтинг', 'Цена', 'Винодельня']]
                st.dataframe(top_expensive)
            
            with col2:
                st.subheader("Топ-5 самых высокооцененных")
                top_rated = filtered_data.nlargest(5, 'Рейтинг')[['Страна', 'Рейтинг', 'Цена', 'Винодельня']]
                st.dataframe(top_rated)
            
            st.subheader("Статистика по странам")
            try:
                stats = filtered_data.groupby('Страна').agg({
                    'Рейтинг': ['mean', 'median', 'std'],
                    'Цена': ['mean', 'median', 'std'],
                    'Сорт': 'count'
                })
                
                stats_translated = translate_stats(stats)
                
                formatted_stats = stats_translated.style.format({
                    ('Рейтинг', 'Среднее'): '{:.1f}',
                    ('Рейтинг', 'Медиана'): '{:.1f}',
                    ('Рейтинг', 'Станд. отклонение'): '{:.2f}',
                    ('Цена', 'Среднее'): '${:.2f}',
                    ('Цена', 'Медиана'): '${:.2f}',
                    ('Цена', 'Станд. отклонение'): '${:.2f}',
                    ('Сорт', 'Количество'): '{:.0f}'
                })
                
                st.dataframe(formatted_stats)
                
            except Exception as e:
                st.error(f"Ошибка при формировании статистики: {str(e)}")
                logger.exception("Ошибка формирования статистики")
        
        with tab2:
            st.subheader("Интерактивная визуализация")
            st.markdown("""
            **Интерактивный график зависимости цены от рейтинга:**
            - Точки представляют отдельные вина
            - Цветом обозначены страны
            - При наведении отображается дополнительная информация
            - Используйте фильтры в боковой панели для уточнения данных
            """)
            
            fig = create_interactive_plot(filtered_data, variety)
            st.plotly_chart(fig, use_container_width=True, key=f"interactive_{variety}")
            
            # Фильтры для интерактивного графика
            st.sidebar.subheader("Фильтры для графика")
            min_price, max_price = st.sidebar.slider(
                "Диапазон цен ($)",
                min_value=int(filtered_data['Цена'].min()),
                max_value=int(filtered_data['Цена'].max()),
                value=(int(filtered_data['Цена'].min()), int(filtered_data['Цена'].max()))
            )
            
            min_rating, max_rating = st.sidebar.slider(
                "Диапазон рейтингов",
                min_value=int(filtered_data['Рейтинг'].min()),
                max_value=int(filtered_data['Рейтинг'].max()),
                value=(int(filtered_data['Рейтинг'].min()), int(filtered_data['Рейтинг'].max()))
            )
            
            filtered = filtered_data[
                (filtered_data['Цена'] >= min_price) & 
                (filtered_data['Цена'] <= max_price) &
                (filtered_data['Рейтинг'] >= min_rating) & 
                (filtered_data['Рейтинг'] <= max_rating)
            ]
            
            if len(filtered) > 0:
                fig_filtered = create_interactive_plot(filtered, variety)
                st.plotly_chart(fig_filtered, use_container_width=True, key=f"filtered_{variety}")
            else:
                st.warning("Нет данных для выбранных фильтров")
        
        with tab3:
            st.subheader("Анализ текстовых описаний")
            st.markdown("""
            **Анализ текстовых описаний вин:**
            - Облако слов показывает наиболее частые термины в описаниях
            - Анализ тональности оценивает эмоциональную окраску описаний
            - Топ самых положительных и отрицательных описаний
            """)
            
            if 'Описание' in filtered_data.columns:
                # Облако слов
                st.pyplot(create_wordcloud(filtered_data, variety))
                
                # Анализ тональности
                sentiment_fig, sentiment_data = analyze_sentiment(filtered_data)
                st.plotly_chart(sentiment_fig, use_container_width=True, key="sentiment_analysis")
                
                # Топ положительных и отрицательных описаний
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Самые положительные описания")
                    positive = sentiment_data.nlargest(3, 'sentiment')['Описание']
                    for i, desc in enumerate(positive, 1):
                        st.write(f"{i}. {desc[:200]}...")
                
                with col2:
                    st.subheader("Самые отрицательные описания")
                    negative = sentiment_data.nsmallest(3, 'sentiment')['Описание']
                    for i, desc in enumerate(negative, 1):
                        st.write(f"{i}. {desc[:200]}...")
            else:
                st.warning("В данных отсутствует столбец с описаниями")
        
        with tab4:
            st.subheader("Географический анализ")
            st.markdown("""
            **Географическое распределение вин:**
            - Карта с примерным расположением регионов
            - Статистика по регионам (средний рейтинг, цена)
            - При нажатии на маркер отображается дополнительная информация
            """)
            
            geo_result = create_geographical_analysis(filtered_data)
            if geo_result:
                m, region_stats = geo_result
                st_folium(m, width=800, height=500, returned_objects=[])
                
                st.subheader("Статистика по регионам")
                st.dataframe(region_stats.sort_values('Рейтинг', ascending=False))
            else:
                st.warning("Недостаточно данных для географического анализа")
        
        with tab5:
            st.subheader("Полный отчет")
            st.markdown("""
            **Итоговый отчет в формате PDF:**
            - Содержит основные статистики
            - Список самых дорогих вин
            - Доступен для скачивания
            """)
            
            if 'stats' in locals():
                pdf_path = create_pdf_report(filtered_data, variety, stats)
                
                with open(pdf_path, "rb") as f:
                    base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
                
                st.download_button(
                    label="Скачать PDF отчет",
                    data=open(pdf_path, "rb").read(),
                    file_name=f"wine_analysis_{variety}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )
                
                os.unlink(pdf_path)
            else:
                st.warning("Сначала выполните основной анализ")

def clear_cache_and_reload():
    """Очищает кэш и перезагружает приложение"""
    st.session_state.clear()
    st.rerun()

def show_guide():
    """Показывает руководство по использованию приложения"""
    with st.expander("📖 Руководство по использованию", expanded=False):
        st.markdown("""
        ## Как пользоваться приложением для анализа вин
        
        1. **Загрузка данных**:
           - В боковой панели загрузите файл с данными о винах (CSV или Excel)
           - Приложение автоматически обработает данные и переведет названия столбцов
        
        2. **Настройка анализа**:
           - Выберите страны для сравнения
           - Выберите сорт вина
           - Установите диапазоны цен и рейтингов
        
        3. **Вкладки с анализом**:
           - **Основная аналитика**: Графики распределения, топ вин, статистика
           - **Интерактивные графики**: Интерактивная зависимость цены от рейтинга
           - **Анализ текста**: Облако слов и анализ тональности описаний
           - **Географический анализ**: Карта с распределением вин по регионам
           - **Полный отчет**: PDF с итогами анализа
        
        4. **Дополнительные функции**:
           - Фильтры для уточнения данных
           - Возможность скачать отчет в PDF
           - Кнопка сброса для полного обновления приложения
        
        ## Требования к данным
        Файл должен содержать как минимум следующие столбцы:
        - Страна
        - Рейтинг (числовой)
        - Цена (числовой)
        - Сорт (название вина)
        
        Дополнительные полезные столбцы:
        - Описание (для анализа текста)
        - Год (для анализа по годам)
        - Провинция/Регион (для географического анализа)
        """)

def main():
    """Основной интерфейс приложения"""
    st.title("🍷 Расширенный анализ вин")
    show_guide()
    
    # Кнопка для принудительного обновления
    if st.sidebar.button("🔄 Полный сброс и обновление"):
        clear_cache_and_reload()
    
    uploaded_file = st.sidebar.file_uploader(
        "📂 Загрузите файл с данными о винах", 
        type=["csv", "xlsx"],
        help="Поддерживаются CSV и Excel файлы"
    )
    
    st.sidebar.header("⚙️ Настройки анализа")
    
    if uploaded_file is not None:
        try:
            use_cache = st.sidebar.checkbox("Использовать кэширование", value=True)
            
            uploaded_file.seek(0)
            stringio = StringIO(uploaded_file.getvalue().decode('utf-8'))
            df_preview = pd.read_csv(stringio, encoding='utf-8', on_bad_lines='skip')
            df_preview.columns = df_preview.columns.str.lower().str.replace(' ', '_')
            df_preview = translate_data(df_preview)
            available_countries = sorted(df_preview['Страна'].dropna().unique())
            
            selected_countries = st.sidebar.multiselect(
                "Выберите страны для анализа",
                options=available_countries,
                default=available_countries[:2] if len(available_countries) >= 2 else available_countries
            )
            
            available_varieties = sorted(df_preview['Сорт'].dropna().unique())
            variety = st.sidebar.selectbox(
                "Сорт вина для анализа",
                options=available_varieties,
                index=available_varieties.index('Chardonnay') if 'Chardonnay' in available_varieties else 0
            )
            
            price_range = st.sidebar.slider(
                "Фильтр по цене ($)",
                min_value=0,
                max_value=int(df_preview['Цена'].max()) if 'Цена' in df_preview.columns else 1000,
                value=(0, int(df_preview['Цена'].max()) if 'Цена' in df_preview.columns else 1000)
            )
            
            rating_range = st.sidebar.slider(
                "Фильтр по рейтингу",
                min_value=0,
                max_value=100,
                value=(80, 100)
            )
            
            st.session_state['countries'] = selected_countries
            
            if st.sidebar.button("🚀 Запустить анализ"):
                with st.spinner('Загружаем и обрабатываем данные...'):
                    data = load_data(uploaded_file, use_cache=use_cache)
                    
                    if data is not None:
                        # Применяем фильтры
                        filtered = data[
                            (data['Страна'].isin(selected_countries)) &
                            (data['Цена'] >= price_range[0]) &
                            (data['Цена'] <= price_range[1]) &
                            (data['Рейтинг'] >= rating_range[0]) &
                            (data['Рейтинг'] <= rating_range[1])
                        ]
                        
                        analyze_wine(filtered, variety, selected_countries)
                        
                        st.snow()
                        st.success("Анализ завершен успешно!")
        
        except Exception as e:
            st.error(f"Ошибка при обработке файла: {e}")
            logger.exception("Ошибка при обработке файла")
    else:
        st.info("ℹ️ Пожалуйста, загрузите файл с данными о винах для начала анализа")
        st.image("https://i.imgur.com/Qq6DZQa.png", caption="Пример данных о винах", width=600)

if __name__ == "__main__":
    main()