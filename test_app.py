"""
스마트한 쇼핑 앱 - LangGraph 버전 (수정판)
"""

import streamlit as st

# 페이지 설정 (반드시 첫 번째로 실행)
st.set_page_config(
    page_title="스마트한 쇼핑 (LangGraph)",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 필요한 라이브러리 import
import pandas as pd
from supabase import create_client
from openai import OpenAI
import os
from dotenv import load_dotenv
from datetime import datetime
import time
import json
import re
import requests
from bs4 import BeautifulSoup
import numpy as np
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import io
import base64
import urllib.request

# LangGraph 관련
from typing import TypedDict, Annotated, List, Union, Dict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
import operator

# 앱 시작 시 폰트 자동 다운로드
@st.cache_resource
def ensure_font():
    """폰트 파일 확인 및 다운로드"""
    font_path = "./NanumGothic.ttf"
    
    if not os.path.exists(font_path):
        with st.spinner("한글 폰트 다운로드 중..."):
            try:
                # 나눔고딕 폰트 다운로드 (공식 GitHub 저장소)
                url = "https://github.com/naver/nanumfont/raw/master/fonts/NanumFontSetup_TTF_GOTHIC/NanumGothic.ttf"
                urllib.request.urlretrieve(url, font_path)
                st.success("✅ 한글 폰트 다운로드 완료!")
            except Exception as e:
                st.error(f"❌ 폰트 다운로드 실패: {e}")
                
                # 대체 URL 시도
                try:
                    alt_url = "https://cdn.jsdelivr.net/gh/naver/nanumfont@master/fonts/NanumFontSetup_TTF_GOTHIC/NanumGothic.ttf"
                    urllib.request.urlretrieve(alt_url, font_path)
                    st.success("✅ 한글 폰트 다운로드 완료! (대체 경로)")
                except:
                    st.warning("⚠️ 폰트 다운로드 실패. 키워드가 영문으로 표시될 수 있습니다.")
                    return None
    
    return font_path

# 폰트 확인 (페이지 설정 후 실행)
font_path = ensure_font()

# 환경 변수 로드
load_dotenv()

# API 키 설정
SUPABASE_URL = os.getenv("SUPABASE_URL") or st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID") or st.secrets.get("NAVER_CLIENT_ID", "")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET") or st.secrets.get("NAVER_CLIENT_SECRET", "")

# LangSmith 설정 (선택적)
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY") or st.secrets.get("LANGSMITH_API_KEY", "")
if LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "smart-shopping-app"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

# 세션 상태 초기화
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'bookmarks' not in st.session_state:
    st.session_state.bookmarks = []
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'total_searches' not in st.session_state:
    st.session_state.total_searches = 0
if 'saved_products' not in st.session_state:
    st.session_state.saved_products = 0

# CSS 스타일 - 다크모드 지원
if st.session_state.dark_mode:
    bg_gradient = "linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)"
    card_bg = "#0f3460"
    text_color = "#ffffff"
    secondary_text = "#e94560"
    header_gradient = "linear-gradient(135deg, #e94560 0%, #0f3460 100%)"
else:
    bg_gradient = "linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)"
    card_bg = "white"
    text_color = "#333333"
    secondary_text = "#667eea"
    header_gradient = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"

st.markdown(f"""
<style>
    /* 전체 배경 및 기본 스타일 */
    .stApp {{
        background: {bg_gradient};
    }}
    
    /* 메인 헤더 개선 */
    .main-header {{
        text-align: center;
        padding: 3rem 0;
        background: {header_gradient};
        color: white;
        border-radius: 20px;
        margin-bottom: 3rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }}
    
    .main-header h1 {{
        text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.3), 
                     0 0 20px rgba(255, 255, 255, 0.2);
    }}
    
    .main-header p {{
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }}
    
    .main-header::before {{
        content: "";
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: shimmer 3s infinite;
    }}
    
    @keyframes shimmer {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    
    /* 카드 스타일 */
    .search-card {{
        background: {card_bg};
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.08);
        margin-bottom: 2rem;
        color: {text_color};
    }}
    
    /* 장점 섹션 개선 */
    .pros-section {{
        background: linear-gradient(135deg, #d4f1d4 0%, #b8e6b8 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: none;
        box-shadow: 0 5px 15px rgba(40, 167, 69, 0.1);
        transition: transform 0.3s ease;
    }}
    
    .pros-section:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(40, 167, 69, 0.15);
    }}
    
    /* 단점 섹션 개선 */
    .cons-section {{
        background: linear-gradient(135deg, #ffd6d6 0%, #ffb8b8 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: none;
        box-shadow: 0 5px 15px rgba(220, 53, 69, 0.1);
        transition: transform 0.3s ease;
    }}
    
    .cons-section:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(220, 53, 69, 0.15);
    }}
    
    /* 프로세스 정보 개선 */
    .process-info {{
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        border: none;
        box-shadow: 0 3px 10px rgba(33, 150, 243, 0.1);
    }}
    
    /* 버튼 스타일 개선 */
    .stButton > button {{
        background: {header_gradient};
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 30px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }}
    
    /* 입력 필드 스타일 */
    .stTextInput > div > div > input {{
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem 1rem;
        transition: all 0.3s ease;
        background: {card_bg};
        color: {text_color};
    }}
    
    .stTextInput > div > div > input:focus {{
        border-color: {secondary_text};
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }}
    
    /* 메트릭 카드 */
    .metric-card {{
        background: {card_bg};
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.08);
        text-align: center;
        transition: all 0.3s ease;
        color: {text_color};
    }}
    
    .metric-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.12);
    }}
    
    /* 애니메이션 효과 */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    .fade-in {{
        animation: fadeIn 0.6s ease-out;
    }}
    
    /* 로딩 스피너 */
    .spinner {{
        width: 50px;
        height: 50px;
        margin: 0 auto;
        border: 5px solid #f3f3f3;
        border-top: 5px solid {secondary_text};
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }}
    
    @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    
    /* 프로스/콘스 아이템 */
    .pros-item, .cons-item {{
        background: white;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        transition: all 0.3s ease;
        animation: fadeIn 0.5s ease-out;
    }}
    
    .pros-item {{
        border-left: 4px solid #28a745;
    }}
    
    .cons-item {{
        border-left: 4px solid #dc3545;
    }}
    
    .pros-item:hover, .cons-item:hover {{
        transform: translateX(5px);
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
    }}
    
    /* 모바일 반응형 */
    @media (max-width: 768px) {{
        .main-header {{
            padding: 2rem 1rem;
            font-size: 0.9rem;
        }}
        .main-header h1 {{
            font-size: 1.8rem;
        }}
        .search-card {{
            padding: 1.5rem 1rem;
        }}
        .pros-section, .cons-section {{
            padding: 1.5rem 1rem;
        }}
    }}
    
    /* 프로그레스 바 */
    .progress-bar {{
        width: 100%;
        height: 8px;
        background-color: #e0e0e0;
        border-radius: 4px;
        overflow: hidden;
        margin: 1rem 0;
    }}
    
    .progress-fill {{
        height: 100%;
        background: {header_gradient};
        animation: progress 2s ease-out;
    }}
    
    @keyframes progress {{
        from {{ width: 0%; }}
        to {{ width: 100%; }}
    }}
</style>
""", unsafe_allow_html=True)

# 사이드바 설정
with st.sidebar:
    st.markdown("### ⚙️ 설정")
    dark_mode = st.checkbox("🌙 다크모드", value=st.session_state.dark_mode)
    st.session_state.dark_mode = dark_mode
    
    st.markdown("### 📌 북마크")
    if st.session_state.bookmarks:
        for bookmark in st.session_state.bookmarks:
            if st.button(f"🔖 {bookmark}", key=f"bookmark_{bookmark}"):
                st.session_state.selected_bookmark = bookmark
    else:
        st.info("북마크가 없습니다")
    
    st.markdown("### 📊 사용 통계")
    st.metric("총 검색 수", f"{st.session_state.total_searches}회")
    st.metric("저장된 제품", f"{st.session_state.saved_products}개")

# 헤더 - 스마트한 쇼핑 제목만
st.markdown("""
<div class="main-header">
    <h1 style="margin-bottom: 0.5rem;">🛒 스마트한 쇼핑 (LangGraph Edition)</h1>
    <p style="font-size: 1.2rem; margin-top: 0.5rem;">
        LangGraph로 구현한 지능형 제품 리뷰 분석 시스템
    </p>
    <p style="font-size: 0.9rem; margin-top: 0.3rem; opacity: 0.8;">
        <i class="fas fa-robot"></i> AI가 수천 개의 리뷰를 분석하여 핵심 장단점을 요약해드립니다
    </p>
</div>
""", unsafe_allow_html=True)

# ========================
# LangGraph State 정의
# ========================

class SearchState(TypedDict):
    """검색 프로세스의 상태"""
    product_name: str
    search_method: str  # "database" or "web_crawling"
    results: dict
    pros: List[str]
    cons: List[str]
    sources: List[dict]
    messages: Annotated[List[Union[HumanMessage, AIMessage]], operator.add]
    error: str

# ========================
# 크롤링 클래스
# ========================

class ProConsLaptopCrawler:
    def __init__(self, naver_client_id, naver_client_secret):
        self.naver_headers = {
            "X-Naver-Client-Id": naver_client_id,
            "X-Naver-Client-Secret": naver_client_secret
        }
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        # 통계
        self.stats = {
            'total_crawled': 0,
            'valid_pros_cons': 0,
            'api_errors': 0
        }
    
    def remove_html_tags(self, text):
        """HTML 태그 제거"""
        text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(r'<[^>]+>', '', text)
        return text.strip()
    
    def search_blog(self, query, display=20):
        """네이버 블로그 검색"""
        url = "https://openapi.naver.com/v1/search/blog"
        params = {
            "query": query,
            "display": display,
            "sort": "sim"
        }
        
        try:
            response = requests.get(url, headers=self.naver_headers, params=params)
            if response.status_code == 200:
                result = response.json()
                for item in result.get('items', []):
                    item['title'] = self.remove_html_tags(item['title'])
                    item['description'] = self.remove_html_tags(item['description'])
                return result
        except Exception as e:
            print(f"검색 오류: {e}")
        return None
    
    def crawl_content(self, url):
        """블로그 본문 크롤링"""
        try:
            if "blog.naver.com" in url:
                parts = url.split('/')
                if len(parts) >= 5:
                    blog_id = parts[3]
                    post_no = parts[4].split('?')[0]
                    mobile_url = f"https://m.blog.naver.com/{blog_id}/{post_no}"
                    
                    response = requests.get(mobile_url, headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    })
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        content = ""
                        for selector in ['div.se-main-container', 'div#postViewArea', 'div.post_ct']:
                            elem = soup.select_one(selector)
                            if elem:
                                content = elem.get_text(separator='\n', strip=True)
                                break
                        
                        if not content:
                            content = soup.get_text(separator='\n', strip=True)
                        
                        content = re.sub(r'\s+', ' ', content)
                        content = content.replace('\u200b', '')
                        
                        return content if len(content) > 300 else None
        except Exception as e:
            print(f"크롤링 오류: {e}")
        return None
    
    def extract_pros_cons_with_gpt(self, product_name, content):
        """ChatGPT로 장단점 추출"""
        if not content or len(content) < 200:
            return None
        
        content_preview = content[:1500]
        
        prompt = f"""다음은 "{product_name}"에 대한 블로그 리뷰입니다.

[블로그 내용]
{content_preview}

위 내용에서 {product_name}의 장점과 단점을 추출해주세요.
실제 사용 경험에 기반한 구체적인 내용만 포함하세요.

다음 형식으로 응답해주세요:

장점:
- (구체적인 장점 1)
- (구체적인 장점 2)
- (구체적인 장점 3)

단점:
- (구체적인 단점 1)
- (구체적인 단점 2)
- (구체적인 단점 3)

만약 장단점 정보가 충분하지 않으면 "정보 부족"이라고 답해주세요."""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": "당신은 제품 리뷰 분석 전문가입니다. 실제 사용 경험에 기반한 장단점만 추출합니다."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            result = response.choices[0].message.content.strip()
            
            if result and "정보 부족" not in result:
                pros = []
                cons = []
                
                lines = result.split('\n')
                current_section = None
                
                for line in lines:
                    line = line.strip()
                    if '장점:' in line or '장점 :' in line:
                        current_section = 'pros'
                    elif '단점:' in line or '단점 :' in line:
                        current_section = 'cons'
                    elif line.startswith('-') and current_section:
                        point = line[1:].strip()
                        if point and len(point) > 5:
                            if current_section == 'pros':
                                pros.append(point)
                            else:
                                cons.append(point)
                
                if pros or cons:
                    self.stats['valid_pros_cons'] += 1
                    return {
                        'pros': pros[:5],
                        'cons': cons[:5]
                    }
            
            return None
                
        except Exception as e:
            self.stats['api_errors'] += 1
            print(f"GPT API 오류: {str(e)[:100]}")
            return None
    
    def deduplicate_points(self, points):
        """유사한 장단점 중복 제거"""
        if not points:
            return []
        
        unique_points = []
        seen_keywords = set()
        
        for point in points:
            keywords = set(word for word in point.split() if len(word) > 2)
            
            if len(keywords & seen_keywords) < len(keywords) * 0.5:
                unique_points.append(point)
                seen_keywords.update(keywords)
            
            if len(unique_points) >= 10:
                break
        
        return unique_points

# ========================
# 유틸리티 함수들
# ========================

def show_loading_animation():
    """로딩 애니메이션 표시"""
    loading_placeholder = st.empty()
    loading_placeholder.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <div class="spinner"></div>
        <p style="margin-top: 1rem; color: #667eea; font-weight: 600;">
            <i class="fas fa-brain"></i> AI가 제품 정보를 분석하고 있습니다...
        </p>
        <div class="progress-bar">
            <div class="progress-fill"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    return loading_placeholder

def create_pros_cons_chart(pros_count, cons_count):
    """장단점 차트 생성"""
    fig = go.Figure(data=[
        go.Bar(
            name='장점',
            x=['분석 결과'],
            y=[pros_count],
            marker_color='#28a745',
            text=f'{pros_count}개',
            textposition='auto',
            hovertemplate='장점: %{y}개<extra></extra>'
        ),
        go.Bar(
            name='단점',
            x=['분석 결과'],
            y=[cons_count],
            marker_color='#dc3545',
            text=f'{cons_count}개',
            textposition='auto',
            hovertemplate='단점: %{y}개<extra></extra>'
        )
    ])
    
    fig.update_layout(
        barmode='group',
        height=300,
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14),
        showlegend=True,
        legend=dict(x=0.3, y=1.1, orientation='h'),
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
        bargap=0.3
    )
    
    return fig

def extract_keywords(texts):
    """텍스트에서 핵심 키워드 추출"""
    # 확장된 불용어 정의
    stopwords = {
        # 일반 불용어
        '수', '있습니다', '있어요', '있음', '좋습니다', '좋아요', '좋음', 
        '나쁩니다', '나빠요', '나쁨', '않습니다', '않아요', '않음',
        '입니다', '이다', '되다', '하다', '있다', '없다', '같다',
        '위해', '통해', '대해', '매우', '정말', '너무', '조금',
        '그리고', '하지만', '그러나', '또한', '때문', '경우',
        '제공합니다', '제공', '합니다', '해요', '드립니다', '드려요',
        '위한', '위하여', '따라', '따른', '통한', '대한', '관한',
        '됩니다', '됨', '되어', '되었습니다', '했습니다', '하는',
        '이', '그', '저', '것', '것이', '것을', '것은', '것도',
        '더', '덜', '꽤', '약간', '살짝', '많이', '적게', '조금',
        '모든', '각', '각각', '여러', '몇', '몇몇', '전체', '일부',
        '항상', '가끔', '종종', '자주', '언제나', '절대', '전혀',
        '만', '도', '까지', '부터', '에서', '에게', '으로', '로',
        '와', '과', '하고', '이고', '이며', '거나', '든지', '라고',
        '들', '등', '등등', '따위', '및', '또는', '혹은', '즉',
        '의', '를', '을', '에', '가', '이', '은', '는', '와', '과',
        '했다', '한다', '하며', '하여', '해서', '하고', '하니', '하면',
        '그래서', '그러니', '그러므로', '따라서', '때문에', '왜냐하면',
        '비해', '보다', '처럼', '같이', '만큼', '대로', '듯이',
        '점', '면', '측면', '부분', '경우', '상황', '상태', '정도',
        '이런', '저런', '그런', '어떤', '무슨', '어느', '어떻게',
        '가능', '불가능', '필요', '불필요', '중요', '사용', '이용',
        '느낌', '기분', '마음', '생각', '의견', '감정', '인상',
        '한', '두', '세', '네', '몇', '여러', '많은', '적은',
        '첫', '둘', '셋', '넷', '첫째', '둘째', '셋째', '마지막',
        '좀', '꼭', '딱', '막', '참', '진짜', '정말로', '확실히',
        '거의', '대부분', '대체로', '보통', '일반적', '평균적',
        '특히', '특별히', '주로', '대개', '대체로', '전반적'
    }
    
    # 모든 텍스트를 결합하고 키워드 추출
    all_text = ' '.join(texts)
    
    # 한글만 추출 (영어, 숫자 제외)
    words = re.findall(r'[가-힣]+', all_text)
    
    # 필터링 조건 강화
    # 1. 2글자 이상
    # 2. 불용어가 아님
    # 3. 너무 일반적인 단어 제외
    filtered_words = []
    for word in words:
        if (len(word) >= 2 and 
            word not in stopwords and
            not word.endswith('습니다') and
            not word.endswith('합니다') and
            not word.endswith('입니다') and
            not word.endswith('됩니다') and
            not word.startswith('있') and
            not word.startswith('없') and
            not word.startswith('하') and
            not word.startswith('되') and
            not word.startswith('않')):
            filtered_words.append(word)
    
    # 단어 빈도 계산
    word_freq = Counter(filtered_words)
    
    # 빈도수가 1인 단어는 제외 (더 중요한 키워드만 남김)
    word_freq = {word: freq for word, freq in word_freq.items() if freq > 1}
    
    return word_freq

def create_wordcloud(texts, title, color_scheme):
    """워드클라우드 생성"""
    if not texts:
        return None
    
    # 키워드 추출
    word_freq = extract_keywords(texts)
    
    if not word_freq:
        return None
    
    # 빈도수 기준으로 상위 키워드만 선택 (최대 40개)
    top_keywords = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:40])
    
    # matplotlib 한글 폰트 설정
    import matplotlib.pyplot as plt
    
    # 프로젝트 루트의 폰트 파일 우선 사용
    font_path = "./NanumGothic.ttf"
    
    # 폰트 파일이 없는 경우 다른 경로 시도
    if not os.path.exists(font_path):
        font_paths = [
            "NanumGothic.ttf",  # 같은 디렉토리
            "./fonts/NanumGothic.ttf",  # fonts 폴더
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Linux
            "C:/Windows/Fonts/malgun.ttf",  # Windows
            "/System/Library/Fonts/AppleSDGothicNeo.ttc"  # macOS
        ]
        
        for path in font_paths:
            if os.path.exists(path):
                font_path = path
                break
    
    # 워드클라우드 생성
    plt.figure(figsize=(10, 6), facecolor='white')
    
    if font_path and os.path.exists(font_path) and top_keywords:
        try:
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap=color_scheme,
                font_path=font_path,
                relative_scaling=0.7,  # 크기 차이를 더 크게
                min_font_size=14,     # 최소 폰트 크기 증가
                max_words=30,         # 표시할 단어 수 제한
                prefer_horizontal=0.8, # 가로 방향 선호도 증가
                margin=15,            # 여백 증가
                collocations=False    # 연어 처리 비활성화
            ).generate_from_frequencies(top_keywords)
            
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout(pad=0)
            
            # 이미지를 bytes로 변환
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
            buf.seek(0)
            plt.close()
            
            return buf
            
        except Exception as e:
            plt.close()
            st.error(f"워드클라우드 생성 오류: {str(e)}")
            return None
    else:
        st.warning(f"한글 폰트를 찾을 수 없습니다. NanumGothic.ttf 파일을 프로젝트 루트에 추가해주세요.")
        return None

def create_text_cloud(texts, title, color):
    """워드클라우드 대신 텍스트 기반 시각화"""
    if not texts:
        return
    
    # 키워드 추출
    word_freq = extract_keywords(texts)
    
    if not word_freq:
        return
    
    # 상위 20개 키워드
    top_words = word_freq.most_common(20)
    
    # 최대 빈도수
    max_freq = top_words[0][1] if top_words else 1
    
    # HTML로 워드클라우드 스타일 표현
    html_words = []
    for word, freq in top_words:
        # 빈도수에 따른 크기 계산 (1rem ~ 3rem)
        size = 1 + (freq / max_freq) * 2
        # 빈도수에 따른 투명도 (0.5 ~ 1.0)
        opacity = 0.5 + (freq / max_freq) * 0.5
        
        html_words.append(
            f'<span style="font-size: {size}rem; color: {color}; opacity: {opacity}; '
            f'margin: 0.3rem; display: inline-block; font-weight: bold;">{word}</span>'
        )
    
    # 랜덤하게 섞기
    import random
    random.shuffle(html_words)
    
    # HTML 출력
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem; background: white; 
                border-radius: 15px; border: 2px solid {color}20;">
        <h4 style="color: {color}; margin-bottom: 1rem;">{title}</h4>
        <div style="line-height: 2.5;">
            {''.join(html_words)}
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_wordclouds(pros, cons):
    """장단점 워드클라우드 표시"""
    col1, col2 = st.columns(2)
    
    with col1:
        if pros:
            st.markdown("""
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #d4f1d4 0%, #b8e6b8 100%); border-radius: 15px;">
                <h3 style="color: #28a745; margin: 0;">
                    <i class="fas fa-check-circle"></i> 장점 키워드
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            # 장점 워드클라우드 생성 시도
            pros_wordcloud = create_wordcloud(pros, "", "Greens")
            if pros_wordcloud:
                st.image(pros_wordcloud, use_container_width=True)
            else:
                # 워드클라우드 실패 시 텍스트 기반 시각화
                create_text_cloud(pros, "장점 키워드 분석", "#28a745")
            
            # 주요 키워드 표시
            keywords = extract_keywords(pros)
            if keywords and isinstance(keywords, dict):
                # Counter가 아닌 dict인 경우 처리
                sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:5]
                if sorted_keywords:
                    st.markdown("**🔑 주요 키워드:**")
                    keyword_html = " ".join([f'<span style="background: #d4f1d4; padding: 0.2rem 0.5rem; border-radius: 15px; margin: 0.2rem; display: inline-block;">{word} ({count})</span>' 
                                            for word, count in sorted_keywords])
                    st.markdown(keyword_html, unsafe_allow_html=True)
    
    with col2:
        if cons:
            st.markdown("""
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #ffd6d6 0%, #ffb8b8 100%); border-radius: 15px;">
                <h3 style="color: #dc3545; margin: 0;">
                    <i class="fas fa-times-circle"></i> 단점 키워드
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            # 단점 워드클라우드 생성 시도
            cons_wordcloud = create_wordcloud(cons, "", "Reds")
            if cons_wordcloud:
                st.image(cons_wordcloud, use_container_width=True)
            else:
                # 워드클라우드 실패 시 텍스트 기반 시각화
                create_text_cloud(cons, "단점 키워드 분석", "#dc3545")
            
            # 주요 키워드 표시
            keywords = extract_keywords(cons)
            if keywords and isinstance(keywords, dict):
                # Counter가 아닌 dict인 경우 처리
                sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:5]
                if sorted_keywords:
                    st.markdown("**🔑 주요 키워드:**")
                    keyword_html = " ".join([f'<span style="background: #ffd6d6; padding: 0.2rem 0.5rem; border-radius: 15px; margin: 0.2rem; display: inline-block;">{word} ({count})</span>' 
                                            for word, count in sorted_keywords])
                    st.markdown(keyword_html, unsafe_allow_html=True)

def create_comparison_chart(pros, cons):
    """장단점 비교 시각화"""
    # 카테고리별 분류
    categories = {
        '성능': ['성능', '속도', '빠르', '느리', '렉', '버벅', '프로세서', 'CPU', 'GPU', '메모리'],
        '디자인': ['디자인', '외관', '예쁘', '이쁘', '못생', '색상', '모양', '두께', '얇'],
        '가격': ['가격', '비싸', '저렴', '가성비', '비용', '돈', '할인', '세일'],
        '품질': ['품질', '마감', '재질', '튼튼', '약하', '고장', '내구성', '견고'],
        '기능': ['기능', '편의', '편리', '불편', '사용', '조작', '인터페이스'],
        '배터리': ['배터리', '충전', '전원', '지속', '방전'],
        '화면': ['화면', '디스플레이', '선명', '밝기', '해상도'],
        '기타': []
    }
    
    # 각 카테고리별 장단점 수 계산
    category_pros = {cat: 0 for cat in categories}
    category_cons = {cat: 0 for cat in categories}
    
    # 장점 분류
    for pro in pros:
        categorized = False
        for cat, keywords in categories.items():
            if cat != '기타' and any(keyword in pro for keyword in keywords):
                category_pros[cat] += 1
                categorized = True
                break
        if not categorized:
            category_pros['기타'] += 1
    
    # 단점 분류
    for con in cons:
        categorized = False
        for cat, keywords in categories.items():
            if cat != '기타' and any(keyword in con for keyword in keywords):
                category_cons[cat] += 1
                categorized = True
                break
        if not categorized:
            category_cons['기타'] += 1
    
    # 데이터가 있는 카테고리만 필터링
    active_categories = [cat for cat in categories if category_pros[cat] > 0 or category_cons[cat] > 0]
    
    if not active_categories:
        return None
    
    # 레이더 차트 생성
    fig = go.Figure()
    
    # 장점 데이터
    fig.add_trace(go.Scatterpolar(
        r=[category_pros[cat] for cat in active_categories],
        theta=active_categories,
        fill='toself',
        fillcolor='rgba(40, 167, 69, 0.3)',
        line=dict(color='#28a745', width=2),
        name='장점',
        hovertemplate='%{theta}<br>장점: %{r}개<extra></extra>'
    ))
    
    # 단점 데이터
    fig.add_trace(go.Scatterpolar(
        r=[category_cons[cat] for cat in active_categories],
        theta=active_categories,
        fill='toself',
        fillcolor='rgba(220, 53, 69, 0.3)',
        line=dict(color='#dc3545', width=2),
        name='단점',
        hovertemplate='%{theta}<br>단점: %{r}개<extra></extra>'
    ))
    
    # 최대값 계산
    max_value = max(
        max(category_pros.values()) if category_pros else 1,
        max(category_cons.values()) if category_cons else 1
    )
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max_value + 1]
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=True,
        title={
            'text': '🎯 카테고리별 장단점 분포',
            'font': {'size': 20, 'color': text_color},
            'x': 0.5,
            'xanchor': 'center'
        },
        height=400,
        margin=dict(l=80, r=80, t=80, b=80),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(x=0.85, y=0.95)
    )
    
    return fig

def create_summary_metrics(pros, cons):
    """요약 메트릭 시각화"""
    # 텍스트 통계
    total_reviews = len(pros) + len(cons)
    avg_length_pros = np.mean([len(p) for p in pros]) if pros else 0
    avg_length_cons = np.mean([len(c) for c in cons]) if cons else 0
    
    # 키워드 다양성
    pros_keywords = extract_keywords(pros)
    cons_keywords = extract_keywords(cons)
    
    diversity_score = len(set(pros_keywords.keys()) | set(cons_keywords.keys()))
    
    return {
        'total_reviews': total_reviews,
        'avg_length_pros': avg_length_pros,
        'avg_length_cons': avg_length_cons,
        'diversity_score': diversity_score,
        'positive_ratio': len(pros) / total_reviews * 100 if total_reviews > 0 else 0
    }

# ========================
# LangGraph 노드 함수들
# ========================

# 클라이언트 초기화
@st.cache_resource
def get_supabase_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

@st.cache_resource
def get_crawler():
    return ProConsLaptopCrawler(NAVER_CLIENT_ID, NAVER_CLIENT_SECRET) if NAVER_CLIENT_ID and NAVER_CLIENT_SECRET else None

def search_database(state: SearchState) -> SearchState:
    """데이터베이스에서 제품 검색"""
    product_name = state["product_name"]
    supabase = get_supabase_client()
    
    if not supabase:
        state["messages"].append(
            AIMessage(content="⚠️ 데이터베이스가 설정되지 않았습니다. 웹 검색으로 진행합니다.")
        )
        state["results"] = {"data": None}
        return state
    
    state["messages"].append(
        HumanMessage(content=f"📊 데이터베이스에서 '{product_name}' 검색 중...")
    )
    
    try:
        # 정확한 매칭만 시도
        exact_match = supabase.table('laptop_pros_cons').select("*").eq('product_name', product_name).execute()
        if exact_match.data:
            state["search_method"] = "database"
            state["results"] = {"data": exact_match.data}
            state["messages"].append(
                AIMessage(content=f"✅ 데이터베이스에서 '{product_name}' 정보를 찾았습니다! ({len(exact_match.data)}개 항목)")
            )
            return state
        
        state["messages"].append(
            AIMessage(content=f"❌ 데이터베이스에서 '{product_name}'을(를) 찾을 수 없습니다. 웹에서 검색합니다...")
        )
        state["results"] = {"data": None}
        return state
        
    except Exception as e:
        state["error"] = str(e)
        state["messages"].append(
            AIMessage(content=f"⚠️ 데이터베이스 검색 오류: {str(e)}")
        )
        state["results"] = {"data": None}
        return state

def crawl_web(state: SearchState) -> SearchState:
    """웹에서 제품 정보 크롤링"""
    if state["results"].get("data"):  # 이미 DB에서 찾은 경우
        return state
    
    product_name = state["product_name"]
    state["search_method"] = "web_crawling"
    crawler = get_crawler()
    
    if not crawler:
        state["messages"].append(
            AIMessage(content="⚠️ 웹 크롤링이 설정되지 않았습니다.")
        )
        return state
    
    state["messages"].append(
        HumanMessage(content=f"🌐 웹에서 '{product_name}' 리뷰 수집 시작...")
    )
    
    # 샘플 데이터 (API 키가 없을 때)
    if not OPENAI_API_KEY:
        state["pros"] = [
            "가볍고 휴대성이 좋습니다",
            "배터리 지속 시간이 깁니다",
            "디스플레이가 선명합니다",
            "성능이 우수합니다",
            "디자인이 세련되었습니다"
        ]
        state["cons"] = [
            "가격이 비쌉니다",
            "포트가 부족합니다",
            "키보드 키감이 아쉽습니다"
        ]
        state["messages"].append(
            AIMessage(content="📌 샘플 데이터를 표시합니다 (API 키 설정 필요)")
        )
        return state
    
    all_pros = []
    all_cons = []
    sources = []
    
    # 검색 쿼리
    search_queries = [
        f"{product_name} 장단점 실사용",
        f"{product_name} 단점 후기",
        f"{product_name} 장점 리뷰"
    ]
    
    for query in search_queries:
        state["messages"].append(
            AIMessage(content=f"🔍 검색어: '{query}'")
        )
        
        # 네이버 검색
        result = crawler.search_blog(query, display=10)
        if not result or 'items' not in result:
            continue
        
        posts = result['items']
        state["messages"].append(
            AIMessage(content=f"→ {len(posts)}개 포스트 발견")
        )
        
        # 각 포스트 처리
        for idx, post in enumerate(posts[:5]):
            state["messages"].append(
                AIMessage(content=f"📖 분석 중: {post['title'][:40]}...")
            )
            
            # 크롤링
            content = crawler.crawl_content(post['link'])
            if not content:
                continue
            
            crawler.stats['total_crawled'] += 1
            
            # 장단점 추출
            pros_cons = crawler.extract_pros_cons_with_gpt(product_name, content)
            
            if pros_cons:
                all_pros.extend(pros_cons['pros'])
                all_cons.extend(pros_cons['cons'])
                sources.append({
                    'title': post['title'],
                    'link': post['link'],
                    'date': post.get('postdate', '')
                })
                
                state["messages"].append(
                    AIMessage(content=f"✓ 장점 {len(pros_cons['pros'])}개, 단점 {len(pros_cons['cons'])}개 추출")
                )
            
            time.sleep(1)
        
        time.sleep(2)
    
    # 중복 제거 및 정리
    unique_pros = crawler.deduplicate_points(all_pros)
    unique_cons = crawler.deduplicate_points(all_cons)
    
    state["pros"] = unique_pros
    state["cons"] = unique_cons
    state["sources"] = sources[:10]
    
    if state["pros"] or state["cons"]:
        state["messages"].append(
            AIMessage(content=f"🎉 웹 크롤링 완료! 총 장점 {len(state['pros'])}개, 단점 {len(state['cons'])}개 수집")
        )
        
        # DB에 저장
        try:
            supabase = get_supabase_client()
            if supabase:
                data = []
                
                for pro in state["pros"]:
                    data.append({
                        'product_name': product_name,
                        'type': 'pro',
                        'content': pro
                    })
                
                for con in state["cons"]:
                    data.append({
                        'product_name': product_name,
                        'type': 'con',
                        'content': con
                    })
                
                if data:
                    supabase.table('laptop_pros_cons').insert(data).execute()
                    state["messages"].append(
                        AIMessage(content="💾 데이터베이스에 저장 완료!")
                    )
                    st.session_state.saved_products += 1
        except Exception as e:
            state["messages"].append(
                AIMessage(content=f"⚠️ DB 저장 실패: {str(e)}")
            )
    else:
        state["messages"].append(
            AIMessage(content=f"😢 '{product_name}'에 대한 정보를 찾을 수 없습니다.")
        )
    
    # 최종 통계
    state["messages"].append(
        AIMessage(content=f"📊 크롤링 통계: 총 {crawler.stats['total_crawled']}개 페이지, 유효 추출 {crawler.stats['valid_pros_cons']}개")
    )
    
    return state

def process_results(state: SearchState) -> SearchState:
    """결과 처리 및 정리"""
    if state["search_method"] == "database" and state["results"].get("data"):
        # DB 결과 처리
        data = state["results"]["data"]
        state["pros"] = [item['content'] for item in data if item['type'] == 'pro']
        state["cons"] = [item['content'] for item in data if item['type'] == 'con']
        state["sources"] = []
        
        state["messages"].append(
            AIMessage(content=f"📋 결과 정리 완료: 장점 {len(state['pros'])}개, 단점 {len(state['cons'])}개")
        )
    
    return state

def should_search_web(state: SearchState) -> str:
    """웹 검색이 필요한지 판단"""
    if state["results"].get("data"):
        return "process"
    else:
        return "crawl"

# ========================
# LangGraph 워크플로우 생성
# ========================

@st.cache_resource
def create_search_workflow():
    workflow = StateGraph(SearchState)
    
    # 노드 추가
    workflow.add_node("search_db", search_database)
    workflow.add_node("crawl_web", crawl_web)
    workflow.add_node("process", process_results)
    
    # 엣지 설정
    workflow.set_entry_point("search_db")
    workflow.add_conditional_edges(
        "search_db",
        should_search_web,
        {
            "crawl": "crawl_web",
            "process": "process"
        }
    )
    workflow.add_edge("crawl_web", "process")
    workflow.add_edge("process", END)
    
    return workflow.compile()

# 워크플로우 인스턴스 생성
search_app = create_search_workflow()

# ========================
# Streamlit UI
# ========================

# 검색 섹션
st.markdown("""
<style>
    /* 검색 섹션 전체 스타일 */
    .search-section {
        margin-top: -3rem;
        padding: 1rem 0 2rem 0;
    }
    
    /* 검색 제목 스타일 - 위치 상향 조정 */
    .search-title {
        text-align: center;
        color: #333;
        margin-bottom: 1.5rem;
        margin-top: -1rem;
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* 검색 입력창 크기 대폭 확대 및 굵은 글씨 */
    .big-search .stTextInput > div > div > input {
        height: 100px !important;
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        padding: 2rem 3.5rem !important;
        border-radius: 50px !important;
        border: 4px solid #e0e0e0 !important;
        transition: all 0.3s ease !important;
        text-align: center !important;
        letter-spacing: 1px !important;
        line-height: 1.2 !important;
    }
    
    .big-search .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 8px rgba(102, 126, 234, 0.15) !important;
        transform: translateY(-2px) !important;
        border-width: 4px !important;
    }
    
    /* 플레이스홀더 스타일 */
    .big-search .stTextInput > div > div > input::placeholder {
        color: #aaa !important;
        font-size: 1.5rem !important;
        text-align: center !important;
        font-weight: 400 !important;
        opacity: 0.7 !important;
    }
    
    /* 버튼 크기 조정 */
    .search-buttons .stButton > button {
        height: 60px !important;
        font-size: 1.4rem !important;
        padding: 0 3.5rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
    }
    
    /* 검색 카드 조정 */
    .search-card {
        background: transparent !important;
        box-shadow: none !important;
        padding: 0.5rem !important;
    }
    
    /* 인기 검색어 버튼 스타일 */
    .popular-search-buttons .stButton > button {
        height: 45px !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 5, 1])

with col2:
    st.markdown('<div class="search-section">', unsafe_allow_html=True)
    
    # 제목을 위에 배치 - 더 위로 올림
    st.markdown("""
    <h2 class="search-title">
        어떤 제품을 찾고 계신가요?
    </h2>
    """, unsafe_allow_html=True)
    
    # 북마크에서 선택된 항목이 있으면 자동 입력
    default_value = ""
    if 'selected_bookmark' in st.session_state:
        default_value = st.session_state.selected_bookmark
        del st.session_state.selected_bookmark
    elif 'search_query' in st.session_state:
        default_value = st.session_state.search_query
    
    # 더 큰 검색창
    st.markdown('<div class="big-search">', unsafe_allow_html=True)
    product_name = st.text_input(
        "제품명 입력",
        placeholder="예: 맥북 프로 M3, LG 그램 2024, 갤럭시북4 프로",
        value=default_value,
        label_visibility="collapsed",
        key="product_search_input"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 버튼들
    st.markdown('<div class="search-buttons" style="margin-top: 1.8rem;">', unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns([3, 2.5, 0.5])
    with col_btn1:
        search_button = st.button("🔍 검색하기", use_container_width=True, type="primary")
    with col_btn2:
        show_process = st.checkbox("🔧 프로세스 보기", value=True)
    with col_btn3:
        if product_name and st.button("📌", help="북마크에 추가", key="bookmark_btn"):
            if product_name not in st.session_state.bookmarks:
                st.session_state.bookmarks.append(product_name)
                st.success("북마크에 추가되었습니다!")
                st.session_state.total_searches += 1
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 인기 검색어
    st.markdown("""
    <div class="popular-search-buttons" style="text-align: center; margin-top: 2rem;">
        <p style="opacity: 0.7; font-size: 1.2rem; margin-bottom: 1rem; color: #666; font-weight: 500;">인기 검색어</p>
    """, unsafe_allow_html=True)
    
    popular_searches = ["맥북 프로 M3", "LG 그램 2024", "갤럭시북4 프로", "델 XPS 15"]
    cols = st.columns(len(popular_searches))
    for idx, (col, search) in enumerate(zip(cols, popular_searches)):
        with col:
            if st.button(
                search, 
                key=f"popular_{idx}", 
                use_container_width=True,
                help=f"{search} 검색하기"
            ):
                # 검색어를 직접 세션 상태에 저장
                st.session_state.search_query = search
                st.rerun()
    
    st.markdown('</div></div>', unsafe_allow_html=True)

# 검색 실행
if search_button:
    # 인기 검색어로 선택된 경우 해당 검색어 사용
    if 'search_query' in st.session_state and st.session_state.search_query:
        search_term = st.session_state.search_query
        # 검색 후 세션 상태 정리
        st.session_state.search_query = ""
    else:
        search_term = product_name
    
    if search_term:
        loading_placeholder = show_loading_animation()
        
        # LangGraph 실행
        initial_state = {
            "product_name": search_term,
            "search_method": "",
            "results": {},
            "pros": [],
            "cons": [],
            "sources": [],
            "messages": [],
            "error": ""
        }
        
        # 워크플로우 실행
        final_state = search_app.invoke(initial_state)
        
        # 로딩 애니메이션 제거
        loading_placeholder.empty()
    
    # 프로세스 로그 표시
    if show_process and final_state["messages"]:
        with st.expander("🔧 검색 프로세스", expanded=False):
            for msg in final_state["messages"]:
                if isinstance(msg, HumanMessage):
                    st.write(f"👤 {msg.content}")
                else:
                    st.write(f"🤖 {msg.content}")
    
    # 결과 표시
    if final_state["pros"] or final_state["cons"]:
        # 검색 정보
        st.markdown(f"""
        <div class="process-info fade-in">
            <strong><i class="fas fa-info-circle"></i> 검색 방법:</strong> {
                '데이터베이스' if final_state["search_method"] == "database" else '웹 크롤링'
            } | 
            <strong><i class="fas fa-thumbs-up"></i> 장점:</strong> {len(final_state["pros"])}개 | 
            <strong><i class="fas fa-thumbs-down"></i> 단점:</strong> {len(final_state["cons"])}개
        </div>
        """, unsafe_allow_html=True)
        
        # 장단점 상세 표시 (워드클라우드보다 먼저)
        st.markdown("---")
        st.markdown("### 📋 상세 분석 결과")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="pros-section fade-in">
                <h3 style="color: #28a745; margin-bottom: 1.5rem;">
                    <i class="fas fa-check-circle"></i> 장점
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            if final_state["pros"]:
                for idx, pro in enumerate(final_state["pros"], 1):
                    st.markdown(f"""
                    <div class="pros-item">
                        <span style="color: #28a745; font-weight: bold;">
                            <i class="fas fa-check"></i> {idx}.
                        </span> {pro}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("장점 정보가 없습니다.")
        
        with col2:
            st.markdown("""
            <div class="cons-section fade-in">
                <h3 style="color: #dc3545; margin-bottom: 1.5rem;">
                    <i class="fas fa-times-circle"></i> 단점
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            if final_state["cons"]:
                for idx, con in enumerate(final_state["cons"], 1):
                    st.markdown(f"""
                    <div class="cons-item">
                        <span style="color: #dc3545; font-weight: bold;">
                            <i class="fas fa-times"></i> {idx}.
                        </span> {con}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("단점 정보가 없습니다.")
        
        # 워드클라우드 표시
        st.markdown("---")
        st.markdown("### 🔤 키워드 분석")
        display_wordclouds(final_state["pros"], final_state["cons"])
        
        # 심층 분석 섹션 - 수정된 부분
        st.markdown("---")
        st.markdown("### 📊 심층 분석")
        
        # 카테고리별 장단점 분포 (레이더 차트)와 해석
        col1, col2 = st.columns([1, 1])
        
        with col1:
            comparison_chart = create_comparison_chart(final_state["pros"], final_state["cons"])
            if comparison_chart:
                st.plotly_chart(comparison_chart, use_container_width=True)
            else:
                st.info("카테고리별 분석을 위한 데이터가 부족합니다.")
        
        with col2:
            # 레이더 차트 해석 섹션
            if final_state["pros"] or final_state["cons"]:
                # 카테고리별 분석을 위한 데이터 추출
                categories = {
                    '성능': ['성능', '속도', '빠르', '느리', '렉', '버벅', '프로세서', 'CPU', 'GPU', '메모리'],
                    '디자인': ['디자인', '외관', '예쁘', '이쁘', '못생', '색상', '모양', '두께', '얇'],
                    '가격': ['가격', '비싸', '저렴', '가성비', '비용', '돈', '할인', '세일'],
                    '품질': ['품질', '마감', '재질', '튼튼', '약하', '고장', '내구성', '견고'],
                    '기능': ['기능', '편의', '편리', '불편', '사용', '조작', '인터페이스'],
                    '배터리': ['배터리', '충전', '전원', '지속', '방전'],
                    '화면': ['화면', '디스플레이', '선명', '밝기', '해상도'],
                    '기타': []
                }
                
                # 각 카테고리별 장단점 수 계산
                category_pros = {cat: 0 for cat in categories}
                category_cons = {cat: 0 for cat in categories}
                
                # 장점 분류
                for pro in final_state["pros"]:
                    categorized = False
                    for cat, keywords in categories.items():
                        if cat != '기타' and any(keyword in pro for keyword in keywords):
                            category_pros[cat] += 1
                            categorized = True
                            break
                    if not categorized:
                        category_pros['기타'] += 1
                
                # 단점 분류
                for con in final_state["cons"]:
                    categorized = False
                    for cat, keywords in categories.items():
                        if cat != '기타' and any(keyword in con for keyword in keywords):
                            category_cons[cat] += 1
                            categorized = True
                            break
                    if not categorized:
                        category_cons['기타'] += 1
                
                # 가장 강한 장점 카테고리
                strongest_pro_cat = max(category_pros.items(), key=lambda x: x[1])
                # 가장 큰 단점 카테고리  
                strongest_con_cat = max(category_cons.items(), key=lambda x: x[1])
                
                # 균형잡힌 카테고리 (장단점 차이가 적은)
                balanced_categories = []
                for cat in categories:
                    if category_pros[cat] > 0 and category_cons[cat] > 0:
                        diff = abs(category_pros[cat] - category_cons[cat])
                        if diff <= 1:
                            balanced_categories.append(cat)
                
                st.markdown("""
                <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                            padding: 2rem; border-radius: 20px; border-left: 5px solid #667eea;">
                    <h4 style="color: #667eea; margin-bottom: 1.5rem; text-align: center;">
                        <i class="fas fa-chart-line"></i> 카테고리별 분석 인사이트
                    </h4>
                """, unsafe_allow_html=True)
                
                # 주요 강점 분석
                if strongest_pro_cat[1] > 0:
                    st.markdown(f"""
                    <div style="background: rgba(40, 167, 69, 0.1); padding: 1.2rem; 
                                border-radius: 12px; margin-bottom: 1rem; border-left: 4px solid #28a745;">
                        <h5 style="color: #28a745; margin-bottom: 0.8rem;">
                            <i class="fas fa-star"></i> 최고 강점 영역
                        </h5>
                        <p style="margin: 0; line-height: 1.6; color: #2d5016;">
                            <strong>"{strongest_pro_cat[0]}"</strong> 분야에서 가장 높은 평가를 받고 있습니다. 
                            총 <strong>{strongest_pro_cat[1]}개</strong>의 긍정적인 의견이 집중되어 있어, 
                            이 제품의 핵심 경쟁력으로 보입니다.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 주요 개선점 분석
                if strongest_con_cat[1] > 0:
                    st.markdown(f"""
                    <div style="background: rgba(220, 53, 69, 0.1); padding: 1.2rem; 
                                border-radius: 12px; margin-bottom: 1rem; border-left: 4px solid #dc3545;">
                        <h5 style="color: #dc3545; margin-bottom: 0.8rem;">
                            <i class="fas fa-exclamation-triangle"></i> 주요 개선 필요 영역
                        </h5>
                        <p style="margin: 0; line-height: 1.6; color: #721c24;">
                            <strong>"{strongest_con_cat[0]}"</strong> 부분에서 가장 많은 불만이 제기되고 있습니다. 
                            총 <strong>{strongest_con_cat[1]}개</strong>의 개선 요청이 있어, 
                            구매 전 신중한 검토가 필요한 영역입니다.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 균형잡힌 영역 분석
                if balanced_categories:
                    st.markdown(f"""
                    <div style="background: rgba(255, 193, 7, 0.1); padding: 1.2rem; 
                                border-radius: 12px; margin-bottom: 1rem; border-left: 4px solid #ffc107;">
                        <h5 style="color: #d39e00; margin-bottom: 0.8rem;">
                            <i class="fas fa-balance-scale"></i> 균형잡힌 영역
                        </h5>
                        <p style="margin: 0; line-height: 1.6; color: #533f03;">
                            <strong>{', '.join(balanced_categories[:2])}</strong> 영역에서는 장단점이 고르게 나타나고 있습니다. 
                            개인의 사용 패턴과 선호도에 따라 만족도가 달라질 수 있는 부분입니다.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 전체적인 평가
                total_pros = sum(category_pros.values())
                total_cons = sum(category_cons.values())
                if total_pros > 0 or total_cons > 0:
                    positive_ratio = (total_pros / (total_pros + total_cons)) * 100
                    
                    if positive_ratio >= 70:
                        overall_sentiment = "매우 긍정적"
                        sentiment_color = "#28a745"
                        sentiment_icon = "fa-thumbs-up"
                        recommendation = "강력 추천합니다"
                    elif positive_ratio >= 55:
                        overall_sentiment = "긍정적"
                        sentiment_color = "#20c997"
                        sentiment_icon = "fa-smile"
                        recommendation = "추천합니다"
                    elif positive_ratio >= 45:
                        overall_sentiment = "중립적"
                        sentiment_color = "#ffc107"
                        sentiment_icon = "fa-meh"
                        recommendation = "신중한 검토 후 구매하세요"
                    else:
                        overall_sentiment = "개선 필요"
                        sentiment_color = "#dc3545"
                        sentiment_icon = "fa-frown"
                        recommendation = "다른 대안을 고려해보세요"
                    
                    st.markdown(f"""
                    <div style="background: rgba(108, 117, 125, 0.1); padding: 1.2rem; 
                                border-radius: 12px; text-align: center; border: 2px solid {sentiment_color};">
                        <h5 style="color: {sentiment_color}; margin-bottom: 0.8rem;">
                            <i class="fas {sentiment_icon}"></i> 종합 평가
                        </h5>
                        <p style="margin: 0; line-height: 1.6; color: #495057; font-size: 1.1rem;">
                            전체적으로 <strong style="color: {sentiment_color};">{overall_sentiment}</strong>인 평가를 받고 있으며, 
                            <br>구매를 고려 중이시라면 <strong>{recommendation}</strong>.
                        </p>
                        <div style="margin-top: 1rem; padding: 0.8rem; background: rgba(255,255,255,0.5); 
                                    border-radius: 8px;">
                            <span style="font-size: 0.9rem; color: #6c757d;">
                                긍정 의견 <strong>{positive_ratio:.1f}%</strong> | 
                                부정 의견 <strong>{100-positive_ratio:.1f}%</strong>
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("분석할 데이터가 부족합니다.")
        
        # 요약 통계
        metrics = create_summary_metrics(final_state["pros"], final_state["cons"])
        
        st.markdown("---")
        st.markdown("### 📈 분석 인사이트")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); 
                        padding: 1.5rem; border-radius: 15px; text-align: center; height: 120px;">
                <i class="fas fa-comments" style="font-size: 2rem; color: #1976d2;"></i>
                <h3 style="margin: 0.5rem 0; color: #1976d2;">{metrics['total_reviews']}</h3>
                <p style="margin: 0; font-size: 0.9rem; color: #555;">전체 리뷰 수</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); 
                        padding: 1.5rem; border-radius: 15px; text-align: center; height: 120px;">
                <i class="fas fa-percentage" style="font-size: 2rem; color: #388e3c;"></i>
                <h3 style="margin: 0.5rem 0; color: #388e3c;">{metrics['positive_ratio']:.0f}%</h3>
                <p style="margin: 0; font-size: 0.9rem; color: #555;">긍정 비율</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%); 
                        padding: 1.5rem; border-radius: 15px; text-align: center; height: 120px;">
                <i class="fas fa-lightbulb" style="font-size: 2rem; color: #7b1fa2;"></i>
                <h3 style="margin: 0.5rem 0; color: #7b1fa2;">{metrics['diversity_score']}</h3>
                <p style="margin: 0; font-size: 0.9rem; color: #555;">키워드 다양성</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            balance_score = 100 - abs(len(final_state['pros']) - len(final_state['cons'])) / max(len(final_state['pros']), len(final_state['cons']), 1) * 100
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); 
                        padding: 1.5rem; border-radius: 15px; text-align: center; height: 120px;">
                <i class="fas fa-balance-scale" style="font-size: 2rem; color: #f57c00;"></i>
                <h3 style="margin: 0.5rem 0; color: #f57c00;">{balance_score:.0f}%</h3>
                <p style="margin: 0; font-size: 0.9rem; color: #555;">균형 지수</p>
            </div>
            """, unsafe_allow_html=True)
        
        # 추가 인사이트
        st.markdown("---")
        
        # 주요 발견사항
        col1, col2 = st.columns(2)
        
        with col1:
            # 장점에서 가장 많이 언급된 구체적인 키워드 추출
            pros_keywords = extract_keywords(final_state["pros"])
            if pros_keywords and isinstance(pros_keywords, dict):
                # 제품 특성과 관련된 키워드만 필터링
                product_keywords = {
                    k: v for k, v in pros_keywords.items() 
                    if len(k) >= 2 and not any(skip in k for skip in ['언급', '회', '개', '점'])
                }
                if product_keywords:
                    sorted_keywords = sorted(product_keywords.items(), key=lambda x: x[1], reverse=True)[:3]
                    top_pros_keywords = sorted_keywords
                else:
                    top_pros_keywords = []
            else:
                top_pros_keywords = []
            
            st.markdown(f"""
            <div style="background: rgba(40, 167, 69, 0.1); padding: 1.5rem; border-radius: 15px; 
                        border-left: 4px solid #28a745;">
                <h5 style="color: #28a745; margin-bottom: 1rem;">
                    <i class="fas fa-star"></i> 핵심 강점
                </h5>
                <ul style="margin: 0; padding-left: 1.5rem;">
            """, unsafe_allow_html=True)
            
            if top_pros_keywords:
                for keyword, count in top_pros_keywords:
                    # 키워드가 포함된 원본 문장 찾기
                    related_sentences = [pro for pro in final_state["pros"] if keyword in pro]
                    if related_sentences:
                        # 가장 대표적인 문장 선택
                        representative = min(related_sentences, key=len)
                        # 키워드 부분을 강조
                        highlighted = representative.replace(keyword, f"<strong>{keyword}</strong>")
                        st.markdown(f"<li>{highlighted}</li>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<li><strong>{keyword}</strong> 관련 특징</li>", unsafe_allow_html=True)
            else:
                # 키워드가 없을 경우 원본 장점 중 짧은 것 3개 표시
                short_pros = sorted(final_state["pros"], key=len)[:3]
                for pro in short_pros:
                    st.markdown(f"<li>{pro}</li>", unsafe_allow_html=True)
            
            st.markdown("</ul></div>", unsafe_allow_html=True)
        
        with col2:
            # 단점에서 가장 많이 언급된 구체적인 키워드 추출
            cons_keywords = extract_keywords(final_state["cons"])
            if cons_keywords and isinstance(cons_keywords, dict):
                # 제품 특성과 관련된 키워드만 필터링
                product_keywords = {
                    k: v for k, v in cons_keywords.items() 
                    if len(k) >= 2 and not any(skip in k for skip in ['언급', '회', '개', '점'])
                }
                if product_keywords:
                    sorted_keywords = sorted(product_keywords.items(), key=lambda x: x[1], reverse=True)[:3]
                    top_cons_keywords = sorted_keywords
                else:
                    top_cons_keywords = []
            else:
                top_cons_keywords = []
            
            st.markdown(f"""
            <div style="background: rgba(220, 53, 69, 0.1); padding: 1.5rem; border-radius: 15px; 
                        border-left: 4px solid #dc3545;">
                <h5 style="color: #dc3545; margin-bottom: 1rem;">
                    <i class="fas fa-exclamation-triangle"></i> 주요 개선점
                </h5>
                <ul style="margin: 0; padding-left: 1.5rem;">
            """, unsafe_allow_html=True)
            
            if top_cons_keywords:
                for keyword, count in top_cons_keywords:
                    # 키워드가 포함된 원본 문장 찾기
                    related_sentences = [con for con in final_state["cons"] if keyword in con]
                    if related_sentences:
                        # 가장 대표적인 문장 선택
                        representative = min(related_sentences, key=len)
                        # 키워드 부분을 강조
                        highlighted = representative.replace(keyword, f"<strong>{keyword}</strong>")
                        st.markdown(f"<li>{highlighted}</li>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<li><strong>{keyword}</strong> 관련 문제</li>", unsafe_allow_html=True)
            else:
                # 키워드가 없을 경우 원본 단점 중 짧은 것 3개 표시
                short_cons = sorted(final_state["cons"], key=len)[:3]
                for con in short_cons:
                    st.markdown(f"<li>{con}</li>", unsafe_allow_html=True)
            
            st.markdown("</ul></div>", unsafe_allow_html=True)
        
        # 출처 (웹 크롤링인 경우)
        if final_state["sources"]:
            with st.expander("📚 출처 보기"):
                for idx, source in enumerate(final_state["sources"], 1):
                    st.markdown(f"""
                    <div style="padding: 0.5rem; margin: 0.3rem 0;">
                        <i class="fas fa-link"></i> {idx}. 
                        <a href="{source['link']}" target="_blank" style="color: {secondary_text};">
                            {source['title']}
                        </a>
                    </div>
                    """, unsafe_allow_html=True)
        
        # 통계 카드
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <i class="fas fa-thumbs-up" style="font-size: 2rem; color: #28a745;"></i>
                <h3 style="margin: 0.5rem 0;">{}</h3>
                <p style="margin: 0; opacity: 0.7;">총 장점</p>
            </div>
            """.format(len(final_state['pros'])), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <i class="fas fa-thumbs-down" style="font-size: 2rem; color: #dc3545;"></i>
                <h3 style="margin: 0.5rem 0;">{}</h3>
                <p style="margin: 0; opacity: 0.7;">총 단점</p>
            </div>
            """.format(len(final_state['cons'])), unsafe_allow_html=True)
        
        with col3:
            icon = "fa-database" if final_state["search_method"] == "database" else "fa-globe"
            st.markdown("""
            <div class="metric-card">
                <i class="fas {}" style="font-size: 2rem; color: #2196f3;"></i>
                <h3 style="margin: 0.5rem 0;">{}</h3>
                <p style="margin: 0; opacity: 0.7;">검색 방법</p>
            </div>
            """.format(icon, "DB" if final_state["search_method"] == "database" else "웹"), unsafe_allow_html=True)
        
        with col4:
            total_score = len(final_state['pros']) / (len(final_state['pros']) + len(final_state['cons'])) * 100 if (len(final_state['pros']) + len(final_state['cons'])) > 0 else 0
            st.markdown("""
            <div class="metric-card">
                <i class="fas fa-star" style="font-size: 2rem; color: #ffc107;"></i>
                <h3 style="margin: 0.5rem 0;">{:.0f}%</h3>
                <p style="margin: 0; opacity: 0.7;">긍정 비율</p>
            </div>
            """.format(total_score), unsafe_allow_html=True)
        
        # 공유 버튼
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            share_text = f"{product_name} 분석 결과: 장점 {len(final_state['pros'])}개, 단점 {len(final_state['cons'])}개"
            st.markdown(f"""
            <div style="text-align: center;">
                <a href="https://twitter.com/intent/tweet?text={share_text}" target="_blank" 
                   style="margin: 0 10px; color: #1DA1F2;">
                    <i class="fab fa-twitter" style="font-size: 1.5rem;"></i>
                </a>
                <a href="https://www.facebook.com/sharer/sharer.php?u=#" target="_blank" 
                   style="margin: 0 10px; color: #4267B2;">
                    <i class="fab fa-facebook" style="font-size: 1.5rem;"></i>
                </a>
                <button onclick="navigator.clipboard.writeText('{share_text}')" 
                        style="margin: 0 10px; background: none; border: none; cursor: pointer;">
                    <i class="fas fa-link" style="font-size: 1.5rem; color: #666;"></i>
                </button>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error(f"'{product_name}'에 대한 정보를 찾을 수 없습니다.")

# 하단 정보
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="metric-card">
        <i class="fas fa-brain" style="color: #667eea;"></i>
        <p>LangGraph로 구현된<br>체계적인 검색 프로세스</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="metric-card">
        <i class="fas fa-sync-alt" style="color: #28a745;"></i>
        <p>DB 우선 검색<br>→ 없으면 웹 크롤링</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="metric-card">
        <i class="fas fa-save" style="color: #dc3545;"></i>
        <p>검색 결과<br>자동 저장</p>
    </div>
    """, unsafe_allow_html=True)

current_date = datetime.now().strftime('%Y년 %m월 %d일')
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 2rem; margin-top: 2rem;">
    <p style="margin-bottom: 0.5rem;">
        <i class="fas fa-clock"></i> 마지막 업데이트: {current_date}
    </p>
    <p style="font-size: 0.9rem; opacity: 0.8;">
        Powered by LangGraph & OpenAI | Made with <i class="fas fa-heart" style="color: #e74c3c;"></i> by Smart Shopping Team
    </p>
</div>
""", unsafe_allow_html=True)
