import streamlit as st

# 페이지 설정 (반드시 첫 번째로 실행)
st.set_page_config(
    page_title="커리어 인사이트 (LangGraph)",
    page_icon="💼",
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
    os.environ["LANGCHAIN_PROJECT"] = "career-insight-app"
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
if 'saved_careers' not in st.session_state:
    st.session_state.saved_careers = 0

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
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
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
    
    /* pulse와 sparkle 애니메이션 추가 */
    @keyframes pulse {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.02); }}
        100% {{ transform: scale(1); }}
    }}
    
    @keyframes sparkle {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    
    /* 검색 섹션 스타일 */
    .search-section {{
        margin-top: -3rem;
        padding: 1rem 0 2rem 0;
    }}
    
    .search-title {{
        text-align: center;
        color: {text_color};
        margin-bottom: 1.5rem;
        margin-top: -1rem;
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }}
    
    /* 검색 입력창 크기 대폭 확대 및 굵은 글씨 */
    .big-search .stTextInput > div > div > input {{
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
    }}
    
    .big-search .stTextInput > div > div > input:focus {{
        border-color: {secondary_text} !important;
        box-shadow: 0 0 0 8px rgba(102, 126, 234, 0.15) !important;
        transform: translateY(-2px) !important;
        border-width: 4px !important;
    }}
    
    /* 플레이스홀더 스타일 */
    .big-search .stTextInput > div > div > input::placeholder {{
        color: #aaa !important;
        font-size: 1.5rem !important;
        text-align: center !important;
        font-weight: 400 !important;
        opacity: 0.7 !important;
    }}
    
    /* 버튼 크기 조정 */
    .search-buttons .stButton > button {{
        height: 60px !important;
        font-size: 1.4rem !important;
        padding: 0 3.5rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
    }}
    
    /* 인기 검색어 버튼 스타일 */
    .popular-search-buttons .stButton > button {{
        height: 45px !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
    }}
    
    /* 직업 관련 섹션 스타일 */
    .career-insight {{
        background: linear-gradient(135deg, #f0f4ff 0%, #e0e7ff 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: none;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.1);
    }}
    
    .salary-section {{
        background: linear-gradient(135deg, #fff9e6 0%, #ffe5b4 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: none;
        box-shadow: 0 5px 15px rgba(255, 193, 7, 0.1);
    }}
    
    .career-path {{
        background: linear-gradient(135deg, #e6f3ff 0%, #c5e0ff 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: none;
        box-shadow: 0 5px 15px rgba(33, 150, 243, 0.1);
    }}
    
    /* 채용 공고 카드 호버 효과 */
    .job-posting-card {{
        transition: all 0.3s ease;
        cursor: pointer;
    }}
    
    .job-posting-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }}
    
    /* 지원하기 버튼 호버 효과 */
    .apply-button {{
        transition: all 0.3s ease;
    }}
    
    .apply-button:hover {{
        transform: translateX(5px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }}
</style>
""", unsafe_allow_html=True)

# 사이드바 설정
with st.sidebar:
    # 인프런 SQL 강의 광고 배너 - 간단한 버전
    st.markdown("### ⛰️ 모든 IT 직종의 공통 과목 SQL")
    st.markdown("**데이터 분석의 시작**")
    st.markdown("**실무 SQL 완전정복**")
    st.markdown("🔥온라인으로 편하게 수강하세요")
    
    # 링크 버튼
    st.link_button(
        "수강하러 가기 →",
        "https://inf.run/R9Te3",
        use_container_width=True
    )
    
    st.markdown("---")
    
    
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
    st.metric("저장된 직업", f"{st.session_state.saved_careers}개")
    
# 헤더 - 커리어 인사이트
st.markdown("""
<div class="main-header">
    <h1 style="margin-bottom: 0.5rem;">💼 커리어 인사이트 (LangGraph Edition)</h1>
    <p style="font-size: 1.2rem; margin-top: 0.5rem;">
        LangGraph로 구현한 지능형 직업 장단점 분석 시스템
    </p>
    <p style="font-size: 0.9rem; margin-top: 0.3rem; opacity: 0.8;">
        <i class="fas fa-robot"></i> AI가 다양한 직업 정보를 분석하여 현실적인 장단점을 제공합니다
    </p>
</div>
""", unsafe_allow_html=True)

# ========================
# LangGraph State 정의
# ========================

class CareerState(TypedDict):
    """직업 분석 프로세스의 상태"""
    career_name: str
    search_method: str  # "database" or "web_crawling"
    results: dict
    pros: List[str]
    cons: List[str]
    sources: List[dict]
    salary_info: dict
    career_path: List[str]
    messages: Annotated[List[Union[HumanMessage, AIMessage]], operator.add]
    error: str

# ========================
# 크롤링 클래스
# ========================

class CareerInfoCrawler:
    def __init__(self, naver_client_id, naver_client_secret):
        self.naver_headers = {
            "X-Naver-Client-Id": naver_client_id,
            "X-Naver-Client-Secret": naver_client_secret
        }
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        
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
    
    def search_career_info(self, query, display=20):
        """네이버 검색 API를 통해 직업 정보 검색"""
        all_results = []
        
        # 직업 관련 다양한 검색어 조합
        search_queries = [
            f"{query} 직업 장단점",
            f"{query} 현실 단점",
            f"{query} 실제 장점",
            f"{query} 연봉 워라밸",
            f"{query} 직업 후기",
            f"{query} 직업 현실",
            f"{query} 일하면서 느낀점",
            f"{query} 직업 추천",
            f"{query} 직업 경험담",
            f"{query} 커리어 조언"
        ]
        
        # 블로그와 뉴스 모두 검색
        search_types = [
            ("blog", "https://openapi.naver.com/v1/search/blog"),
            ("news", "https://openapi.naver.com/v1/search/news")
        ]
        
        for search_type, url in search_types:
            for search_query in search_queries[:5]:  # 각 타입별로 5개 쿼리만 사용
                params = {
                    "query": search_query,
                    "display": 10,
                    "sort": "sim"
                }
                
                try:
                    response = requests.get(url, headers=self.naver_headers, params=params)
                    if response.status_code == 200:
                        result = response.json()
                        for item in result.get('items', []):
                            item['title'] = self.remove_html_tags(item['title'])
                            item['description'] = self.remove_html_tags(item['description'])
                            item['search_type'] = search_type  # 블로그인지 뉴스인지 구분
                        all_results.extend(result.get('items', []))
                    time.sleep(0.1)  # API 호출 제한을 위한 짧은 대기
                except Exception as e:
                    print(f"{search_type} 검색 오류: {e}")
        
        # 중복 제거 (제목 기준)
        seen_titles = set()
        unique_results = []
        for item in all_results:
            if item['title'] not in seen_titles:
                seen_titles.add(item['title'])
                unique_results.append(item)
        
        return unique_results[:30]  # 최대 30개 결과 반환
    
    def crawl_content(self, url):
        """블로그 및 뉴스 본문 크롤링"""
        try:
            # 네이버 블로그 처리
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
            
            # 일반 웹페이지 및 뉴스 처리
            else:
                response = requests.get(url, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # 뉴스 기사 본문 추출 시도
                    content = ""
                    article_selectors = [
                        'article', 'div.article_body', 'div.news_body', 
                        'div.content', 'main', 'div#articleBody',
                        'div.article_content', 'div.news_content'
                    ]
                    
                    for selector in article_selectors:
                        elem = soup.select_one(selector)
                        if elem:
                            content = elem.get_text(separator='\n', strip=True)
                            break
                    
                    if not content:
                        # 일반적인 텍스트 추출
                        content = soup.get_text(separator='\n', strip=True)
                    
                    content = re.sub(r'\s+', ' ', content)
                    content = content.replace('\u200b', '')
                    
                    return content if len(content) > 300 else None
                    
        except Exception as e:
            print(f"크롤링 오류: {e}")
        return None
    
    def extract_career_pros_cons_with_gpt(self, career_name, content):
        """ChatGPT로 직업 장단점 추출"""
        if not content or len(content) < 200 or not self.openai_client:
            return None
        
        content_preview = content[:2000]
        
        prompt = f"""다음은 "{career_name}" 직업에 대한 블로그 글입니다.

[블로그 내용]
{content_preview}

위 내용에서 {career_name} 직업의 현실적인 장점과 단점을 추출해주세요.
실제 경험에 기반한 구체적인 내용만 포함하세요.

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
                        "content": "당신은 직업 상담 전문가입니다. 각 직업의 현실적인 장단점을 객관적으로 분석합니다."
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
    
    def extract_career_pros_cons_simple(self, career_name, content):
        """키워드 기반 간단한 장단점 추출 (GPT API 없을 때 사용)"""
        if not content or len(content) < 200:
            return None
        
        content_lower = content.lower()
        
        # 장점 관련 키워드
        pros_keywords = [
            '장점', '좋은점', '좋은 점', '메리트', '이점', '강점',
            '좋다', '좋았다', '좋습니다', '만족', '추천',
            '높은 연봉', '워라밸', '안정적', '성장', '발전',
            '보람', '재미있', '흥미로', '유연한'
        ]
        
        # 단점 관련 키워드
        cons_keywords = [
            '단점', '나쁜점', '나쁜 점', '어려운점', '힘든점',
            '어렵다', '힘들다', '스트레스', '야근', '박봉',
            '불안정', '경쟁', '부담', '압박', '피곤',
            '지루', '반복적', '단순'
        ]
        
        pros = []
        cons = []
        
        # 문장 단위로 분리
        sentences = re.split(r'[.!?]\s*', content)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10 or len(sentence) > 200:
                continue
            
            sentence_lower = sentence.lower()
            
            # 장점 추출
            for keyword in pros_keywords:
                if keyword in sentence_lower and career_name.lower() in sentence_lower:
                    if len(pros) < 5 and sentence not in pros:
                        pros.append(sentence)
                        break
            
            # 단점 추출
            for keyword in cons_keywords:
                if keyword in sentence_lower and career_name.lower() in sentence_lower:
                    if len(cons) < 5 and sentence not in cons:
                        cons.append(sentence)
                        break
        
        if pros or cons:
            return {
                'pros': pros[:3],
                'cons': cons[:3]
            }
        
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
    
    def get_career_salary_info(self, career_name):
        """직업 연봉 정보 추출 (샘플)"""
        # 실제로는 API나 크롤링으로 데이터를 가져와야 함
        # 여기서는 샘플 데이터 제공
        sample_salaries = {
            "데이터 분석가": {"min": 4000, "avg": 6000, "max": 9000},
            "데이터 엔지니어": {"min": 5000, "avg": 7500, "max": 12000},
            "DBA": {"min": 4500, "avg": 7000, "max": 11000},
            "DB 엔지니어": {"min": 4500, "avg": 6500, "max": 10000},
            "AI 개발자": {"min": 5500, "avg": 8000, "max": 15000},
            "AI 엔지니어": {"min": 6000, "avg": 9000, "max": 18000},
            "자바 개발자": {"min": 4000, "avg": 6000, "max": 10000},
            "백엔드 개발자": {"min": 4500, "avg": 6500, "max": 11000}
        }
        
        # 기본값
        default_salary = {"min": 4000, "avg": 6000, "max": 9000}
        
        for key, value in sample_salaries.items():
            if key in career_name:
                return value
        
        return default_salary
    
    def get_career_path(self, career_name):
        """경력 개발 경로 제공 (샘플)"""
        sample_paths = {
            "데이터 분석가": ["주니어 분석가", "분석가", "시니어 분석가", "리드 분석가", "데이터 팀장", "CDO"],
            "데이터 엔지니어": ["주니어 엔지니어", "엔지니어", "시니어 엔지니어", "리드 엔지니어", "데이터 아키텍트", "CTO"],
            "DBA": ["주니어 DBA", "DBA", "시니어 DBA", "DB 팀장", "DB 아키텍트", "CTO"],
            "DB 엔지니어": ["주니어 엔지니어", "DB 엔지니어", "시니어 엔지니어", "DB 아키텍트", "솔루션 아키텍트"],
            "AI 개발자": ["주니어 개발자", "AI 개발자", "시니어 개발자", "AI 리드", "AI 팀장", "CTO"],
            "AI 엔지니어": ["주니어 엔지니어", "AI 엔지니어", "시니어 엔지니어", "MLOps 리드", "AI 플랫폼 팀장"],
            "자바 개발자": ["주니어 개발자", "개발자", "시니어 개발자", "테크 리드", "개발 팀장", "CTO"],
            "백엔드 개발자": ["주니어 개발자", "개발자", "시니어 개발자", "테크 리드", "백엔드 팀장", "CTO"]
        }
        
        default_path = ["신입", "경력 3년차", "경력 5년차", "팀장급", "임원급"]
        
        for key, value in sample_paths.items():
            if key in career_name:
                return value
        
        return default_path

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
            <i class="fas fa-brain"></i> AI가 직업 정보를 분석하고 있습니다...
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
    # 직업 관련 불용어
    stopwords = {
        '수', '있습니다', '있어요', '있음', '좋습니다', '좋아요', '좋음', 
        '나쁩니다', '나빠요', '나쁨', '않습니다', '않아요', '않음',
        '입니다', '이다', '되다', '하다', '있다', '없다', '같다',
        '직업', '일', '업무', '근무', '회사', '직장', '분야',
        '위해', '통해', '대해', '매우', '정말', '너무', '조금',
        '그리고', '하지만', '그러나', '또한', '때문', '경우',
        '제공합니다', '제공', '합니다', '해요', '드립니다', '드려요'
    }
    
    # 모든 텍스트를 결합하고 키워드 추출
    all_text = ' '.join(texts)
    
    # 한글만 추출
    words = re.findall(r'[가-힣]+', all_text)
    
    # 필터링
    filtered_words = []
    for word in words:
        if (len(word) >= 2 and 
            word not in stopwords and
            not word.endswith('습니다') and
            not word.endswith('합니다')):
            filtered_words.append(word)
    
    # 단어 빈도 계산
    word_freq = Counter(filtered_words)
    
    # 빈도수가 1인 단어는 제외
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
    
    # 빈도수 기준으로 상위 키워드만 선택
    top_keywords = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:30])
    
    # matplotlib 한글 폰트 설정
    font_path = "./NanumGothic.ttf"
    
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
                relative_scaling=0.7,
                min_font_size=14,
                max_words=30,
                prefer_horizontal=0.8,
                margin=15,
                collocations=False
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
        st.warning(f"한글 폰트를 찾을 수 없습니다.")
        return None

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
            
            pros_wordcloud = create_wordcloud(pros, "", "Greens")
            if pros_wordcloud:
                st.image(pros_wordcloud, use_container_width=True)
    
    with col2:
        if cons:
            st.markdown("""
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #ffd6d6 0%, #ffb8b8 100%); border-radius: 15px;">
                <h3 style="color: #dc3545; margin: 0;">
                    <i class="fas fa-times-circle"></i> 단점 키워드
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            cons_wordcloud = create_wordcloud(cons, "", "Reds")
            if cons_wordcloud:
                st.image(cons_wordcloud, use_container_width=True)

def create_salary_chart(salary_info, career_name):
    """연봉 차트 생성"""
    fig = go.Figure()
    
    # 막대 그래프
    fig.add_trace(go.Bar(
        x=['최소', '평균', '최대'],
        y=[salary_info['min'], salary_info['avg'], salary_info['max']],
        marker_color=['#dc3545', '#ffc107', '#28a745'],
        text=[f"{salary_info['min']:,}만원", f"{salary_info['avg']:,}만원", f"{salary_info['max']:,}만원"],
        textposition='auto',
        hovertemplate='%{x}: %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': f'💰 {career_name} 연봉 범위',
            'font': {'size': 20},
            'x': 0.5,
            'xanchor': 'center'
        },
        height=400,
        margin=dict(l=0, r=0, t=50, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14),
        showlegend=False,
        yaxis=dict(
            title='연봉 (만원)',
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        xaxis=dict(showgrid=False)
    )
    
    return fig

def create_career_path_timeline(career_path):
    """경력 개발 타임라인 생성"""
    fig = go.Figure()
    
    # 각 단계별 연차 추정 (IT 직군 기준)
    years = [0, 2, 5, 8, 12, 15]
    
    # 타임라인 생성
    fig.add_trace(go.Scatter(
        x=years[:len(career_path)],
        y=[1] * len(career_path),
        mode='markers+text',
        marker=dict(
            size=40,
            color=list(range(len(career_path))),
            colorscale='Viridis',
            showscale=False
        ),
        text=career_path,
        textposition="top center",
        textfont=dict(size=12),
        hovertemplate='%{text}<br>예상 연차: %{x}년<extra></extra>'
    ))
    
    # 연결선 추가
    fig.add_trace(go.Scatter(
        x=years[:len(career_path)],
        y=[1] * len(career_path),
        mode='lines',
        line=dict(color='lightgray', width=2),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title={
            'text': '🎯 경력 개발 경로',
            'font': {'size': 20},
            'x': 0.5,
            'xanchor': 'center'
        },
        height=300,
        margin=dict(l=0, r=0, t=50, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title='연차',
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            range=[-1, max(years) + 1]
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            range=[0.5, 1.5]
        ),
        showlegend=False
    )
    
    return fig

# ========================
# LangGraph 노드 함수들
# ========================

# 클라이언트 초기화
@st.cache_resource
def get_supabase_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

@st.cache_resource
def get_crawler():
    return CareerInfoCrawler(NAVER_CLIENT_ID, NAVER_CLIENT_SECRET) if NAVER_CLIENT_ID and NAVER_CLIENT_SECRET else None

def search_database(state: CareerState) -> CareerState:
    """데이터베이스에서 직업 정보 검색"""
    career_name = state["career_name"]
    supabase = get_supabase_client()
    
    if not supabase:
        state["messages"].append(
            AIMessage(content="⚠️ 데이터베이스가 설정되지 않았습니다. 웹 검색으로 진행합니다.")
        )
        state["results"] = {"data": None}
        return state
    
    state["messages"].append(
        HumanMessage(content=f"📊 데이터베이스에서 '{career_name}' 검색 중...")
    )
    
    try:
        # 정확한 매칭 시도
        exact_match = supabase.table('career_pros_cons').select("*").eq('career_name', career_name).execute()
        if exact_match.data:
            state["search_method"] = "database"
            state["results"] = {"data": exact_match.data}
            state["messages"].append(
                AIMessage(content=f"✅ 데이터베이스에서 '{career_name}' 정보를 찾았습니다! ({len(exact_match.data)}개 항목)")
            )
            return state
        
        state["messages"].append(
            AIMessage(content=f"❌ 데이터베이스에서 '{career_name}'을(를) 찾을 수 없습니다. 웹에서 검색합니다...")
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

def crawl_web(state: CareerState) -> CareerState:
    """웹에서 직업 정보 크롤링"""
    if state["results"].get("data"):  # 이미 DB에서 찾은 경우
        return state
    
    career_name = state["career_name"]
    state["search_method"] = "web_crawling"
    crawler = get_crawler()
    
    if not crawler:
        state["messages"].append(
            AIMessage(content="⚠️ 웹 크롤링이 설정되지 않았습니다.")
        )
        return state
    
    state["messages"].append(
        HumanMessage(content=f"🌐 웹에서 '{career_name}' 정보 수집 시작...")
    )
    
    # API 키가 없을 때만 샘플 데이터 사용
    if not OPENAI_API_KEY:
        # 기본 데이터
        career_data = {
            "pros": [
                "전문성을 개발할 수 있습니다",
                "안정적인 수입이 가능합니다",
                "경력 개발 기회가 있습니다",
                "사회적 기여를 할 수 있습니다",
                "네트워크를 확장할 수 있습니다"
            ],
            "cons": [
                "업무 스트레스가 있을 수 있습니다",
                "워라밸 유지가 어려울 수 있습니다",
                "경쟁이 치열할 수 있습니다",
                "지속적인 자기계발이 필요합니다",
                "초기 연봉이 낮을 수 있습니다"
            ]
        }
        
        state["pros"] = career_data["pros"]
        state["cons"] = career_data["cons"]
        state["salary_info"] = crawler.get_career_salary_info(career_name)
        state["career_path"] = crawler.get_career_path(career_name)
        
        state["messages"].append(
            AIMessage(content="📌 샘플 데이터를 표시합니다 (OpenAI API 키 설정 필요)")
        )
        return state
    
    # 실제 크롤링 로직
    all_pros = []
    all_cons = []
    sources = []
    
    # 직업 정보 검색
    search_results = crawler.search_career_info(career_name)
    
    if search_results:
        state["messages"].append(
            AIMessage(content=f"→ {len(search_results)}개 포스트/기사 발견 (블로그 + 뉴스)")
        )
        
        # 각 포스트 처리 (최대 15개까지 처리)
        processed_count = 0
        for idx, post in enumerate(search_results[:15]):
            search_type = post.get('search_type', 'blog')
            state["messages"].append(
                AIMessage(content=f"📖 [{search_type}] 분석 중: {post['title'][:40]}...")
            )
            
            # 크롤링
            content = crawler.crawl_content(post['link'])
            if not content:
                continue
            
            crawler.stats['total_crawled'] += 1
            processed_count += 1
            
            # 장단점 추출
            if OPENAI_API_KEY:
                # GPT API를 사용한 추출
                pros_cons = crawler.extract_career_pros_cons_with_gpt(career_name, content)
            else:
                # 키워드 기반 간단한 추출
                pros_cons = crawler.extract_career_pros_cons_simple(career_name, content)
            
            if pros_cons:
                all_pros.extend(pros_cons['pros'])
                all_cons.extend(pros_cons['cons'])
                sources.append({
                    'title': post['title'],
                    'link': post['link'],
                    'date': post.get('postdate', ''),
                    'type': search_type
                })
                
                state["messages"].append(
                    AIMessage(content=f"✓ 장점 {len(pros_cons['pros'])}개, 단점 {len(pros_cons['cons'])}개 추출")
                )
            
            # API 호출 제한을 위한 대기
            time.sleep(0.3)
            
            # 충분한 데이터를 수집했으면 중단
            if len(all_pros) >= 20 and len(all_cons) >= 20:
                break
    
    # 중복 제거 및 정리
    unique_pros = crawler.deduplicate_points(all_pros)
    unique_cons = crawler.deduplicate_points(all_cons)
    
    # 크롤링 결과가 없거나 부족한 경우 기본 데이터 사용
    if not unique_pros and not unique_cons:
        state["messages"].append(
            AIMessage(content="⚠️ 웹에서 충분한 정보를 찾지 못했습니다. 기본 정보를 제공합니다.")
        )
        
        # 기본 데이터 (폴백)
        unique_pros = [
            f"{career_name} 분야의 전문성을 개발할 수 있습니다",
            "안정적인 수입과 경력 개발이 가능합니다",
            "다양한 프로젝트 경험을 쌓을 수 있습니다",
            "업계 네트워크를 확장할 기회가 많습니다",
            "지속적인 성장과 발전이 가능합니다"
        ]
        unique_cons = [
            "업무 강도가 높을 수 있습니다",
            "지속적인 학습과 자기계발이 필요합니다",
            "초기에는 연봉이 낮을 수 있습니다",
            "워라밸 유지가 어려울 수 있습니다",
            "경쟁이 치열한 분야입니다"
        ]
    
    state["pros"] = unique_pros[:10]  # 최대 10개
    state["cons"] = unique_cons[:10]  # 최대 10개
    state["sources"] = sources[:10]
    state["salary_info"] = crawler.get_career_salary_info(career_name)
    state["career_path"] = crawler.get_career_path(career_name)
    
    if state["pros"] or state["cons"]:
        state["messages"].append(
            AIMessage(content=f"🎉 웹 크롤링 완료! 총 장점 {len(state['pros'])}개, 단점 {len(state['cons'])}개 수집")
        )
        
        # DB에 저장
        try:
            supabase = get_supabase_client()
            if supabase:
                # 기존 데이터 삭제 (중복 방지)
                try:
                    supabase.table('career_pros_cons').delete().eq('career_name', career_name).execute()
                except:
                    pass
                
                data = []
                
                for pro in state["pros"]:
                    data.append({
                        'career_name': career_name,
                        'type': 'pro',
                        'content': pro
                    })
                
                for con in state["cons"]:
                    data.append({
                        'career_name': career_name,
                        'type': 'con',
                        'content': con
                    })
                
                if data:
                    supabase.table('career_pros_cons').insert(data).execute()
                    state["messages"].append(
                        AIMessage(content="💾 데이터베이스에 저장 완료! 다음 검색 시 더 빠른 결과를 제공합니다.")
                    )
                    st.session_state.saved_careers += 1
        except Exception as e:
            state["messages"].append(
                AIMessage(content=f"⚠️ DB 저장 실패: {str(e)}")
            )
    else:
        state["messages"].append(
            AIMessage(content=f"😢 '{career_name}'에 대한 정보를 찾을 수 없습니다.")
        )
    
    return state

def process_results(state: CareerState) -> CareerState:
    """결과 처리 및 정리"""
    if state["search_method"] == "database" and state["results"].get("data"):
        # DB 결과 처리
        data = state["results"]["data"]
        state["pros"] = [item['content'] for item in data if item['type'] == 'pro']
        state["cons"] = [item['content'] for item in data if item['type'] == 'con']
        state["sources"] = []
        
        # 연봉 정보와 경력 경로는 크롤러를 통해 가져옴
        crawler = get_crawler()
        if crawler:
            state["salary_info"] = crawler.get_career_salary_info(state["career_name"])
            state["career_path"] = crawler.get_career_path(state["career_name"])
        
        state["messages"].append(
            AIMessage(content=f"📋 결과 정리 완료: 장점 {len(state['pros'])}개, 단점 {len(state['cons'])}개")
        )
    
    return state

def should_search_web(state: CareerState) -> str:
    """웹 검색이 필요한지 판단"""
    if state["results"].get("data"):
        return "process"
    else:
        return "crawl"

# ========================
# LangGraph 워크플로우 생성
# ========================

@st.cache_resource
def create_career_workflow():
    workflow = StateGraph(CareerState)
    
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
career_app = create_career_workflow()

# ========================
# Streamlit UI
# ========================

# 검색 섹션
col1, col2, col3 = st.columns([1, 5, 1])

with col2:
    st.markdown('<div class="search-section">', unsafe_allow_html=True)
    
    # 제목
    st.markdown("""
    <h2 class="search-title">
        어떤 직업을 알아보고 계신가요?
    </h2>
    """, unsafe_allow_html=True)
    
    # 북마크에서 선택된 항목이 있으면 자동 입력
    default_value = ""
    if 'selected_bookmark' in st.session_state:
        default_value = st.session_state.selected_bookmark
        del st.session_state.selected_bookmark
    elif 'search_query' in st.session_state:
        default_value = st.session_state.search_query
    
    # 검색창
    st.markdown('<div class="big-search">', unsafe_allow_html=True)
    career_name = st.text_input(
        "직업명 입력",
        placeholder="예: 데이터 분석가, 데이터 엔지니어, DBA 등",
        value=default_value,
        label_visibility="collapsed",
        key="career_search_input"
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
        if career_name and st.button("📌", help="북마크에 추가", key="bookmark_btn"):
            if career_name not in st.session_state.bookmarks:
                st.session_state.bookmarks.append(career_name)
                st.success("북마크에 추가되었습니다!")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 인기 검색어
    st.markdown("""
    <div class="popular-search-buttons" style="text-align: center; margin-top: 2rem;">
        <p style="opacity: 0.7; font-size: 1.2rem; margin-bottom: 1rem; color: #666; font-weight: 500;">인기 직업</p>
    """, unsafe_allow_html=True)
    
    popular_careers = ["데이터 분석가", "데이터 엔지니어", "DBA", "DB 엔지니어", "AI 개발자", "AI 엔지니어", "자바 개발자", "백엔드 개발자"]
    cols = st.columns(4)
    for idx, (col, career) in enumerate(zip(cols * 2, popular_careers)):
        with col:
            if st.button(
                career, 
                key=f"popular_{idx}", 
                use_container_width=True,
                help=f"{career} 검색하기"
            ):
                st.session_state.search_query = career
                st.rerun()
    
    st.markdown('</div></div>', unsafe_allow_html=True)

# 검색 실행
if search_button:
    # 인기 검색어로 선택된 경우 해당 검색어 사용
    if 'search_query' in st.session_state and st.session_state.search_query:
        search_term = st.session_state.search_query
        st.session_state.search_query = ""
    else:
        search_term = career_name
    
    if search_term:
        # 검색 통계 증가
        st.session_state.total_searches += 1
        
        loading_placeholder = show_loading_animation()
        
        # LangGraph 실행
        initial_state = {
            "career_name": search_term,
            "search_method": "",
            "results": {},
            "pros": [],
            "cons": [],
            "sources": [],
            "salary_info": {},
            "career_path": [],
            "messages": [],
            "error": ""
        }
        
        # 워크플로우 실행
        final_state = career_app.invoke(initial_state)
        
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
            
            # 연봉 정보 및 경력 경로
            st.markdown("---")
            st.markdown("### 💼 직업 개요")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # 연봉 차트
                if final_state.get("salary_info"):
                    salary_chart = create_salary_chart(final_state["salary_info"], final_state["career_name"])
                    st.plotly_chart(salary_chart, use_container_width=True)
            
            with col2:
                # 경력 경로
                if final_state.get("career_path"):
                    career_timeline = create_career_path_timeline(final_state["career_path"])
                    st.plotly_chart(career_timeline, use_container_width=True)
            
            # 장단점 차트 및 워드클라우드
            st.markdown("---")
            
            # 워드클라우드 표시
            display_wordclouds(final_state["pros"], final_state["cons"])
            
            # 장단점 통계 차트
            pros_cons_chart = create_pros_cons_chart(len(final_state["pros"]), len(final_state["cons"]))
            st.plotly_chart(pros_cons_chart, use_container_width=True)
            
            # 장단점 상세 표시
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
        else:
            st.error(f"'{search_term}'에 대한 정보를 찾을 수 없습니다.")

# 하단 정보
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="metric-card">
        <i class="fas fa-brain" style="color: #667eea; font-size: 2rem;"></i>
        <p style="margin-top: 0.5rem;">LangGraph로 구현된<br>체계적인 분석 프로세스</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="metric-card">
        <i class="fas fa-sync-alt" style="color: #28a745; font-size: 2rem;"></i>
        <p style="margin-top: 0.5rem;">DB 우선 검색<br>→ 없으면 웹 크롤링</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="metric-card">
        <i class="fas fa-save" style="color: #dc3545; font-size: 2rem;"></i>
        <p style="margin-top: 0.5rem;">검색 결과<br>자동 저장</p>
    </div>
    """, unsafe_allow_html=True)

current_date = datetime.now().strftime('%Y년 %m월 %d일')
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 2rem; margin-top: 2rem;">
    <p style="margin-bottom: 0.5rem;">
        <i class="fas fa-clock"></i> 마지막 업데이트: {current_date}
    </p>
    <p style="font-size: 0.9rem; opacity: 0.8;">
        Powered by LangGraph & OpenAI | Made with <i class="fas fa-heart" style="color: #e74c3c;"></i> by Career Insight Team
    </p>
</div>
""", unsafe_allow_html=True)
