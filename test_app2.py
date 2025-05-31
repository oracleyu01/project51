

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
import urllib.parse

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
    job_postings: List[dict]  # 채용 공고 추가
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
        url = "https://openapi.naver.com/v1/search/blog"
        
        # 직업 관련 검색어 조합
        search_queries = [
            f"{query} 직업 장단점",
            f"{query} 현실 단점",
            f"{query} 실제 장점",
            f"{query} 연봉 워라밸",
            f"{query} 직업 후기"
        ]
        
        all_results = []
        
        for search_query in search_queries:
            params = {
                "query": search_query,
                "display": 10,
                "sort": "sim"
            },
            "자바 개발자": {
                "pros": [
                    "안정적인 기술 스택",
                    "대기업과 금융권 수요 높음",
                    "체계적인 개발 프로세스",
                    "풍부한 레퍼런스와 커뮤니티",
                    "Spring 생태계의 강력함"
                ],
                "cons": [
                    "레거시 시스템 유지보수",
                    "보수적인 기술 환경",
                    "긴 빌드 시간",
                    "무거운 프레임워크",
                    "최신 기술 도입 어려움"
                ]
            },
            "DBA": {
                "pros": [
                    "안정적인 직무로 수요가 꾸준함",
                    "기술의 변화가 상대적으로 느림",
                    "높은 전문성으로 대체 불가능",
                    "체계적인 업무 프로세스",
                    "금융, 대기업 등 안정적인 직장"
                ],
                "cons": [
                    "24시간 온콜 대응 부담",
                    "장애 발생 시 큰 책임감",
                    "반복적인 모니터링 업무",
                    "새로운 기술 도입이 보수적",
                    "야간 작업이 잦음"
                ]
            },
            "DB 엔지니어": {
                "pros": [
                    "데이터 모델링의 창의성",
                    "성능 튜닝의 성취감",
                    "개발과 운영의 균형잡힌 역할",
                    "다양한 프로젝트 경험 가능",
                    "백엔드 개발로 전환 용이"
                ],
                "cons": [
                    "복잡한 쿼리 최적화 압박",
                    "레거시 DB 마이그레이션 스트레스",
                    "다양한 DBMS 학습 부담",
                    "개발팀과 운영팀 사이의 갈등",
                    "성능 이슈에 대한 책임"
                ]
            },
            "AI 엔지니어": {
                "pros": [
                    "최신 기술을 실제 서비스에 적용",
                    "높은 연봉과 대우",
                    "MLOps 분야의 성장 가능성",
                    "다양한 AI 모델 경험",
                    "글로벌 기업 진출 기회"
                ],
                "cons": [
                    "복잡한 인프라 관리",
                    "높은 컴퓨팅 비용 부담",
                    "모델과 시스템 양쪽 지식 필요",
                    "실시간 서빙의 기술적 난이도",
                    "빠른 기술 변화 속도"
                ]
            }
            
            try:
                response = requests.get(url, headers=self.naver_headers, params=params)
                if response.status_code == 200:
                    result = response.json()
                    for item in result.get('items', []):
                        item['title'] = self.remove_html_tags(item['title'])
                        item['description'] = self.remove_html_tags(item['description'])
                    all_results.extend(result.get('items', []))
            except Exception as e:
                print(f"검색 오류: {e}")
        
        return all_results
    
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
            "AI 엔지니어": {"min": 6000, "avg": 9000, "max": 18000            },
            "데이터 엔지니어": {
                "pros": [
                    "높은 수요와 좋은 처우",
                    "대용량 데이터 처리의 성취감",
                    "클라우드 기술 전문성 확보",
                    "데이터 기반 의사결정의 핵심 역할",
                    "다양한 도메인 경험 가능"
                ],
                "cons": [
                    "24/7 파이프라인 모니터링",
                    "복잡한 기술 스택 관리",
                    "데이터 품질 이슈 대응",
                    "가시적 성과 부족",
                    "지속적인 최적화 압박"
                ]
            },
            "자바 개발자": {"min": 4000, "avg": 6000, "max": 10000},
            "백엔드 개발자": {"min": 4500, "avg": 6500, "max": 11000}
        }
        
        # 기본값
        default_salary = {"min": 4000, "avg": 6000, "max": 9000}
        
        for key, value in sample_salaries.items():
            if key in career_name:
                return value
        
        return default_salary
    
    def get_realtime_job_postings(self, career_name):
        """실시간 채용 공고 크롤링"""
        import urllib.parse
        
        # 검색어 생성 (신입, 경력 1-2년)
        search_keywords = [
            f"{career_name} 신입",
            f"{career_name} 주니어",
            f"{career_name} 경력 1년",
            f"{career_name} 경력 2년"
        ]
        
        all_postings = []
        
        try:
            # 네이버 검색으로 채용 정보 수집
            for keyword in search_keywords[:2]:  # API 호출 제한으로 2개만
                encoded_query = urllib.parse.quote(f"{keyword} 채용")
                url = "https://openapi.naver.com/v1/search/webkr.json"
                
                params = {
                    "query": encoded_query,
                    "display": 10,
                    "sort": "date"  # 최신순
                }
                
                response = requests.get(url, headers=self.naver_headers, params=params)
                
                if response.status_code == 200:
                    results = response.json()
                    
                    for item in results.get('items', []):
                        title = self.remove_html_tags(item.get('title', ''))
                        description = self.remove_html_tags(item.get('description', ''))
                        link = item.get('link', '')
                        
                        # 채용 관련 키워드 확인
                        if any(word in title + description for word in ['채용', '모집', '구인', 'hiring', 'recruit']):
                            # 회사명 추출 시도
                            company = ""
                            for corp_indicator in ['(주)', '㈜', '주식회사', 'Corp', 'Inc', 'Ltd']:
                                if corp_indicator in title:
                                    parts = title.split(corp_indicator)
                                    if parts:
                                        company = parts[0].strip() + corp_indicator
                                        break
                            
                            if not company:
                                # 제목에서 첫 단어를 회사명으로 추정
                                company = title.split()[0] if title.split() else "기업"
                            
                            # 연봉 정보 추출 시도
                            salary = "회사 내규"
                            salary_patterns = [
                                r'(\d{3,4})\s*만\s*원',
                                r'(\d{3,4})\s*만원',
                                r'(\d{3,4})\s*~\s*(\d{3,4})',
                                r'연봉\s*(\d{3,4})'
                            ]
                            
                            for pattern in salary_patterns:
                                match = re.search(pattern, description)
                                if match:
                                    if match.lastindex == 2:  # 범위인 경우
                                        salary = f"{match.group(1)}~{match.group(2)}만원"
                                    else:
                                        salary = f"{match.group(1)}만원~"
                                    break
                            
                            # 위치 정보 추출
                            location = "서울"
                            locations = ['강남', '판교', '분당', '여의도', '구로', '가산', '상암', '을지로', '종로', '마포']
                            for loc in locations:
                                if loc in description:
                                    location = loc
                                    break
                            
                            posting = {
                                "company": company[:20],  # 회사명 길이 제한
                                "title": title[:50],
                                "description": description[:200],
                                "link": link,
                                "salary": salary,
                                "location": location,
                                "type": "정규직",
                                "requirements": []
                            }
                            
                            # 요구사항 추출
                            req_keywords = ['경험', '능력', '우대', '필수', '자격', '조건']
                            req_sentences = [sent for sent in description.split('.') 
                                           if any(kw in sent for kw in req_keywords)]
                            posting["requirements"] = req_sentences[:3]
                            
                            all_postings.append(posting)
                
                time.sleep(0.5)  # API 호출 제한
                
        except Exception as e:
            print(f"채용 정보 크롤링 오류: {e}")
        
        # 중복 제거 및 정렬
        seen_companies = set()
        unique_postings = []
        for posting in all_postings:
            if posting["company"] not in seen_companies and len(unique_postings) < 5:
                seen_companies.add(posting["company"])
                unique_postings.append(posting)
        
        return unique_postings

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
    
    # API 키가 없을 때 샘플 데이터
    if not OPENAI_API_KEY:
        # 직업별 샘플 데이터
        sample_data = {
            "데이터 분석가": {
                "pros": [
                    "데이터 기반 의사결정을 통한 비즈니스 가치 창출",
                    "다양한 도구와 기술을 배울 수 있는 기회",
                    "높은 수요와 좋은 처우",
                    "재택근무 등 유연한 근무 환경",
                    "다양한 산업 분야로의 이직 가능성"
                ],
                "cons": [
                    "끊임없는 새로운 기술 학습 필요",
                    "데이터 품질 이슈로 인한 스트레스",
                    "비즈니스와 기술 사이의 커뮤니케이션 어려움",
                    "반복적인 리포트 작성 업무",
                    "성과를 정량화하기 어려운 경우가 많음"
                ]
            },
            "AI 개발자": {
                "pros": [
                    "최첨단 기술을 다루는 흥미로운 업무",
                    "높은 연봉과 좋은 복지 혜택",
                    "글로벌 기업 취업 기회",
                    "연구와 개발을 병행할 수 있음",
                    "사회 변화를 이끄는 혁신적인 일"
                ],
                "cons": [
                    "빠른 기술 변화에 대한 지속적 학습 압박",
                    "모델 성능 개선에 대한 압박감",
                    "컴퓨팅 리소스 제약으로 인한 한계",
                    "설명 가능성과 윤리적 문제 고민",
                    "경쟁이 매우 치열한 분야"
                ]
            },
            "백엔드 개발자": {
                "pros": [
                    "안정적이고 높은 수요가 있는 직군",
                    "명확한 커리어 패스와 성장 가능성",
                    "다양한 기술 스택 경험 가능",
                    "문제 해결의 성취감이 큼",
                    "원격 근무 기회가 많음"
                ],
                "cons": [
                    "24/7 서비스 운영으로 인한 온콜 대응",
                    "레거시 코드 유지보수의 어려움",
                    "프론트엔드에 비해 가시적 성과 부족",
                    "지속적인 기술 부채 관리 필요",
                    "디버깅과 트러블슈팅에 많은 시간 소요"
                ]
            },
            "자바 개발자": {
                "pros": [
                    "안정적인 기술 스택",
                    "대기업과 금융권 수요 높음",
                    "체계적인 개발 프로세스",
                    "풍부한 레퍼런스와 커뮤니티",
                    "Spring 생태계의 강력함"
                ],
                "cons": [
                    "레거시 시스템 유지보수",
                    "보수적인 기술 환경",
                    "긴 빌드 시간",
                    "무거운 프레임워크",
                    "최신 기술 도입 어려움"
                ]
            }
        }
        
        # 기본 데이터
        default_data = {
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
        
        # 직업명에 맞는 데이터 찾기
        career_data = default_data
        for key, data in sample_data.items():
            if key in career_name:
                career_data = data
                break
        
        state["pros"] = career_data["pros"]
        state["cons"] = career_data["cons"]
        state["salary_info"] = crawler.get_career_salary_info(career_name)
        state["career_path"] = crawler.get_career_path(career_name)
        
        # 샘플 채용 공고 (API 키가 없을 때)
        state["job_postings"] = [
            {
                "company": "네이버",
                "title": f"{career_name} 신입 채용 (2025년 상반기)",
                "description": f"{career_name} 포지션에서 함께 성장할 인재를 찾습니다. 신입 및 경력 1-2년 지원 가능.",
                "link": f"https://www.saramin.co.kr/zf_user/search?searchType=search&searchword={urllib.parse.quote(career_name)}",
                "salary": "신입 기준 4,000만원~",
                "location": "판교",
                "type": "정규직",
                "source": "saramin.co.kr"
            },
            {
                "company": "카카오",
                "title": f"주니어 {career_name} 모집",
                "description": "카카오와 함께 세상을 변화시킬 주니어 개발자를 모집합니다.",
                "link": f"https://www.wanted.co.kr/search?query={urllib.parse.quote(career_name)}",
                "salary": "협의",
                "location": "판교",
                "type": "정규직",
                "source": "wanted.co.kr"
            }
        ]
        
        state["messages"].append(
            AIMessage(content="📌 샘플 데이터를 표시합니다 (API 키 설정 필요)")
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
            AIMessage(content=f"→ {len(search_results)}개 포스트 발견")
        )
        
        # 각 포스트 처리
        for idx, post in enumerate(search_results[:10]):
            state["messages"].append(
                AIMessage(content=f"📖 분석 중: {post['title'][:40]}...")
            )
            
            # 크롤링
            content = crawler.crawl_content(post['link'])
            if not content:
                continue
            
            crawler.stats['total_crawled'] += 1
            
            # 장단점 추출
            pros_cons = crawler.extract_career_pros_cons_with_gpt(career_name, content)
            
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
            
            time.sleep(0.5)
    
    # 중복 제거 및 정리
    unique_pros = crawler.deduplicate_points(all_pros)
    unique_cons = crawler.deduplicate_points(all_cons)
    
    state["pros"] = unique_pros
    state["cons"] = unique_cons
    state["sources"] = sources[:10]
    state["salary_info"] = crawler.get_career_salary_info(career_name)
    state["career_path"] = crawler.get_career_path(career_name)
    
    # 실시간 채용 정보 가져오기
    try:
        state["job_postings"] = crawler.get_realtime_job_postings(career_name)
    except:
        state["job_postings"] = []
    
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
                        AIMessage(content="💾 데이터베이스에 저장 완료!")
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
            # 실시간 채용 정보
            try:
                state["job_postings"] = crawler.get_realtime_job_postings(state["career_name"])
            except:
                state["job_postings"] = []
        
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
        placeholder="예: 개발자, 의사, 교사, 디자이너, 변호사",
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
                st.session_state.total_searches += 1
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
            "job_postings": [],
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
            
            # 구체적인 업무 내용 표시
            st.markdown("---")
            st.markdown("### 💻 구체적인 업무 내용")
            
            # 직업별 구체적인 업무 정의
            job_tasks = {
                "데이터 분석가": [
                    "비즈니스 데이터 수집 및 정제 작업",
                    "SQL을 활용한 데이터 추출 및 가공",
                    "Python/R을 이용한 통계 분석 수행",
                    "대시보드 및 시각화 리포트 작성 (Tableau, Power BI)",
                    "A/B 테스트 설계 및 결과 분석",
                    "비즈니스 인사이트 도출 및 의사결정 지원",
                    "데이터 품질 관리 및 검증"
                ],
                "데이터 엔지니어": [
                    "데이터 파이프라인 설계 및 구축",
                    "ETL/ELT 프로세스 개발 및 관리",
                    "대용량 데이터 처리 시스템 구축 (Hadoop, Spark)",
                    "실시간 데이터 스트리밍 처리 (Kafka, Flink)",
                    "데이터 웨어하우스/레이크 아키텍처 설계",
                    "클라우드 기반 데이터 인프라 구축 (AWS, GCP, Azure)",
                    "데이터 품질 모니터링 시스템 개발"
                ],
                "DBA": [
                    "데이터베이스 설치, 구성 및 업그레이드",
                    "데이터베이스 성능 튜닝 및 최적화",
                    "백업 및 복구 전략 수립 및 실행",
                    "데이터베이스 보안 정책 수립 및 관리",
                    "용량 계획 및 스토리지 관리",
                    "데이터베이스 모니터링 및 장애 대응",
                    "SQL 쿼리 최적화 지원"
                ],
                "DB 엔지니어": [
                    "데이터베이스 스키마 설계 및 모델링",
                    "저장 프로시저, 함수, 트리거 개발",
                    "데이터베이스 마이그레이션 계획 및 실행",
                    "인덱스 설계 및 쿼리 성능 최적화",
                    "데이터베이스 연동 API 개발",
                    "NoSQL 데이터베이스 설계 및 구현",
                    "데이터 정합성 및 무결성 관리"
                ],
                "AI 개발자": [
                    "머신러닝/딥러닝 모델 설계 및 개발",
                    "데이터 전처리 및 특징 공학 (Feature Engineering)",
                    "모델 학습 및 하이퍼파라미터 튜닝",
                    "TensorFlow, PyTorch 등 프레임워크 활용",
                    "모델 성능 평가 및 개선",
                    "AI 서비스 API 개발 및 배포",
                    "MLOps 파이프라인 구축"
                ],
                "AI 엔지니어": [
                    "AI 모델 서빙 인프라 구축",
                    "모델 경량화 및 최적화 (양자화, 프루닝)",
                    "엣지 디바이스용 AI 모델 배포",
                    "실시간 추론 시스템 개발",
                    "분산 학습 환경 구축 및 관리",
                    "AI 모델 버전 관리 및 A/B 테스트",
                    "GPU/TPU 클러스터 관리 및 최적화"
                ],
                "자바 개발자": [
                    "Spring Framework 기반 웹 애플리케이션 개발",
                    "RESTful API 설계 및 구현",
                    "JPA/Hibernate를 활용한 데이터 액세스 계층 개발",
                    "단위 테스트 및 통합 테스트 작성 (JUnit, Mockito)",
                    "멀티스레드 프로그래밍 및 동시성 제어",
                    "Maven/Gradle 빌드 도구 활용",
                    "코드 리뷰 및 리팩토링"
                ],
                "백엔드 개발자": [
                    "서버 사이드 비즈니스 로직 구현",
                    "데이터베이스 설계 및 쿼리 작성",
                    "API 설계 및 문서화 (Swagger)",
                    "인증/인가 시스템 구현 (JWT, OAuth)",
                    "캐싱 전략 수립 및 구현 (Redis)",
                    "마이크로서비스 아키텍처 설계",
                    "CI/CD 파이프라인 구축 및 배포 자동화"
                ]
            }
            
            # 현재 직업에 해당하는 업무 찾기
            current_tasks = []
            career_lower = final_state["career_name"].lower()
            
            for job_key, tasks in job_tasks.items():
                if job_key.lower() in career_lower or career_lower in job_key.lower():
                    current_tasks = tasks
                    break
            
            # 기본 업무 (매칭되는 직업이 없을 경우)
            if not current_tasks:
                current_tasks = [
                    "해당 분야의 전문 지식을 활용한 업무 수행",
                    "프로젝트 계획 수립 및 실행",
                    "팀원들과의 협업 및 커뮤니케이션",
                    "업무 관련 문서 작성 및 보고",
                    "지속적인 학습 및 역량 개발",
                    "품질 관리 및 개선 활동",
                    "고객 요구사항 분석 및 대응"
                ]
            
            # 업무 내용을 2열로 표시
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #e6f3ff 0%, #c5e0ff 100%); 
                            padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem;">
                    <h5 style="color: #2196F3; margin-bottom: 1rem;">
                        <i class="fas fa-tasks"></i> 주요 업무
                    </h5>
                </div>
                """, unsafe_allow_html=True)
                
                for idx, task in enumerate(current_tasks[:len(current_tasks)//2], 1):
                    st.markdown(f"""
                    <div style="background: white; padding: 1rem; margin: 0.5rem 0; 
                                border-radius: 8px; border-left: 4px solid #2196F3;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <span style="color: #2196F3; font-weight: bold;">
                            <i class="fas fa-chevron-right"></i> {idx}.
                        </span> {task}
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #e6f3ff 0%, #c5e0ff 100%); 
                            padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem;">
                    <h5 style="color: #2196F3; margin-bottom: 1rem;">
                        <i class="fas fa-clipboard-list"></i> 추가 업무
                    </h5>
                </div>
                """, unsafe_allow_html=True)
                
                for idx, task in enumerate(current_tasks[len(current_tasks)//2:], len(current_tasks)//2 + 1):
                    st.markdown(f"""
                    <div style="background: white; padding: 1rem; margin: 0.5rem 0; 
                                border-radius: 8px; border-left: 4px solid #2196F3;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <span style="color: #2196F3; font-weight: bold;">
                            <i class="fas fa-chevron-right"></i> {idx}.
                        </span> {task}
                    </div>
                    """, unsafe_allow_html=True)
            
            # 실시간 채용 공고 섹션
            st.markdown("---")
            st.markdown("### 💼 실시간 채용 공고")
            
            # 직업별 맞춤 인사이트
            career_insights = {
                "데이터 분석가": {
                    "suitable_for": [
                        "논리적이고 분석적인 사고를 좋아하는 사람",
                        "숫자와 통계에 관심이 많은 사람",
                        "비즈니스 인사이트 도출에 흥미가 있는 사람",
                        "새로운 도구와 기술 학습을 즐기는 사람",
                        "커뮤니케이션 능력이 뛰어난 사람"
                    ],
                    "considerations": [
                        "반복적인 리포트 작업에 대한 인내심 필요",
                        "비즈니스와 기술 양쪽 역량이 모두 필요함",
                        "데이터 품질 이슈로 인한 스트레스 관리 필요",
                        "끊임없는 기술 변화에 대한 적응력 필요",
                        "성과를 정량화하기 어려운 경우가 많음"
                    ]
                },
                "데이터 엔지니어": {
                    "suitable_for": [
                        "대용량 데이터 처리에 관심이 있는 사람",
                        "시스템 설계와 아키텍처를 좋아하는 사람",
                        "자동화와 효율성을 추구하는 사람",
                        "문제 해결 능력이 뛰어난 사람",
                        "백엔드 기술에 흥미가 있는 사람"
                    ],
                    "considerations": [
                        "24/7 데이터 파이프라인 관리 부담",
                        "장애 대응을 위한 온콜 근무 가능성",
                        "복잡한 기술 스택 학습 필요",
                        "가시적 성과가 잘 드러나지 않을 수 있음",
                        "지속적인 시스템 모니터링 스트레스"
                    ]
                },
                "DBA": {
                    "suitable_for": [
                        "안정성과 신뢰성을 중시하는 사람",
                        "세심하고 꼼꼼한 성격의 사람",
                        "위기 상황 대처 능력이 좋은 사람",
                        "체계적인 업무 처리를 선호하는 사람",
                        "인프라 운영에 관심이 있는 사람"
                    ],
                    "considerations": [
                        "24/7 장애 대응 준비 필요",
                        "실수에 대한 심리적 부담감이 큼",
                        "야간 및 주말 작업 가능성",
                        "새로운 기술보다 안정성 우선시",
                        "반복적인 모니터링 업무의 지루함"
                    ]
                },
                "DB 엔지니어": {
                    "suitable_for": [
                        "데이터 모델링에 흥미가 있는 사람",
                        "성능 최적화를 즐기는 사람",
                        "논리적 사고력이 뛰어난 사람",
                        "세부사항에 주의를 기울이는 사람",
                        "백엔드 개발에 관심이 있는 사람"
                    ],
                    "considerations": [
                        "복잡한 쿼리 최적화에 대한 압박",
                        "레거시 시스템 마이그레이션 스트레스",
                        "다양한 DB 기술 학습 부담",
                        "개발팀과의 지속적인 협업 필요",
                        "성능 이슈에 대한 책임감"
                    ]
                },
                "AI 개발자": {
                    "suitable_for": [
                        "수학과 통계에 강한 사람",
                        "연구와 실험을 좋아하는 사람",
                        "최신 기술 트렌드에 민감한 사람",
                        "창의적 문제 해결을 즐기는 사람",
                        "지속적인 학습에 열정이 있는 사람"
                    ],
                    "considerations": [
                        "빠른 기술 변화에 대한 학습 압박",
                        "모델 성능 개선에 대한 지속적 요구",
                        "높은 컴퓨팅 리소스 비용 문제",
                        "설명 가능한 AI에 대한 고민 필요",
                        "경쟁이 매우 치열한 분야"
                    ]
                },
                "AI 엔지니어": {
                    "suitable_for": [
                        "시스템 최적화에 관심이 많은 사람",
                        "MLOps와 인프라를 좋아하는 사람",
                        "실시간 처리 시스템에 흥미가 있는 사람",
                        "대규모 시스템 운영 경험을 원하는 사람",
                        "DevOps 문화를 선호하는 사람"
                    ],
                    "considerations": [
                        "모델 서빙 인프라 관리 복잡성",
                        "GPU 클러스터 운영 비용 부담",
                        "모델 버전 관리의 어려움",
                        "실시간 추론 성능 보장 압박",
                        "다양한 프레임워크 호환성 문제"
                    ]
                },
                "자바 개발자": {
                    "suitable_for": [
                        "안정적인 기술 스택을 선호하는 사람",
                        "대규모 엔터프라이즈 환경을 원하는 사람",
                        "체계적인 개발 프로세스를 좋아하는 사람",
                        "Spring 생태계에 관심이 있는 사람",
                        "금융/공공 도메인에 관심이 있는 사람"
                    ],
                    "considerations": [
                        "레거시 시스템 유지보수 부담",
                        "보수적인 기술 스택의 한계",
                        "긴 빌드 시간과 무거운 프레임워크",
                        "엔터프라이즈 환경의 경직성",
                        "최신 기술 도입의 어려움"
                    ]
                },
                "백엔드 개발자": {
                    "suitable_for": [
                        "서버 사이드 로직에 흥미가 있는 사람",
                        "시스템 설계와 아키텍처를 좋아하는 사람",
                        "API 설계에 관심이 많은 사람",
                        "성능 최적화를 즐기는 사람",
                        "다양한 기술 스택을 경험하고 싶은 사람"
                    ],
                    "considerations": [
                        "24/7 서비스 운영에 대한 부담",
                        "프론트엔드 대비 가시성 부족",
                        "복잡한 비즈니스 로직 관리",
                        "레거시 코드 리팩토링 압박",
                        "지속적인 기술 부채 해결 필요"
                    ]
                }
            }
            
            # 현재 직업에 맞는 인사이트 찾기
            current_insights = {"suitable_for": [], "considerations": []}
            career_lower = final_state["career_name"].lower()
            
            for job_key, insights in career_insights.items():
                if job_key.lower() in career_lower or career_lower in job_key.lower():
                    current_insights = insights
                    break
            
            # 기본 인사이트 (매칭되는 직업이 없을 경우)
            if not current_insights["suitable_for"]:
                current_insights = {
                    "suitable_for": [
                        "해당 분야에 열정이 있는 사람",
                        "지속적인 학습을 즐기는 사람",
                        "문제 해결 능력이 뛰어난 사람",
                        "팀워크를 중시하는 사람",
                        "책임감이 강한 사람"
                    ],
                    "considerations": [
                        "업무에 대한 전문성 개발 필요",
                        "지속적인 자기계발 요구",
                        "업무 스트레스 관리 능력 필요",
                        "워라밸 유지를 위한 노력 필요",
                        "경력 개발 계획 수립 필요"
                    ]
                }
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="career-insight">
                    <h5 style="color: #667eea; margin-bottom: 1rem;">
                        <i class="fas fa-star"></i> 이 직업이 맞는 사람
                    </h5>
                    <ul style="margin: 0; padding-left: 1.5rem;">
                """, unsafe_allow_html=True)
                
                for suitable in current_insights["suitable_for"]:
                    st.markdown(f"<li>{suitable}</li>", unsafe_allow_html=True)
                
                st.markdown("</ul></div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="career-insight">
                    <h5 style="color: #667eea; margin-bottom: 1rem;">
                        <i class="fas fa-exclamation-triangle"></i> 고려할 점
                    </h5>
                    <ul style="margin: 0; padding-left: 1.5rem;">
                """, unsafe_allow_html=True)
                
                for consideration in current_insights["considerations"]:
                    st.markdown(f"<li>{consideration}</li>", unsafe_allow_html=True)
                
                st.markdown("</ul></div>", unsafe_allow_html=True)
            
            # 관련 직업 추천
            st.markdown("---")
            st.markdown(f"""
            <div style="text-align: center; margin: 2rem 0;">
                <h4 style="color: {text_color}; margin-bottom: 1rem;">
                    <i class="fas fa-link"></i> 관련 직업 추천
                </h4>
                <p style="color: {text_color}; font-size: 0.9rem; opacity: 0.7;">
                    비슷한 스킬셋을 가진 다른 직업들도 살펴보세요
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # 직업별 관련 직업
            related_careers = {
                "데이터 분석가": ["비즈니스 애널리스트", "데이터 사이언티스트", "프로덕트 애널리스트", "마케팅 애널리스트"],
                "데이터 엔지니어": ["빅데이터 엔지니어", "데이터 아키텍트", "클라우드 엔지니어", "MLOps 엔지니어"],
                "DBA": ["데이터 아키텍트", "DB 컨설턴트", "시스템 관리자", "클라우드 DBA"],
                "DB 엔지니어": ["백엔드 개발자", "데이터 엔지니어", "솔루션 아키텍트", "DBA"],
                "AI 개발자": ["머신러닝 엔지니어", "딥러닝 리서처", "컴퓨터 비전 엔지니어", "NLP 엔지니어"],
                "AI 엔지니어": ["MLOps 엔지니어", "AI 플랫폼 엔지니어", "모델 최적화 엔지니어", "엣지 AI 엔지니어"],
                "자바 개발자": ["스프링 개발자", "안드로이드 개발자", "풀스택 개발자", "엔터프라이즈 개발자"],
                "백엔드 개발자": ["DevOps 엔지니어", "풀스택 개발자", "API 개발자", "시스템 아키텍트"]
            }
            
            # 현재 직업과 관련된 직업 찾기
            current_related = []
            for key, values in related_careers.items():
                if key in final_state["career_name"]:
                    current_related = values
                    break
            
            if not current_related:
                current_related = ["컨설턴트", "프리랜서", "창업가", "연구원"]
            
            cols = st.columns(4)
            for idx, (col, related) in enumerate(zip(cols, current_related)):
                with col:
                    st.markdown(f"""
                    <div style="background: {card_bg}; padding: 1rem; border-radius: 10px; 
                                text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                                transition: all 0.3s ease; cursor: pointer;">
                        <i class="fas fa-briefcase" style="color: #667eea; font-size: 1.5rem;"></i>
                        <p style="margin: 0.5rem 0; font-weight: 600; color: {text_color};">{related}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
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
