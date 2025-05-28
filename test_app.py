"""
스마트한 쇼핑 앱 - LangGraph 버전 (완전 개선판)
"""

import streamlit as st
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

# LangGraph 관련
from typing import TypedDict, Annotated, List, Union, Dict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
import operator

# 페이지 설정
st.set_page_config(
    page_title="스마트한 쇼핑 (LangGraph)",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# 사이드바 설정
with st.sidebar:
    st.markdown("### ⚙️ 설정")
    dark_mode = st.checkbox("🌙 다크모드", value=st.session_state.dark_mode)
    st.session_state.dark_mode = dark_mode
    
    # 테마 선택
    theme = st.selectbox(
        "🎨 테마 선택",
        ["기본", "네온", "파스텔", "그라디언트", "미니멀"]
    )
    
    st.markdown("### 📌 북마크")
    if st.session_state.bookmarks:
        for bookmark in st.session_state.bookmarks:
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(f"🔖 {bookmark}", key=f"bookmark_{bookmark}"):
                    st.session_state.selected_bookmark = bookmark
            with col2:
                if st.button("❌", key=f"remove_{bookmark}"):
                    st.session_state.bookmarks.remove(bookmark)
                    st.rerun()
    else:
        st.info("북마크가 없습니다")
    
    st.markdown("### 📊 사용 통계")
    st.metric("총 검색 수", f"{st.session_state.total_searches}회")
    st.metric("저장된 제품", f"{st.session_state.saved_products}개")
    
    # 검색 기록
    st.markdown("### 🕐 최근 검색")
    if st.session_state.search_history:
        for item in st.session_state.search_history[-5:]:
            st.text(f"• {item}")
    else:
        st.info("검색 기록이 없습니다")

# CSS 스타일 - 다크모드 및 테마 지원
if st.session_state.dark_mode:
    bg_gradient = "linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)"
    card_bg = "#0f3460"
    text_color = "#ffffff"
    secondary_text = "#e94560"
    header_gradient = "linear-gradient(135deg, #e94560 0%, #0f3460 100%)"
    glass_bg = "rgba(15, 52, 96, 0.6)"
else:
    bg_gradient = "linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)"
    card_bg = "white"
    text_color = "#333333"
    secondary_text = "#667eea"
    header_gradient = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
    glass_bg = "rgba(255, 255, 255, 0.8)"

# 테마별 스타일 조정
if theme == "네온":
    header_gradient = "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
    secondary_text = "#f5576c"
elif theme == "파스텔":
    header_gradient = "linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)"
    secondary_text = "#fcb69f"
elif theme == "그라디언트":
    header_gradient = "linear-gradient(135deg, #FA8BFF 0%, #2BD2FF 50%, #2BFF88 100%)"
    secondary_text = "#2BD2FF"
elif theme == "미니멀":
    header_gradient = "linear-gradient(135deg, #000000 0%, #434343 100%)"
    secondary_text = "#000000"

st.markdown(f"""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
<style>
    /* 전체 배경 및 기본 스타일 */
    .stApp {{
        background: {bg_gradient};
        position: relative;
        overflow-x: hidden;
    }}
    
    /* 배경 애니메이션 파티클 */
    .particles {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        overflow: hidden;
        z-index: -1;
    }}
    
    .particle {{
        position: absolute;
        background: radial-gradient(circle, rgba(255,255,255,0.3) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
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
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }}
    
    .main-header h1 {{
        text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.3), 
                     0 0 30px rgba(255, 255, 255, 0.3),
                     0 0 60px {secondary_text};
        font-size: 3rem;
        animation: glow 2s ease-in-out infinite alternate;
    }}
    
    @keyframes glow {{
        from {{ text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.3), 0 0 30px rgba(255, 255, 255, 0.3); }}
        to {{ text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.3), 0 0 40px rgba(255, 255, 255, 0.5), 0 0 50px {secondary_text}; }}
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
    
    /* 글래스모피즘 카드 스타일 */
    .search-card {{
        background: {glass_bg};
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-bottom: 2rem;
        color: {text_color};
        transition: all 0.3s ease;
    }}
    
    .search-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }}
    
    /* 장점 섹션 개선 */
    .pros-section {{
        background: linear-gradient(135deg, rgba(212, 241, 212, 0.9) 0%, rgba(184, 230, 184, 0.9) 100%);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        border: 1px solid rgba(40, 167, 69, 0.3);
        box-shadow: 0 5px 15px rgba(40, 167, 69, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .pros-section::before {{
        content: "✨";
        position: absolute;
        font-size: 100px;
        opacity: 0.1;
        right: -20px;
        top: -20px;
        animation: float 4s ease-in-out infinite;
    }}
    
    .pros-section:hover {{
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 8px 25px rgba(40, 167, 69, 0.2);
    }}
    
    /* 단점 섹션 개선 */
    .cons-section {{
        background: linear-gradient(135deg, rgba(255, 214, 214, 0.9) 0%, rgba(255, 184, 184, 0.9) 100%);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        border: 1px solid rgba(220, 53, 69, 0.3);
        box-shadow: 0 5px 15px rgba(220, 53, 69, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .cons-section::before {{
        content: "⚠️";
        position: absolute;
        font-size: 100px;
        opacity: 0.1;
        right: -20px;
        top: -20px;
        animation: float 4s ease-in-out infinite;
    }}
    
    .cons-section:hover {{
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 8px 25px rgba(220, 53, 69, 0.2);
    }}
    
    /* 프로세스 정보 개선 */
    .process-info {{
        background: linear-gradient(135deg, rgba(227, 242, 253, 0.9) 0%, rgba(187, 222, 251, 0.9) 100%);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border: 1px solid rgba(33, 150, 243, 0.3);
        box-shadow: 0 3px 10px rgba(33, 150, 243, 0.1);
        animation: slideIn 0.5s ease-out;
    }}
    
    @keyframes slideIn {{
        from {{ transform: translateX(-100%); opacity: 0; }}
        to {{ transform: translateX(0); opacity: 1; }}
    }}
    
    /* 버튼 스타일 개선 */
    .stButton > button {{
        background: {header_gradient};
        color: white;
        border: none;
        padding: 1rem 2.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    .stButton > button::before {{
        content: "";
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        background: rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    }}
    
    .stButton > button:hover::before {{
        width: 300px;
        height: 300px;
    }}
    
    .stButton > button:active {{
        transform: translateY(0) scale(0.98);
    }}
    
    /* 입력 필드 스타일 */
    .stTextInput > div > div > input {{
        border-radius: 50px;
        border: 2px solid transparent;
        padding: 1rem 2rem;
        transition: all 0.3s ease;
        background: {glass_bg};
        backdrop-filter: blur(10px);
        color: {text_color};
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }}
    
    .stTextInput > div > div > input:focus {{
        border-color: {secondary_text};
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2),
                    0 8px 25px rgba(102, 126, 234, 0.2);
        transform: translateY(-2px);
        background: {card_bg};
    }}
    
    .stTextInput > div > div > input::placeholder {{
        color: #999;
        font-style: italic;
    }}
    
    /* 메트릭 카드 */
    .metric-card {{
        background: {glass_bg};
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: all 0.3s ease;
        color: {text_color};
        border: 1px solid rgba(255, 255, 255, 0.18);
        position: relative;
        overflow: hidden;
    }}
    
    .metric-card::before {{
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        transform: rotate(45deg);
        transition: all 0.5s;
    }}
    
    .metric-card:hover {{
        transform: translateY(-5px) scale(1.05);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }}
    
    .metric-card:hover::before {{
        animation: shine 0.5s ease-in-out;
    }}
    
    @keyframes shine {{
        0% {{ transform: translateX(-100%) translateY(-100%) rotate(45deg); }}
        100% {{ transform: translateX(100%) translateY(100%) rotate(45deg); }}
    }}
    
    /* 애니메이션 효과 */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(30px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    .fade-in {{
        animation: fadeIn 0.8s ease-out;
    }}
    
    /* 카드 호버 애니메이션 */
    @keyframes pulse {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.05); }}
        100% {{ transform: scale(1); }}
    }}
    
    /* 플로팅 효과 */
    @keyframes float {{
        0% {{ transform: translateY(0px) rotate(0deg); opacity: 0.7; }}
        50% {{ transform: translateY(-20px) rotate(180deg); opacity: 1; }}
        100% {{ transform: translateY(0px) rotate(360deg); opacity: 0.7; }}
    }}
    
    /* 타이핑 효과 */
    @keyframes typing {{
        from {{ width: 0; }}
        to {{ width: 100%; }}
    }}
    
    @keyframes blink {{
        50% {{ border-color: transparent; }}
    }}
    
    /* 로딩 스피너 개선 */
    .spinner {{
        width: 60px;
        height: 60px;
        margin: 0 auto;
        position: relative;
    }}
    
    .spinner::before,
    .spinner::after {{
        content: "";
        position: absolute;
        width: 100%;
        height: 100%;
        border-radius: 50%;
        border: 5px solid transparent;
        border-top-color: {secondary_text};
        animation: spin 1s linear infinite;
    }}
    
    .spinner::after {{
        border-top-color: transparent;
        border-bottom-color: {secondary_text};
        animation-delay: 0.5s;
    }}
    
    @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    
    /* 프로스/콘스 아이템 */
    .pros-item, .cons-item {{
        background: {glass_bg};
        backdrop-filter: blur(10px);
        padding: 1.2rem;
        margin: 0.8rem 0;
        border-radius: 15px;
        transition: all 0.3s ease;
        animation: fadeIn 0.5s ease-out;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
    }}
    
    .pros-item {{
        border-left: 4px solid #28a745;
        background: linear-gradient(to right, rgba(40, 167, 69, 0.1), transparent);
    }}
    
    .cons-item {{
        border-left: 4px solid #dc3545;
        background: linear-gradient(to right, rgba(220, 53, 69, 0.1), transparent);
    }}
    
    .pros-item:hover, .cons-item:hover {{
        transform: translateX(10px) scale(1.02);
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.15);
    }}
    
    /* 모바일 반응형 */
    @media (max-width: 768px) {{
        .main-header {{
            padding: 2rem 1rem;
            font-size: 0.9rem;
        }}
        .main-header h1 {{
            font-size: 2rem;
        }}
        .search-card {{
            padding: 1.5rem 1rem;
        }}
        .pros-section, .cons-section {{
            padding: 1.5rem 1rem;
        }}
        .metric-card {{
            padding: 1.5rem;
            margin: 0.5rem 0;
        }}
    }}
    
    /* 프로그레스 바 */
    .progress-bar {{
        width: 100%;
        height: 10px;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        overflow: hidden;
        margin: 1rem 0;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
    }}
    
    .progress-fill {{
        height: 100%;
        background: {header_gradient};
        animation: progress 2s ease-out;
        box-shadow: 0 2px 10px {secondary_text};
    }}
    
    @keyframes progress {{
        from {{ width: 0%; }}
        to {{ width: 100%; }}
    }}
    
    /* 플로팅 장식 요소 */
    .floating-decoration {{
        position: fixed;
        pointer-events: none;
        opacity: 0.15;
        animation: float 6s ease-in-out infinite;
        z-index: 1;
        filter: blur(1px);
    }}
    
    .decoration-1 {{
        top: 10%;
        left: 5%;
        font-size: 4rem;
        animation-delay: 0s;
    }}
    
    .decoration-2 {{
        top: 70%;
        right: 10%;
        font-size: 3rem;
        animation-delay: 2s;
    }}
    
    .decoration-3 {{
        top: 40%;
        left: 90%;
        font-size: 2.5rem;
        animation-delay: 4s;
    }}
    
    .decoration-4 {{
        top: 80%;
        left: 50%;
        font-size: 3.5rem;
        animation-delay: 1s;
    }}
    
    .decoration-5 {{
        top: 20%;
        right: 30%;
        font-size: 2rem;
        animation-delay: 3s;
    }}
    
    /* 반짝이는 별 효과 */
    @keyframes sparkle {{
        0%, 100% {{ opacity: 0; transform: scale(0) rotate(0deg); }}
        50% {{ opacity: 1; transform: scale(1) rotate(180deg); }}
    }}
    
    .sparkle {{
        position: absolute;
        animation: sparkle 3s ease-in-out infinite;
        color: {secondary_text};
    }}
    
    /* 툴팁 스타일 */
    .tooltip {{
        position: relative;
        display: inline-block;
    }}
    
    .tooltip .tooltiptext {{
        visibility: hidden;
        width: 200px;
        background-color: {card_bg};
        color: {text_color};
        text-align: center;
        border-radius: 10px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }}
    
    .tooltip:hover .tooltiptext {{
        visibility: visible;
        opacity: 1;
    }}
    
    /* 스크롤바 스타일 */
    ::-webkit-scrollbar {{
        width: 12px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: rgba(0, 0, 0, 0.1);
        border-radius: 10px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {header_gradient};
        border-radius: 10px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {secondary_text};
    }}
    
    /* 파티클 배경 */
    @keyframes particle-animation {{
        0% {{
            transform: translateY(0) translateX(0) scale(1);
            opacity: 1;
        }}
        100% {{
            transform: translateY(-1000px) translateX(100px) scale(0);
            opacity: 0;
        }}
    }}
    
    .particle {{
        animation: particle-animation 15s linear infinite;
    }}
</style>

<script>
// 파티클 효과 생성
document.addEventListener('DOMContentLoaded', function() {{
    const particlesContainer = document.createElement('div');
    particlesContainer.className = 'particles';
    document.body.appendChild(particlesContainer);
    
    function createParticle() {{
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.width = Math.random() * 10 + 5 + 'px';
        particle.style.height = particle.style.width;
        particle.style.animationDuration = Math.random() * 20 + 10 + 's';
        particle.style.animationDelay = Math.random() * 5 + 's';
        particlesContainer.appendChild(particle);
        
        setTimeout(() => {{
            particle.remove();
        }}, 30000);
    }}
    
    // 초기 파티클 생성
    for (let i = 0; i < 20; i++) {{
        setTimeout(createParticle, i * 300);
    }}
    
    // 지속적으로 파티클 생성
    setInterval(createParticle, 2000);
}});
</script>
""", unsafe_allow_html=True)

# 헤더
st.markdown("""
<div class="main-header">
    <span class="sparkle" style="position: absolute; top: 20px; left: 50px;">✨</span>
    <span class="sparkle" style="position: absolute; top: 60px; right: 80px; animation-delay: 1s;">⭐</span>
    <span class="sparkle" style="position: absolute; bottom: 30px; left: 100px; animation-delay: 2s;">💫</span>
    <h1>🛒 스마트한 쇼핑 (LangGraph Edition)</h1>
    <p style="font-size: 1.2rem; margin-top: 1rem;">
        LangGraph로 구현한 지능형 제품 리뷰 분석 시스템
    </p>
    <p style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.8;">
        <i class="fas fa-robot"></i> AI가 수천 개의 리뷰를 분석하여 핵심 장단점을 요약해드립니다
    </p>
</div>

<!-- 플로팅 장식 요소 -->
<div class="floating-decoration decoration-1">🛍️</div>
<div class="floating-decoration decoration-2">💡</div>
<div class="floating-decoration decoration-3">⭐</div>
<div class="floating-decoration decoration-4">🎯</div>
<div class="floating-decoration decoration-5">✨</div>
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
    """개선된 로딩 애니메이션"""
    loading_placeholder = st.empty()
    loading_messages = [
        "🔍 제품 정보를 검색하고 있습니다...",
        "📊 데이터베이스를 확인하고 있습니다...",
        "🌐 웹에서 리뷰를 수집하고 있습니다...",
        "🤖 AI가 리뷰를 분석하고 있습니다...",
        "✨ 결과를 정리하고 있습니다..."
    ]
    
    for i in range(len(loading_messages)):
        loading_placeholder.markdown(f"""
        <div style="text-align: center; padding: 3rem;">
            <div class="spinner"></div>
            <p style="margin-top: 1.5rem; color: {secondary_text}; font-weight: 600; font-size: 1.1rem;">
                <i class="fas fa-brain"></i> {loading_messages[i % len(loading_messages)]}
            </p>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {(i + 1) * 20}%;"></div>
            </div>
            <p style="margin-top: 0.5rem; opacity: 0.6;">
                잠시만 기다려주세요...
            </p>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.5)
    
    return loading_placeholder

def create_pros_cons_chart(pros_count, cons_count):
    """개선된 장단점 차트"""
    fig = go.Figure()
    
    # 도넛 차트
    fig.add_trace(go.Pie(
        labels=['장점', '단점'],
        values=[pros_count, cons_count],
        hole=0.6,
        marker=dict(
            colors=['#28a745', '#dc3545'],
            line=dict(color='white', width=2)
        ),
        textinfo='label+percent',
        textfont=dict(size=16, color='white'),
        hovertemplate='%{label}: %{value}개<br>%{percent}<extra></extra>'
    ))
    
    # 중앙 텍스트
    total = pros_count + cons_count
    fig.add_annotation(
        text=f'<b>총 {total}개</b><br>분석 결과',
        x=0.5, y=0.5,
        font=dict(size=20, color=text_color),
        showarrow=False
    )
    
    fig.update_layout(
        height=350,
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(
            x=0.5, y=-0.1,
            xanchor='center',
            orientation='h',
            font=dict(size=14)
        )
    )
    
    return fig

# ========================
# LangGraph 노드 함수들
# ========================

# 클라이언트 초기화
@st.cache_resource
def get_supabase_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_resource
def get_crawler():
    return ProConsLaptopCrawler(NAVER_CLIENT_ID, NAVER_CLIENT_SECRET)

def search_database(state: SearchState) -> SearchState:
    """데이터베이스에서 제품 검색"""
    product_name = state["product_name"]
    supabase = get_supabase_client()
    
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
    
    state["messages"].append(
        HumanMessage(content=f"🌐 웹에서 '{product_name}' 리뷰 수집 시작...")
    )
    
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
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    st.markdown('<div class="search-card fade-in">', unsafe_allow_html=True)
    
    st.markdown("""
    <h3 style="text-align: center; margin-bottom: 1.5rem;">
        <i class="fas fa-search"></i> 어떤 제품을 찾고 계신가요?
    </h3>
    """, unsafe_allow_html=True)
    
    # 북마크에서 선택된 항목이 있으면 자동 입력
    default_value = ""
    if 'selected_bookmark' in st.session_state:
        default_value = st.session_state.selected_bookmark
        del st.session_state.selected_bookmark
    
    product_name = st.text_input(
        "",
        placeholder="예: 맥북 프로 M3, LG 그램 2024, 갤럭시북4 프로",
        label_visibility="collapsed",
        value=default_value
    )
    
    col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 1])
    with col_btn1:
        search_button = st.button("🔍 검색하기", use_container_width=True, type="primary")
    with col_btn2:
        show_process = st.checkbox("🔧 프로세스 보기", value=True)
    with col_btn3:
        if product_name and st.button("📌", help="북마크에 추가", key="bookmark_btn"):
            if product_name not in st.session_state.bookmarks:
                st.session_state.bookmarks.append(product_name)
                st.success("북마크에 추가되었습니다!")
    
    # 추천 검색어
    st.markdown("""
    <div style="text-align: center; margin-top: 1rem;">
        <span style="opacity: 0.7; font-size: 0.9rem;">인기 검색어:</span>
    """, unsafe_allow_html=True)
    
    popular_searches = ["맥북 프로 M3", "LG 그램 2024", "갤럭시북4 프로", "에이수스 젠북", "델 XPS 15"]
    cols = st.columns(len(popular_searches))
    for idx, (col, search) in enumerate(zip(cols, popular_searches)):
        with col:
            if st.button(search, key=f"popular_{idx}", use_container_width=True):
                st.session_state.selected_bookmark = search
                st.rerun()
    
    st.markdown('</div></div>', unsafe_allow_html=True)

# 검색 실행
if search_button and product_name:
    # 통계 업데이트
    st.session_state.total_searches += 1
    if product_name not in st.session_state.search_history:
        st.session_state.search_history.append(product_name)
    
    loading_placeholder = show_loading_animation()
    
    # LangGraph 실행
    initial_state = {
        "product_name": product_name,
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
        
        # 차트 표시
        st.plotly_chart(
            create_pros_cons_chart(len(final_state["pros"]), len(final_state["cons"])),
            use_container_width=True
        )
        
        # 장단점 표시
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
                    <div class="pros-item" style="animation-delay: {idx * 0.1}s;">
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
                    <div class="cons-item" style="animation-delay: {idx * 0.1}s;">
                        <span style="color: #dc3545; font-weight: bold;">
                            <i class="fas fa-times"></i> {idx}.
                        </span> {con}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("단점 정보가 없습니다.")
        
        # 출처 (웹 크롤링인 경우)
        if final_state["sources"]:
            with st.expander("📚 출처 보기", expanded=False):
                for idx, source in enumerate(final_state["sources"], 1):
                    st.markdown(f"""
                    <div style="padding: 0.5rem; margin: 0.3rem 0; animation: fadeIn 0.5s ease-out {idx * 0.1}s;">
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
                <i class="fas fa-thumbs-up" style="font-size: 2.5rem; color: #28a745;"></i>
                <h2 style="margin: 0.5rem 0;">{}</h2>
                <p style="margin: 0; opacity: 0.7;">총 장점</p>
            </div>
            """.format(len(final_state['pros'])), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <i class="fas fa-thumbs-down" style="font-size: 2.5rem; color: #dc3545;"></i>
                <h2 style="margin: 0.5rem 0;">{}</h2>
                <p style="margin: 0; opacity: 0.7;">총 단점</p>
            </div>
            """.format(len(final_state['cons'])), unsafe_allow_html=True)
        
        with col3:
            icon = "fa-database" if final_state["search_method"] == "database" else "fa-globe"
            st.markdown("""
            <div class="metric-card">
                <i class="fas {}" style="font-size: 2.5rem; color: #2196f3;"></i>
                <h2 style="margin: 0.5rem 0;">{}</h2>
                <p style="margin: 0; opacity: 0.7;">검색 방법</p>
            </div>
            """.format(icon, "DB" if final_state["search_method"] == "database" else "웹"), unsafe_allow_html=True)
        
        with col4:
            total_score = len(final_state['pros']) / (len(final_state['pros']) + len(final_state['cons'])) * 100 if (len(final_state['pros']) + len(final_state['cons'])) > 0 else 0
            st.markdown("""
            <div class="metric-card">
                <i class="fas fa-star" style="font-size: 2.5rem; color: #ffc107;"></i>
                <h2 style="margin: 0.5rem 0;">{:.0f}%</h2>
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
                <span class="tooltip">
                    <a href="https://twitter.com/intent/tweet?text={share_text}" target="_blank" 
                       style="margin: 0 15px; color: #1DA1F2; font-size: 2rem;">
                        <i class="fab fa-twitter"></i>
                    </a>
                    <span class="tooltiptext">트위터에 공유</span>
                </span>
                <span class="tooltip">
                    <a href="https://www.facebook.com/sharer/sharer.php?u=#" target="_blank" 
                       style="margin: 0 15px; color: #4267B2; font-size: 2rem;">
                        <i class="fab fa-facebook"></i>
                    </a>
                    <span class="tooltiptext">페이스북에 공유</span>
                </span>
                <span class="tooltip">
                    <button onclick="navigator.clipboard.writeText('{share_text}')" 
                            style="margin: 0 15px; background: none; border: none; cursor: pointer; font-size: 2rem;">
                        <i class="fas fa-link" style="color: #666;"></i>
                    </button>
                    <span class="tooltiptext">링크 복사</span>
                </span>
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
        <i class="fas fa-brain" style="color: #667eea; font-size: 2rem;"></i>
        <h4 style="margin-top: 1rem;">LangGraph AI</h4>
        <p>체계적인 검색<br>프로세스</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="metric-card">
        <i class="fas fa-sync-alt" style="color: #28a745; font-size: 2rem;"></i>
        <h4 style="margin-top: 1rem;">스마트 검색</h4>
        <p>DB 우선 검색<br>→ 웹 크롤링</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="metric-card">
        <i class="fas fa-save" style="color: #dc3545; font-size: 2rem;"></i>
        <h4 style="margin-top: 1rem;">자동 저장</h4>
        <p>검색 결과<br>영구 보관</p>
    </div>
    """, unsafe_allow_html=True)

# 추가 기능 섹션
st.markdown("---")
st.markdown("""
<div style="text-align: center; margin: 2rem 0;">
    <h3 style="color: {text_color}; margin-bottom: 1rem;">
        <i class="fas fa-star"></i> 더 많은 기능
    </h3>
</div>
""".format(text_color=text_color), unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="metric-card" style="height: 150px;">
        <i class="fas fa-chart-line" style="color: #ff6b6b; font-size: 2rem;"></i>
        <h5 style="margin-top: 0.5rem;">트렌드 분석</h5>
        <p style="font-size: 0.9rem;">시간별 리뷰 추이</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card" style="height: 150px;">
        <i class="fas fa-users" style="color: #4ecdc4; font-size: 2rem;"></i>
        <h5 style="margin-top: 0.5rem;">커뮤니티</h5>
        <p style="font-size: 0.9rem;">사용자 리뷰 공유</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card" style="height: 150px;">
        <i class="fas fa-bell" style="color: #f7b731; font-size: 2rem;"></i>
        <h5 style="margin-top: 0.5rem;">알림 설정</h5>
        <p style="font-size: 0.9rem;">가격 변동 알림</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card" style="height: 150px;">
        <i class="fas fa-compare" style="color: #5f27cd; font-size: 2rem;"></i>
        <h5 style="margin-top: 0.5rem;">제품 비교</h5>
        <p style="font-size: 0.9rem;">여러 제품 비교</p>
    </div>
    """, unsafe_allow_html=True)

# 푸터
current_date = datetime.now().strftime('%Y년 %m월 %d일')
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 3rem; margin-top: 3rem; background: {glass_bg}; border-radius: 20px;">
    <p style="margin-bottom: 0.5rem;">
        <i class="fas fa-clock"></i> 마지막 업데이트: {current_date}
    </p>
    <p style="font-size: 1rem; margin-bottom: 1rem;">
        <strong>스마트한 쇼핑</strong> - AI가 당신의 현명한 선택을 도와드립니다
    </p>
    <p style="font-size: 0.9rem; opacity: 0.8;">
        Powered by <span style="color: {secondary_text};">LangGraph</span> & 
        <span style="color: {secondary_text};">OpenAI</span> | 
        Made with <i class="fas fa-heart" style="color: #e74c3c;"></i> by Smart Shopping Team
    </p>
    <div style="margin-top: 1rem;">
        <a href="#" style="margin: 0 10px; color: {secondary_text};">
            <i class="fas fa-envelope"></i> 문의하기
        </a>
        <a href="#" style="margin: 0 10px; color: {secondary_text};">
            <i class="fas fa-book"></i> 사용 가이드
        </a>
        <a href="#" style="margin: 0 10px; color: {secondary_text};">
            <i class="fas fa-shield-alt"></i> 개인정보처리방침
        </a>
    </div>
</div>

<!-- 고정 플로팅 버튼 -->
<div style="position: fixed; bottom: 30px; right: 30px; z-index: 1000;">
    <button onclick="window.scrollTo({{top: 0, behavior: 'smooth'}})" 
            style="background: {header_gradient}; color: white; border: none; 
                   width: 60px; height: 60px; border-radius: 50%; 
                   box-shadow: 0 5px 20px rgba(0,0,0,0.3); cursor: pointer;
                   transition: all 0.3s ease;">
        <i class="fas fa-arrow-up" style="font-size: 1.5rem;"></i>
    </button>
</div>
""", unsafe_allow_html=True)

# JavaScript 추가 기능
st.markdown("""
<script>
// 스크롤 애니메이션
const observerOptions = {
    root: null,
    rootMargin: '0px',
    threshold: 0.1
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('fade-in');
        }
    });
}, observerOptions);

// 모든 카드 요소 관찰
document.querySelectorAll('.metric-card, .pros-item, .cons-item').forEach(el => {
    observer.observe(el);
});

// 마우스 추적 효과
document.addEventListener('mousemove', (e) => {
    const mouseX = e.clientX / window.innerWidth;
    const mouseY = e.clientY / window.innerHeight;
    
    document.querySelectorAll('.floating-decoration').forEach((el, index) => {
        const speed = (index + 1) * 0.5;
        const x = (mouseX - 0.5) * speed * 20;
        const y = (mouseY - 0.5) * speed * 20;
        
        el.style.transform = `translate(${x}px, ${y}px)`;
    });
});

// 클릭 리플 효과
document.addEventListener('click', (e) => {
    const ripple = document.createElement('div');
    ripple.className = 'ripple';
    ripple.style.left = e.clientX + 'px';
    ripple.style.top = e.clientY + 'px';
    
    document.body.appendChild(ripple);
    
    setTimeout(() => {
        ripple.remove();
    }, 1000);
});
</script>

<style>
.ripple {
    position: fixed;
    width: 20px;
    height: 20px;
    background: radial-gradient(circle, rgba(255,255,255,0.5) 0%, transparent 70%);
    border-radius: 50%;
    transform: translate(-50%, -50%) scale(0);
    animation: ripple-effect 1s ease-out;
    pointer-events: none;
    z-index: 9999;
}

@keyframes ripple-effect {
    to {
        transform: translate(-50%, -50%) scale(10);
        opacity: 0;
    }
}

/* 스크롤바 애니메이션 */
::-webkit-scrollbar {
    width: 12px;
    transition: all 0.3s ease;
}

::-webkit-scrollbar:hover {
    width: 16px;
}

/* 선택 텍스트 스타일 */
::selection {
    background: {secondary_text};
    color: white;
}

::-moz-selection {
    background: {secondary_text};
    color: white;
}
</style>
""", unsafe_allow_html=True)
