"""
스마트한 쇼핑 앱 - LangGraph 버전 (디자인 개선)
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
    st.metric("총 검색 수", "0회")
    st.metric("저장된 제품", "0개")

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

# 헤더
st.markdown("""
<div class="main-header">
    <h1>🛒 스마트한 쇼핑 (LangGraph Edition)</h1>
    <p style="font-size: 1.2rem; margin-top: 1rem;">
        LangGraph로 구현한 지능형 제품 리뷰 분석 시스템
    </p>
    <p style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.8;">
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
    <h3 style="text-align: center; color: #333; margin-bottom: 1.5rem;">
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
        if product_name and st.button("📌", help="북마크에 추가"):
            if product_name not in st.session_state.bookmarks:
                st.session_state.bookmarks.append(product_name)
                st.success("북마크에 추가되었습니다!")
    
    st.markdown('</div>', unsafe_allow_html=True)

# 검색 실행
if search_button and product_name:
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
