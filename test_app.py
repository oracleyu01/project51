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

# LangGraph 관련
from typing import TypedDict, Annotated, List, Union, Dict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
import operator

# 페이지 설정
st.set_page_config(
    page_title="스마트한 쇼핑 (LangGraph)",
    page_icon="🛒",
    layout="wide"
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

# CSS 스타일 (헤더 애니메이션 및 색상 변경)
st.markdown("""
<style>
    /* 헤더 그라데이션 배경 애니메이션 - 하늘색 계열로 변경 */
    @keyframes gradientAnimation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* 제목 등장 애니메이션 */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translate3d(0, -20px, 0);
        }
        to {
            opacity: 1;
            transform: translate3d(0, 0, 0);
        }
    }

    /* 부제목/일반 텍스트 페이드인 애니메이션 */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    /* 하단에서 위로 페이드인 애니메이션 (결과 항목 등에 사용) */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translate3d(0, 20px, 0);
        }
        to {
            opacity: 1;
            transform: translate3d(0, 0, 0);
        }
    }

    /* 아이콘 회전 애니메이션 */
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    /* Main Header Styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        /* 기존 보라색 계열 대신 예쁜 하늘색 그라데이션으로 변경 */
        background: linear-gradient(90deg, #89cff0 0%, #4682b4 100%); /* 밝은 하늘색 -> 스틸블루 */
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        background-size: 200% 200%; /* 배경 크기를 키워 애니메이션 영역 확보 */
        animation: gradientAnimation 10s ease infinite; /* 10초 동안 부드럽게 반복 */
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2); /* 그림자 추가로 입체감 */
        overflow: hidden; /* 내부 요소 넘침 방지 */
    }
    .main-header h1 {
        font-size: 3.5rem;
        font-weight: bold;
        text-shadow: 3px 3px 5px rgba(0,0,0,0.3); /* 제목 그림자 */
        animation: fadeInDown 1s ease-out; /* 제목 등장 애니메이션 */
        margin-bottom: 0.5rem;
    }
    .main-header p {
        font-size: 1.5rem;
        margin-top: 0;
        opacity: 0; /* 초기 투명하게 설정 */
        animation: fadeIn 1.5s ease-out 0.5s forwards; /* 부제목 페이드인 (0.5초 지연) */
    }

    /* Input and Button Styling */
    .stTextInput > div > div > input {
        border: 2px solid #89cff0; /* 하늘색 계열로 변경 */
        border-radius: 8px;
        padding: 0.75rem 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    .stTextInput > div > div > input:focus {
        border-color: #4682b4 !important; /* 포커스 시 더 진한 하늘색 */
        box-shadow: 0 0 0 0.25rem rgba(70, 130, 180, 0.25); /* 포커스 시 그림자도 변경 */
        outline: none;
    }

    .stButton > button {
        background-color: #4682b4; /* 버튼 색상도 스틸블루 계열로 변경 */
        color: white;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
        border: none; /* Streamlit 기본 테두리 제거 */
    }
    .stButton > button:hover {
        background-color: #89cff0; /* 호버 시 밝은 하늘색 */
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.25);
    }
    /* 검색 버튼 클릭 시 아이콘 회전 애니메이션 클래스 */
    .stButton > button.search-button-clicked .st-emotion-cache-zt5ig8 { /* Streamlit 아이콘 SVG 선택자 */
        animation: rotate 0.5s ease-out;
    }

    /* Pros Section Styling */
    .pros-section {
        background-color: #e6ffe6; /* 기존 유지 */
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 6px solid #28a745;
        box-shadow: 0 4px 12px rgba(40, 167, 69, 0.1);
        transition: all 0.3s ease-in-out;
    }
    .pros-section:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(40, 167, 69, 0.25);
    }
    .pros-section h3 {
        color: #28a745;
        font-size: 1.8rem;
        margin-bottom: 1rem;
    }
    .pros-section p { /* 각 항목에 적용 */
        margin-bottom: 0.5rem;
        animation: fadeInUp 0.5s ease-out forwards; /* 개별 등장 애니메이션 */
        opacity: 0; /* 초기 투명 */
    }
    /* 각 항목에 delay를 줘서 순차적으로 나타나게 함 */
    .pros-section p:nth-child(2) { animation-delay: 0.1s; }
    .pros-section p:nth-child(3) { animation-delay: 0.2s; }
    .pros-section p:nth-child(4) { animation-delay: 0.3s; }
    .pros-section p:nth-child(5) { animation-delay: 0.4s; }
    .pros-section p:nth-child(6) { animation-delay: 0.5s; }
    /* ... 필요한 만큼 추가 (최대 10개) */


    /* Cons Section Styling */
    .cons-section {
        background-color: #ffe6e6; /* 기존 유지 */
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 6px solid #dc3545;
        box-shadow: 0 4px 12px rgba(220, 53, 69, 0.1);
        transition: all 0.3s ease-in-out;
    }
    .cons-section:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(220, 53, 69, 0.25);
    }
    .cons-section h3 {
        color: #dc3545;
        font-size: 1.8rem;
        margin-bottom: 1rem;
    }
    .cons-section p { /* 각 항목에 적용 */
        margin-bottom: 0.5rem;
        animation: fadeInUp 0.5s ease-out forwards; /* 개별 등장 애니메이션 */
        opacity: 0; /* 초기 투명 */
    }
    /* 각 항목에 delay를 줘서 순차적으로 나타나게 함 */
    .cons-section p:nth-child(2) { animation-delay: 0.1s; }
    .cons-section p:nth-child(3) { animation-delay: 0.2s; }
    .cons-section p:nth-child(4) { animation-delay: 0.3s; }
    .cons-section p:nth-child(5) { animation-delay: 0.4s; }
    .cons-section p:nth-child(6) { animation-delay: 0.5s; }
    /* ... 필요한 만큼 추가 */

    /* Process Info Styling */
    .process-info {
        background-color: #e0f2fe; /* 기존 유지 */
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1.5rem 0;
        border-left: 4px solid #2196f3;
        box-shadow: 0 2px 8px rgba(33, 150, 243, 0.1);
        animation: fadeInUp 0.8s ease-out; /* 등장 애니메이션 */
    }

    /* Footer Styling */
    .footer {
        text-align: center;
        color: #888;
        padding: 2rem;
        font-size: 0.9rem;
    }
    .stExpander { /* Expander에 그림자 추가 */
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-radius: 8px;
        margin-top: 1rem;
    }

</style>
""", unsafe_allow_html=True)

# 헤더
st.markdown("""
<div class="main-header">
    <h1>🛒 스마트한 쇼핑 <br> (LangGraph Edition)</h1>
    <p>
        LangGraph로 구현한 지능형 제품 리뷰 분석 시스템
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
        # 정확한 매칭만 시도 (벡터 검색 제거)
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
    
    # 검색 쿼리 - 제품명에 맞춰 동적으로 생성
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
        for idx, post in enumerate(posts[:5]):  # 각 쿼리당 최대 5개
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
            
            time.sleep(1)  # API 제한 방지
        
        time.sleep(2)  # 검색 간 대기
    
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
        # st.error(f"'{product_name}'에 대한 정보를 찾을 수 없습니다.") # 이 부분을 UI에 직접 표시하도록 변경
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
        state["sources"] = []  # DB에는 별도 소스 없음
        
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
    product_name = st.text_input(
        "🔍 제품명을 입력하세요",
        placeholder="예: 맥북 프로 M3, LG 그램 2024, 갤럭시북4 프로, 그릴 요거트",
        key="product_input" # 고유 키 추가
    )
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        # 버튼에 클릭 시 클래스 추가를 위한 트릭 (자바스크립트 주입)
        search_button_clicked = st.button("🔍 검색하기", use_container_width=True, type="primary", key="search_button")
        if search_button_clicked:
            st.markdown(
                """
                <script>
                    const button = window.parent.document.querySelector('[data-testid="stButton"] button');
                    if (button) {
                        button.classList.add('search-button-clicked');
                        setTimeout(() => {
                            button.classList.remove('search-button-clicked');
                        }, 500); // 0.5초 후 클래스 제거
                    }
                </script>
                """,
                unsafe_allow_html=True
            )
            # 버튼 클릭 시 입력 필드 포커스 해제 (키보드 닫힘)
            # st.session_state["product_input"] = product_name # 입력값을 세션 상태에 저장 (이것은 이제 불필요)
            # st.experimental_rerun() # 재실행하여 CSS 변경 적용 (이것은 버튼 클릭 후 바로 결과가 나와야 하므로 주석 처리 또는 제거)
    with col_btn2:
        show_process = st.checkbox("🔧 프로세스 보기", value=True)

# 검색 실행
if search_button_clicked and product_name:
    with st.spinner(f"'{product_name}' 검색 중..."):
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
    
    # 프로세스 로그 표시
    if show_process and final_state["messages"]:
        with st.expander("🔧 검색 프로세스", expanded=True):
            for msg in final_state["messages"]:
                if isinstance(msg, HumanMessage):
                    st.info(f"👤 {msg.content}") # info로 변경하여 시각적 구분
                else:
                    st.success(f"🤖 {msg.content}") # success로 변경
    
    # 결과 표시
    if final_state["pros"] or final_state["cons"]:
        # 검색 정보
        st.markdown(f"""
        <div class="process-info">
            <strong>검색 방법:</strong> {
                '데이터베이스' if final_state["search_method"] == "database" else '웹 크롤링'
            } | 
            <strong>장점:</strong> {len(final_state["pros"])}개 | 
            <strong>단점:</strong> {len(final_state["cons"])}개
        </div>
        """, unsafe_allow_html=True)
        
        # 장단점 표시
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="pros-section">
                <h3>✅ 장점</h3>
            </div>
            """, unsafe_allow_html=True)
            
            if final_state["pros"]:
                # 각 장점 항목에 고유한 지연 시간 부여
                for idx, pro in enumerate(final_state["pros"], 1):
                    # st.markdown을 사용하여 각 항목에 CSS 애니메이션 클래스 적용
                    st.markdown(f'<p style="animation-delay: {idx*0.1}s;">🟢 {idx}. {pro}</p>', unsafe_allow_html=True)
            else:
                st.write("장점 정보가 없습니다.")
        
        with col2:
            st.markdown("""
            <div class="cons-section">
                <h3>❌ 단점</h3>
            </div>
            """, unsafe_allow_html=True)
            
            if final_state["cons"]:
                # 각 단점 항목에 고유한 지연 시간 부여
                for idx, con in enumerate(final_state["cons"], 1):
                    st.markdown(f'<p style="animation-delay: {idx*0.1}s;">🔴 {idx}. {con}</p>', unsafe_allow_html=True)
            else:
                st.write("단점 정보가 없습니다.")
        
        # 출처 (웹 크롤링인 경우)
        if final_state["sources"]:
            with st.expander("📚 출처 보기"):
                for idx, source in enumerate(final_state["sources"], 1):
                    st.write(f"{idx}. [{source['title']}]({source['link']})")
        
        # 통계
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 장점", f"{len(final_state['pros'])}개")
        with col2:
            st.metric("총 단점", f"{len(final_state['cons'])}개")
        with col3:
            st.metric("검색 방법", "DB" if final_state["search_method"] == "database" else "웹")
    else:
        st.error(f"'{product_name}'에 대한 정보를 찾을 수 없습니다. 다시 시도해 주세요.")

# 하단 정보
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.info("💡 LangGraph로 구현된 체계적인 검색 프로세스")
with col2:
    st.info("🔄 DB 우선 검색 → 없으면 웹 크롤링")
with col3:
    st.info("💾 검색 결과 자동 저장")

current_date = datetime.now().strftime('%Y년 %m월 %d일')
st.markdown(f"""
<div class="footer">
    <p>마지막 업데이트: {current_date}</p>
    <p>Powered by LangGraph & OpenAI</p>
</div>
""", unsafe_allow_html=True)
