# 🛍️ Smart Shopping Analysis - AI 리뷰 분석 플랫폼

<div align="center">
 <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
 <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
 <img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI">
 <img src="https://img.shields.io/badge/LangGraph-00ADD8?style=for-the-badge&logo=go&logoColor=white" alt="LangGraph">
</div>

<br>

<div align="center">
 <h3>🤖 AI가 수천 개의 실제 리뷰를 분석하여 제품의 진짜 장단점을 알려드립니다</h3>
</div>

---

## 📌 프로젝트 소개

**Smart Shopping Analysis**는 인공지능을 활용하여 온라인 쇼핑몰의 실제 사용자 리뷰를 수집하고 분석하는 웹 애플리케이션입니다. 

### ✨ 주요 기능

- 🔍 **실시간 리뷰 수집**: 네이버 블로그에서 실제 사용 후기 자동 수집
- 🧠 **AI 기반 분석**: ChatGPT를 활용한 장단점 자동 추출
- 📊 **시각화 대시보드**: 워드클라우드, 차트 등 직관적인 분석 결과 제공
- 💾 **데이터베이스 연동**: Supabase를 통한 분석 결과 저장 및 관리
- 🛒 **쿠팡 파트너스 연동**: 분석된 제품의 구매 링크 제공

## 🎯 왜 이 서비스가 필요한가요?

> "실제 구매자들의 솔직한 후기를 한눈에 파악하고 싶으신가요?"

- ✅ 수백 개의 리뷰를 일일이 읽을 필요 없이 **핵심 장단점만 빠르게 확인**
- ✅ 광고성 리뷰가 아닌 **실제 사용자의 경험** 기반 분석
- ✅ 카테고리별 장단점 분포를 통한 **객관적인 구매 의사결정** 지원

## 🛠️ 기술 스택

### Backend
- **Python 3.9+**
- **LangGraph**: 체계적인 검색 프로세스 구현
- **OpenAI API**: 리뷰 분석 및 장단점 추출
- **Supabase**: 데이터 저장 및 관리

### Frontend
- **Streamlit**: 대화형 웹 애플리케이션
- **Plotly**: 인터랙티브 차트
- **WordCloud**: 키워드 시각화

### API & Services
- **Naver Search API**: 블로그 리뷰 검색
- **Coupang Partners API**: 제품 구매 링크 생성

## 🚀 주요 화면

### 1. 메인 검색 화면
- 제품명 입력 및 인기 검색어 제공
- 실시간 검색 프로세스 표시

### 2. 분석 결과 화면
- 📋 **상세 분석 결과**: 장단점 리스트
- 🔤 **키워드 분석**: 워드클라우드
- 📊 **심층 분석**: 카테고리별 레이더 차트
- 🛒 **구매 추천**: 쿠팡 최저가 링크

## 📝 사용 방법

1. 검색창에 분석하고 싶은 제품명 입력
2. AI가 자동으로 리뷰 수집 및 분석 시작
3. 분석 완료 후 결과 확인
4. 필요시 쿠팡에서 제품 구매

## 🔧 설치 및 실행

```bash
# 저장소 클론
git clone https://github.com/oracleyu01/project51.git

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
# .env 파일에 필요한 API 키 설정

# 실행
streamlit run test_app.py
