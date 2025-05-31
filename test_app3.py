import streamlit as st
import openai
import pandas as pd
import numpy as np
from datetime import datetime
import json
import time
import requests
from urllib.parse import quote
import plotly.graph_objects as go
import plotly.express as px
import os

# 페이지 설정 - 사이드바 숨김
st.set_page_config(
    page_title="빅데이터분석기사 실기 Q&A",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"  # 사이드바 숨김
)

# 세션 상태 초기화
def initialize_session_state():
    # API 키 자동 설정 - GitHub Secrets/Streamlit Secrets에서 가져오기
    if 'openai_api_key' not in st.session_state:
        # Streamlit Secrets에서 먼저 확인
        if "OPENAI_API_KEY" in st.secrets:
            st.session_state.openai_api_key = st.secrets["OPENAI_API_KEY"]
        # 환경 변수에서 확인
        elif "OPENAI_API_KEY" in os.environ:
            st.session_state.openai_api_key = os.environ["OPENAI_API_KEY"]
        else:
            st.session_state.openai_api_key = None
            
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_problem' not in st.session_state:
        st.session_state.current_problem = None
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
    if 'total_questions' not in st.session_state:
        st.session_state.total_questions = 0
    if 'study_progress' not in st.session_state:
        st.session_state.study_progress = {}
    if 'selected_quick_question' not in st.session_state:
        st.session_state.selected_quick_question = None

initialize_session_state()

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
    /* 사이드바 숨기기 */
    section[data-testid="stSidebar"] {{
        display: none;
    }}
    
    /* 전체 배경 및 기본 스타일 */
    .stApp {{
        background: {bg_gradient};
        color: {text_color};
    }}
    
    /* 메인 헤더 개선 */
    .main-header {{
        text-align: center;
        padding: 2rem 1rem;
        background: {header_gradient};
        color: white;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
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
        background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%);
        animation: shimmer 4s infinite;
    }}
    
    @keyframes shimmer {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    
    .main-header h1 {{
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.4);
        letter-spacing: -1px;
    }}
    
    .main-header p {{
        font-size: 1.1rem;
        margin-bottom: 0.3rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        opacity: 0.95;
    }}
    
    /* 카드 스타일 */
    .content-card {{
        background: {card_bg};
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }}
    
    .content-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
    }}
    
    /* 문제 카드 스타일 */
    .problem-card {{
        background: linear-gradient(135deg, {card_bg} 0%, rgba(102, 126, 234, 0.05) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid {secondary_text};
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.1);
        transition: all 0.3s ease;
    }}
    
    .problem-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
    }}
    
    /* 채팅 카드 스타일 */
    .chat-card {{
        background: {card_bg};
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.08);
        margin-bottom: 1rem;
        border: 1px solid rgba(102, 126, 234, 0.1);
    }}
    
    /* AI 답변 */
    .ai-response {{
        background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #4caf50;
        box-shadow: 0 3px 15px rgba(76, 175, 80, 0.1);
    }}
    
    /* 버튼 스타일 개선 */
    .stButton > button {{
        background: {header_gradient};
        color: white;
        border: none;
        padding: 0.7rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        letter-spacing: 0.5px;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        filter: brightness(1.1);
    }}
    
    /* 입력 필드 스타일 */
    .stTextArea > div > div > textarea {{
        border-radius: 15px;
        border: 2px solid rgba(102, 126, 234, 0.2);
        padding: 1rem;
        transition: all 0.3s ease;
        background: {card_bg};
        color: {text_color};
        font-size: 1rem;
    }}
    
    .stTextArea > div > div > textarea:focus {{
        border-color: {secondary_text};
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15);
        outline: none;
    }}
    
    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2rem;
        background: transparent;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: rgba(102, 126, 234, 0.1);
        border-radius: 10px;
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        color: {text_color};
        transition: all 0.3s ease;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: {header_gradient};
        color: white;
        box-shadow: 0 3px 10px rgba(102, 126, 234, 0.3);
    }}
    
    /* 메트릭 카드 */
    .metric-card {{
        background: {card_bg};
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.08);
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid rgba(102, 126, 234, 0.1);
    }}
    
    .metric-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
    }}
    
    /* 애니메이션 효과 */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(30px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    .fade-in {{
        animation: fadeIn 0.8s ease-out;
    }}
    
    /* 프로그레스 바 */
    .progress-container {{
        background: rgba(102, 126, 234, 0.1);
        border-radius: 10px;
        padding: 0.5rem;
        margin: 1rem 0;
    }}
    
    .progress-bar {{
        background: {header_gradient};
        height: 8px;
        border-radius: 5px;
        transition: width 1s ease;
    }}
    
    /* 코드 블록 개선 */
    .stCodeBlock {{
        border-radius: 12px;
        box-shadow: 0 3px 15px rgba(0, 0, 0, 0.1);
    }}
    
    /* 인프런 광고 섹션 */
    .inflearn-ad {{
        background: linear-gradient(135deg, #ff6b6b 0%, #ff8e53 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        color: white;
        text-align: center;
        box-shadow: 0 15px 35px rgba(255, 107, 107, 0.3);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .inflearn-button {{
        background: white;
        color: #ff6b6b;
        padding: 0.8rem 2rem;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.1rem;
        text-decoration: none;
        display: inline-block;
        margin-top: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }}
    
    .inflearn-button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        background: #fff;
        color: #ff6b6b;
        text-decoration: none;
    }}
    
    /* 통계 정보 */
    .stats-info {{
        background: rgba(102, 126, 234, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
    }}
    
    /* 모바일 반응형 */
    @media (max-width: 768px) {{
        .main-header {{
            padding: 1.5rem 1rem;
        }}
        .main-header h1 {{
            font-size: 1.8rem;
        }}
        .content-card {{
            padding: 1.5rem;
        }}
        .problem-card {{
            padding: 1rem;
        }}
    }}
    
    /* 선택박스 스타일 */
    .stSelectbox > div > div {{
        background: {card_bg};
        border-radius: 10px;
        border: 2px solid rgba(102, 126, 234, 0.2);
    }}
    
    /* 체크박스 스타일 */
    .stCheckbox > label {{
        color: {text_color};
        font-weight: 500;
    }}
    
    /* 성공/에러 메시지 개선 */
    .stSuccess {{
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: none;
        border-radius: 10px;
        padding: 1rem;
    }}
    
    .stError {{
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: none;
        border-radius: 10px;
        padding: 1rem;
    }}
    
    .stWarning {{
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: none;
        border-radius: 10px;
        padding: 1rem;
    }}
</style>
""", unsafe_allow_html=True)

# 기출문제 데이터
EXAM_DATA = {
    "8회": {
        "작업형1": {
            "문제1": {
                "제목": "커피 소비량 분석",
                "난이도": "⭐⭐",
                "주요개념": ["groupby", "idxmax", "nlargest"],
                "내용": """
**문제 1-1**: 지역('region')별 커피 소비량('coffee_servings')의 평균을 계산하고, 평균이 가장 높은 지역을 찾으세요.

**문제 1-2**: 1번에서 찾은 지역에서 커피 소비량이 3번째로 많은 도시('city')의 커피 소비량을 구하세요.

**데이터 구조**:
- city: 도시명
- region: 지역명  
- coffee_servings: 커피 소비량
                """,
                "해설": """
이 문제는 pandas의 그룹화와 정렬 기능을 활용하는 문제입니다.

**핵심 포인트**:
1. `groupby()`를 사용한 그룹별 집계
2. `idxmax()`를 사용한 최대값 인덱스 찾기
3. `nlargest()`를 사용한 상위 N개 값 추출
                """,
                "코드": """
import pandas as pd

# 데이터 불러오기
df = pd.read_csv("coffee_data.csv")

# 1-1. 지역별 커피 소비량 평균 계산
region_avg = df.groupby('region')['coffee_servings'].mean()
highest_region = region_avg.idxmax()
print(f"평균이 가장 높은 지역: {highest_region}")

# 1-2. 해당 지역에서 커피 소비량이 3번째로 많은 도시
region_data = df.loc[df['region'] == highest_region, :]
third_highest = region_data.nlargest(3, 'coffee_servings')
result = third_highest.iloc[2]['coffee_servings']
print(f"3번째로 많은 소비량: {result}")
                """
            },
            "문제2": {
                "제목": "생산량 비율 분석",
                "난이도": "⭐⭐⭐",
                "주요개념": ["비율계산", "sort_values", "iloc"],
                "내용": """
'전자 생산 비율'이 세 번째로 높은 국가의 '전자' 생산량을 x라고 정의하세요.
'농업' 생산량이 세 번째로 높은 국가의 '농업' 생산량을 y라고 정의하세요.
x와 y의 합을 구하세요.

**계산 공식**:
- 총 생산량 = 전자 + 농업 + 선박 + 기타
- 전자 생산 비율 = 전자 / 총 생산량
                """,
                "해설": """
이 문제는 파생 변수 생성과 정렬을 조합한 문제입니다.

**단계별 접근**:
1. 총 생산량과 비율 계산
2. 비율 기준으로 정렬하여 3번째 값 추출
3. 절대값 기준으로 정렬하여 3번째 값 추출
                """,
                "코드": """
import pandas as pd

# 데이터 불러오기
df = pd.read_csv("short_prod_data.csv")

# 총 생산량 및 전자 생산 비율 계산
df['total'] = df['Elec'] + df['Agr'] + df['Ship'] + df['Oth']
df['Elec_rate'] = df['Elec'] / df['total']

# 전자 생산 비율이 세 번째로 높은 국가의 전자 생산량
df_sorted_by_rate = df.sort_values(by='Elec_rate', ascending=False)
x = df_sorted_by_rate.iloc[2]['Elec']

# 농업 생산량이 세 번째로 높은 국가의 농업 생산량
df_sorted_by_agr = df.sort_values(by='Agr', ascending=False)
y = df_sorted_by_agr.iloc[2]['Agr']

result = x + y
print(f"x + y = {result}")
                """
            },
            "문제3": {
                "제목": "기후 데이터 Min-Max 스케일링",
                "난이도": "⭐⭐⭐",
                "주요개념": ["MinMaxScaler", "표준편차", "전처리"],
                "내용": """
1. '온도'와 '습도' 열을 각각 Min-Max 스케일링하세요.
2. 스케일링된 '온도'와 '습도' 열의 표준편차를 각각 구하세요.
3. '온도' 열의 표준편차에서 '습도' 열의 표준편차를 뺀 값을 소수점 2자리로 반올림하여 구하세요.
                """,
                "해설": """
데이터 전처리와 통계량 계산을 다루는 문제입니다.

**Min-Max 스케일링 공식**:
X_scaled = (X - X_min) / (X_max - X_min)

**주의사항**:
- 스케일링 후 표준편차는 원본과 다름
- fit_transform 사용법 숙지 필요
                """,
                "코드": """
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

# 가상 기후 데이터 생성 (실제 문제에서는 주어진 데이터 사용)
np.random.seed(42)
data = {
    '온도': np.random.uniform(-10, 40, 50),
    '습도': np.random.uniform(10, 90, 50)
}
df = pd.DataFrame(data)

# 1. Min-Max 스케일링
scaler = MinMaxScaler()
df[['온도','습도']] = scaler.fit_transform(df[['온도','습도']])

# 2. 표준편차 계산
temp_std = df['온도'].std()
humidity_std = df['습도'].std()

# 3. 차이 계산 및 반올림
diff = temp_std - humidity_std
result = round(diff, 2)
print(f"표준편차 차이: {result}")
                """
            }
        },
        "작업형2": {
            "문제1": {
                "제목": "호텔 예약 관리 시스템",
                "난이도": "⭐⭐⭐⭐",
                "주요개념": ["회귀모델", "RandomForest", "MAE"],
                "내용": """
호텔 예약 관리 시스템에서 고객에게 부과된 총 청구 금액을 예측하세요.

**제공 데이터**:
- hotel_train.csv (훈련 데이터)
- hotel_test.csv (평가용 데이터)

**예측 컬럼**: TotalBill (총 청구액)
**평가 지표**: MAE (Mean Absolute Error)
**제출 형식**: result.csv (pred 컬럼 포함)
                """,
                "해설": """
회귀 문제를 해결하는 표준적인 머신러닝 파이프라인입니다.

**모델 선택 가이드**:
- RandomForest: 안정적이고 해석하기 쉬움
- XGBoost: 높은 성능, 하이퍼파라미터 튜닝 필요
- LinearRegression: 단순하지만 기본적인 성능

**성능 향상 팁**:
1. 피처 엔지니어링
2. 하이퍼파라미터 튜닝
3. 교차 검증 활용
                """,
                "코드": """
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 1. 데이터 로드 및 분리
train = pd.read_csv('hotel_train.csv')
test = pd.read_csv('hotel_test.csv')

X_train = train.drop(columns=['TotalBill'])
y_train = train['TotalBill']
X_test = test

# 2. 모델 학습
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    max_depth=10
)
model.fit(X_train, y_train)

# 3. 예측 수행
y_pred = model.predict(X_test)

# 4. 결과 저장
result = pd.DataFrame({'pred': y_pred})
result.to_csv('result.csv', index=False)

# 5. 성능 평가 (훈련 데이터)
y_train_pred = model.predict(X_train)
mae = mean_absolute_error(y_train, y_train_pred)
print(f"Training MAE: {mae:.4f}")
                """
            }
        },
        "작업형3": {
            "문제1": {
                "제목": "직원 이직 분석",
                "난이도": "⭐⭐⭐",
                "주요개념": ["로지스틱회귀", "통계적유의성", "p-value"],
                "내용": """
주어진 데이터에서 로지스틱 회귀 분석을 수행해 유의확률(p-value)이 0.05 이상인 
유의하지 않은 독립변수의 개수를 구하세요.

**종속변수**: Resign (이직 여부)
**독립변수**: Age, YearsAtCompany, MonthlyIncome, JobSatisfaction, Overtime
                """,
                "해설": """
통계적 추론을 위한 로지스틱 회귀 분석 문제입니다.

**로지스틱 회귀 특징**:
- 이진 분류 문제에 적합
- 계수의 해석이 가능
- p-value를 통한 유의성 검정

**유의성 판단**:
- p-value < 0.05: 통계적으로 유의
- p-value ≥ 0.05: 통계적으로 유의하지 않음
                """,
                "코드": """
import statsmodels.api as sm
import pandas as pd
import numpy as np

# 가상 데이터 생성 (실제 문제에서는 주어진 데이터 사용)
np.random.seed(42)
n = 1000
data = {
    'Age': np.random.randint(20, 60, n),
    'YearsAtCompany': np.random.randint(1, 20, n),
    'MonthlyIncome': np.random.uniform(3000, 12000, n),
    'JobSatisfaction': np.random.randint(1, 5, n),
    'Overtime': np.random.choice([0, 1], n),
    'Resign': np.random.choice([0, 1], n, p=[0.7, 0.3])
}
df = pd.DataFrame(data)

# 독립변수와 종속변수 분리
X = df[['Age', 'YearsAtCompany', 'MonthlyIncome', 'JobSatisfaction', 'Overtime']]
y = df['Resign']

# 상수항 추가
X = sm.add_constant(X)

# 로지스틱 회귀 모델 적합
model = sm.Logit(y, X)
result = model.fit()

# 결과 출력
print(result.summary())

# p-value가 0.05 이상인 변수 개수
non_significant_count = (result.pvalues >= 0.05).sum()
print(f"\\n유의하지 않은 변수 개수: {non_significant_count}")
                """
            },
            "문제2": {
                "제목": "학생 성적 예측 모델",
                "난이도": "⭐⭐⭐",
                "주요개념": ["다중선형회귀", "R-squared", "회귀계수"],
                "내용": """
**문제 2-1**: 다중 선형 회귀 분석을 수행하여 유의확률(p-value)이 가장 작은 변수의 회귀 계수 값을 구하세요. (소수 셋째 자리까지)

**문제 2-2**: 모델의 결정 계수(R-squared) 값을 구하세요. (소수 둘째 자리까지)

**변수**:
- 종속변수: Score (성적)
- 독립변수: StudyHours, Attendance, Participation
                """,
                "해설": """
다중 선형 회귀 분석의 기본기를 평가하는 문제입니다.

**회귀 계수 해석**:
- 계수 값: 독립변수 1단위 증가시 종속변수 변화량
- p-value: 계수의 통계적 유의성
- R-squared: 모델의 설명력 (0~1)

**모델 성능 지표**:
- R-squared가 높을수록 좋은 모델
- 0.7 이상이면 양호한 수준
                """,
                "코드": """
import statsmodels.api as sm
import pandas as pd
import numpy as np

# 가상 학생 데이터 생성
np.random.seed(42)
n = 100
data = {
    'StudyHours': np.random.uniform(0, 15, n),
    'Attendance': np.random.uniform(50, 100, n),
    'Participation': np.random.uniform(0, 10, n)
}
df = pd.DataFrame(data)

# 종속변수 생성 (실제 관계 반영)
df['Score'] = (5 * df['StudyHours'] + 
               0.5 * df['Attendance'] + 
               3 * df['Participation'] + 
               np.random.normal(0, 10, n))

# 독립변수와 종속변수 분리
X = df[['StudyHours', 'Attendance', 'Participation']]
y = df['Score']

# 상수항 추가
X = sm.add_constant(X)

# 다중 선형 회귀 모델 적합
model = sm.OLS(y, X)
result = model.fit()

# 결과 출력
print(result.summary())

# 문제 2-1: p-value가 가장 작은 변수의 회귀 계수
min_pvalue_var = result.pvalues.idxmin()
coefficient = round(result.params[min_pvalue_var], 3)
print(f"\\n가장 유의한 변수({min_pvalue_var})의 계수: {coefficient}")

# 문제 2-2: R-squared 값
r_squared = round(result.rsquared, 2)
print(f"R-squared 값: {r_squared}")
                """
            }
        }
    },
    "9회": {
        "작업형1": {
            "문제1": {
                "제목": "매장별 고객유형 매출 분석",
                "난이도": "⭐⭐⭐",
                "주요개념": ["groupby", "unstack", "절대값"],
                "내용": """
총매출액을 계산한 후, 매장코드별 고객유형(일반 고객 vs VIP 고객) 간의 매출 차이를 분석하여,
차이를 절대값으로 계산한 뒤, 그 중 절대값이 가장 큰 매장코드 번호를 출력하시오.

**데이터 구조**:
- 매장코드: 매장 식별자
- 고객유형: 1(일반), 2(VIP)
- 매출액1, 매출액2: 각각의 매출 정보
- 총매출액: 매출액1 + 매출액2
                """,
                "해설": """
피벗 테이블과 그룹 연산을 활용한 데이터 분석 문제입니다.

**핵심 단계**:
1. 매장코드별, 고객유형별 그룹화
2. unstack으로 피벗 테이블 생성
3. 고객유형 간 차이 계산
4. 절대값 최대인 매장 찾기
                """,
                "코드": """
import pandas as pd
import numpy as np

# 가상 데이터 생성
np.random.seed(42)
n_samples = 1000
data = {
    "매장코드": np.random.choice([101, 202, 303, 404, 505], size=n_samples),
    "고객유형": np.random.choice([1, 2], size=n_samples),
    "매출액1": np.random.randint(500, 20000, size=n_samples),
    "매출액2": np.random.randint(300, 15000, size=n_samples),
}
df = pd.DataFrame(data)
df["총매출액"] = df["매출액1"] + df["매출액2"]

# 매장코드별, 고객유형별 총매출액 합계 계산
grouped = df.groupby(["매장코드", "고객유형"])["총매출액"].sum().unstack()

# 고객유형 간 매출 차이 계산 및 절대값
grouped["차액"] = abs(grouped[1] - grouped[2])

# 차액 절대값이 가장 큰 매장코드 찾기
max_diff_store = grouped["차액"].idxmax()
print(f"차액이 가장 큰 매장코드: {max_diff_store}")
                """
            },
            "문제2": {
                "제목": "교통사고 검거율 분석",
                "난이도": "⭐⭐⭐⭐",
                "주요개념": ["검거율계산", "idxmax", "데이터변형"],
                "내용": """
연도별로 각 유형별 교통사고 검거율(검거건수 / 사고건수)을 계산한 후,
검거율이 가장 높은 교통사고 유형의 검거 건수를 모두 더하여 출력하시오.

**데이터 구조**:
- 연도: 2018, 2019, 2020
- 구분: 사고건수, 검거건수
- 교통사고유형: 10개 유형별 건수
                """,
                "해설": """
복잡한 데이터 변형과 비율 계산을 다루는 고난도 문제입니다.

**해결 전략**:
1. 사고건수와 검거건수 분리
2. 검거율 = 검거건수 / 사고건수 계산
3. 연도별 최고 검거율 유형 찾기
4. 해당 유형의 검거건수 합산
                """,
                "코드": """
import pandas as pd
import numpy as np

# 가상 데이터 생성
np.random.seed(42)
years = [2018, 2019, 2020]
accident_types = ["음주운전", "과속", "신호위반", "중앙선침범", "무면허", 
                 "보행자사고", "이륜차사고", "어린이보호구역사고", "화물차사고", "버스사고"]

data = []
for year in years:
    for category in ["사고건수", "검거건수"]:
        row = [year, category]
        row.extend(np.random.randint(50, 5000, size=len(accident_types)))
        data.append(row)

columns = ["연도", "구분"] + accident_types
df = pd.DataFrame(data, columns=columns)

# 사고건수와 검거건수 분리
accidents = df[df["구분"] == "사고건수"].set_index("연도").drop(columns=["구분"])
arrests = df[df["구분"] == "검거건수"].set_index("연도").drop(columns=["구분"])

# 교통사고 유형별 검거율 계산
arrest_rate = arrests / accidents

# 연도별 최고 검거율 사고유형 찾기
highest_types_per_year = arrest_rate.idxmax(axis=1)

# 해당 유형들의 검거건수 합산
total_arrests = 0
for year, accident_type in highest_types_per_year.items():
    total_arrests += arrests.loc[year, accident_type]

print(f"최고 검거율 유형들의 총 검거건수: {total_arrests}")
                """
            }
        }
    }
}

# ChatGPT API 호출 함수
def get_chatgpt_response(question, context, chat_history=[]):
    if not st.session_state.openai_api_key:
        return "❌ OpenAI API 키가 설정되지 않았습니다. GitHub Secrets 또는 환경변수에 OPENAI_API_KEY를 설정해주세요."
    
    try:
        openai.api_key = st.session_state.openai_api_key
        
        # 시스템 메시지 구성
        system_message = f"""당신은 빅데이터분석기사 실기 시험 전문 강사입니다. 
학생들의 질문에 친절하고 자세하게 답변해주세요.

현재 학습 중인 문제:
{context}

답변 가이드라인:
1. 🎯 핵심 개념을 명확히 설명
2. 📝 단계별 해결 방법 제시
3. 💻 실용적인 코드 예시 포함
4. ⚠️ 주의사항과 실무 팁 제공
5. 🔍 관련 문제나 확장 학습 방향 제안

한국어로 답변하며, 이모지를 적절히 사용하여 가독성을 높여주세요."""

        # 메시지 구성
        messages = [{"role": "system", "content": system_message}]
        
        # 이전 대화 기록 추가 (최근 3개만)
        for qa in chat_history[-3:]:
            messages.append({"role": "user", "content": qa["question"]})
            messages.append({"role": "assistant", "content": qa["answer"]})
        
        # 현재 질문 추가
        messages.append({"role": "user", "content": question})
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=2000,
            temperature=0.7,
            presence_penalty=0.1,
            frequency_penalty=0.1
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ 오류가 발생했습니다: {str(e)}\n\n🔧 해결 방법:\n1. API 키가 올바른지 확인\n2. 인터넷 연결 상태 확인\n3. OpenAI 계정 크레딧 확인"

# 문제 선택 함수
def problem_selector():
    st.markdown("### 📚 기출문제 선택")
    
    # 회차 선택
    exam_rounds = list(EXAM_DATA.keys())
    selected_round = st.selectbox("🗓️ 시험 회차", exam_rounds, key="exam_round")
    
    # 문제 유형 선택
    problem_types = list(EXAM_DATA[selected_round].keys())
    selected_type = st.selectbox("📝 문제 유형", problem_types, key="problem_type")
    
    # 세부 문제 선택
    problems = list(EXAM_DATA[selected_round][selected_type].keys())
    selected_problem = st.selectbox("🎯 세부 문제", problems, key="problem_detail")
    
    # 선택된 문제 정보
    problem_data = EXAM_DATA[selected_round][selected_type][selected_problem]
    
    # 진도 체크
    problem_id = f"{selected_round}_{selected_type}_{selected_problem}"
    is_completed = st.checkbox(
        "✅ 학습 완료", 
        value=st.session_state.study_progress.get(problem_id, False),
        key=f"progress_{problem_id}"
    )
    st.session_state.study_progress[problem_id] = is_completed
    
    return selected_round, selected_type, selected_problem, problem_data

# 헤더
st.markdown("""
<div class="main-header fade-in">
    <h1><i class="fas fa-graduation-cap"></i> 빅데이터분석기사 실기 Q&A</h1>
    <p><i class="fas fa-robot"></i> AI 튜터와 함께하는 개인 맞춤형 학습 시스템</p>
    <p style="font-size: 0.9rem; opacity: 0.9;">
        <i class="fas fa-star"></i> 기출문제 완벽 분석 | <i class="fas fa-brain"></i> 실시간 AI 답변 | <i class="fas fa-chart-line"></i> 학습 진도 관리
    </p>
</div>
""", unsafe_allow_html=True)

# API 키 확인 (자동 설정됨)
if not st.session_state.openai_api_key:
    st.markdown("""
    <div class="content-card fade-in">
        <div style="text-align: center; padding: 2rem;">
            <i class="fas fa-exclamation-triangle" style="font-size: 3rem; color: #dc3545; margin-bottom: 1rem;"></i>
            <h3>⚠️ API 키 설정 필요</h3>
            <p>API 키가 환경변수에 설정되지 않았습니다.</p>
            <p>GitHub Secrets 또는 Streamlit Secrets에 OPENAI_API_KEY를 추가해주세요.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 상단 통계 정보
col1, col2, col3, col4 = st.columns(4)
completed = len([v for v in st.session_state.study_progress.values() if v])
total_problems = 15
progress_rate = completed / total_problems if total_problems > 0 else 0

with col1:
    st.markdown(f"""
    <div class="metric-card fade-in">
        <i class="fas fa-question-circle" style="color: #667eea; font-size: 1.5rem;"></i>
        <h3 style="margin: 0.5rem 0; color: {secondary_text};">{st.session_state.total_questions}</h3>
        <p style="margin: 0; font-size: 0.9rem;">총 질문 수</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card fade-in">
        <i class="fas fa-check-circle" style="color: #28a745; font-size: 1.5rem;"></i>
        <h3 style="margin: 0.5rem 0; color: #28a745;">{completed}</h3>
        <p style="margin: 0; font-size: 0.9rem;">완료 문제</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card fade-in">
        <i class="fas fa-chart-line" style="color: #ffc107; font-size: 1.5rem;"></i>
        <h3 style="margin: 0.5rem 0; color: #ffc107;">{progress_rate * 100:.0f}%</h3>
        <p style="margin: 0; font-size: 0.9rem;">진도율</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card fade-in">
        <i class="fas fa-robot" style="color: #e91e63; font-size: 1.5rem;"></i>
        <h3 style="margin: 0.5rem 0; color: #e91e63;">{"ON" if st.session_state.openai_api_key else "OFF"}</h3>
        <p style="margin: 0; font-size: 0.9rem;">AI 상태</p>
    </div>
    """, unsafe_allow_html=True)

# 메인 컨텐츠 - 문제 선택 섹션
st.markdown('<div class="content-card fade-in">', unsafe_allow_html=True)
exam_round, problem_type, problem_num, problem_data = problem_selector()

st.markdown("---")

# 문제 정보 카드
st.markdown(f"""
<div class="problem-card">
    <h3><i class="fas fa-bookmark"></i> {exam_round} {problem_type} {problem_num}</h3>
    <h4 style="color: {secondary_text}; margin: 1rem 0;">📋 {problem_data['제목']}</h4>
    <div style="display: flex; gap: 1rem; margin-bottom: 1rem;">
        <span><strong>난이도:</strong> {problem_data['난이도']}</span>
        <span><strong>핵심 개념:</strong> {', '.join(problem_data['주요개념'])}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# 탭으로 구성
tab1, tab2, tab3 = st.tabs(["📋 문제 내용", "💡 해설", "💻 코드"])

with tab1:
    st.markdown(problem_data['내용'])

with tab2:
    st.markdown(problem_data['해설'])

with tab3:
    st.code(problem_data['코드'], language='python')

st.markdown('</div>', unsafe_allow_html=True)

# AI 튜터 섹션
st.markdown('<div class="chat-card fade-in">', unsafe_allow_html=True)
st.markdown("### 🤖 AI 튜터에게 질문하기")

# 채팅 히스토리 표시
if st.session_state.chat_history:
    st.markdown("#### 💬 최근 대화")
    with st.container():
        for i, chat in enumerate(st.session_state.chat_history[-3:]):
            with st.expander(f"Q{len(st.session_state.chat_history)-2+i}: {chat['question'][:30]}..."):
                st.markdown(f"**👤 질문:** {chat['question']}")
                st.markdown(f"""
                <div class="ai-response">
                    <strong>🤖 AI 답변:</strong><br>
                    {chat['answer']}
                </div>
                """, unsafe_allow_html=True)

# 퀵 질문 선택 확인
if 'selected_quick_question' in st.session_state and st.session_state.selected_quick_question:
    default_question = st.session_state.selected_quick_question
    st.session_state.selected_quick_question = None  # 사용 후 초기화
else:
    default_question = ""

# 질문 입력
question = st.text_area(
    "궁금한 점을 자유롭게 질문하세요:",
    placeholder="예시:\n- 이 문제에서 groupby는 어떻게 작동하나요?\n- Min-Max 스케일링의 장단점은?\n- 다른 해결 방법도 있을까요?",
    height=120,
    value=default_question,
    key="question_input"
)

# 질문 버튼
col_btn1, col_btn2 = st.columns([2, 1])
with col_btn1:
    ask_button = st.button("🚀 질문하기", type="primary", use_container_width=True)
with col_btn2:
    clear_button = st.button("🗑️ 기록 삭제", use_container_width=True)

if clear_button:
    st.session_state.chat_history = []
    st.success("대화 기록이 삭제되었습니다!")
    time.sleep(1)
    st.rerun()

if ask_button and question.strip():
    if st.session_state.openai_api_key:
        with st.spinner("🤔 AI가 답변을 생각하고 있어요..."):
            # 컨텍스트 정보 구성
            context = f"""
            문제: {exam_round} {problem_type} {problem_num} - {problem_data['제목']}
            난이도: {problem_data['난이도']}
            주요개념: {', '.join(problem_data['주요개념'])}
            내용: {problem_data['내용']}
            해설: {problem_data['해설']}
            코드: {problem_data['코드']}
            """
            
            # ChatGPT 응답 받기
            response = get_chatgpt_response(question, context, st.session_state.chat_history)
            
            # 채팅 기록에 추가
            st.session_state.chat_history.append({
                "question": question,
                "answer": response,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "problem": f"{exam_round} {problem_type} {problem_num}"
            })
            
            st.session_state.total_questions += 1
            
            # 응답 표시
            st.markdown("#### 💡 AI 답변")
            st.markdown(f"""
            <div class="ai-response fade-in">
                {response}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("AI 답변을 받으려면 API 키가 필요합니다!")

elif ask_button:
    st.warning("질문을 입력해주세요!")

# 퀵 질문 버튼들
st.markdown("#### ⚡ 빠른 질문")
quick_questions = [
    "이 문제의 핵심 개념은?",
    "다른 해결 방법은?",
    "실무에서는 어떻게 활용?",
    "비슷한 문제 유형은?"
]

cols = st.columns(2)
for i, q in enumerate(quick_questions):
    with cols[i % 2]:
        if st.button(q, key=f"quick_{i}", use_container_width=True):
            st.session_state.selected_quick_question = q
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# 인프런 강의 광고
st.markdown("""
<div class="inflearn-ad fade-in">
    <div style="position: relative; z-index: 1;">
        <h2 style="margin-bottom: 1rem; font-size: 2rem;">
            <i class="fas fa-graduation-cap"></i> 더 깊이 있는 학습을 원하신다면?
        </h2>
        <p style="font-size: 1.2rem; margin-bottom: 0.5rem; opacity: 0.95;">
            <strong>빅데이터분석기사 실기 완전정복 과정</strong>
        </p>
        <p style="font-size: 1rem; margin-bottom: 1.5rem; opacity: 0.9;">
            ✅ 8회~최신회차 기출문제 완벽 분석 | ✅ 실무 중심의 문제 해결 전략<br>
            ✅ 1:1 질문 답변 및 피드백 | ✅ 합격까지 완벽 가이드
        </p>
        <a href="https://inf.run/ZRXQe" target="_blank" class="inflearn-button">
            <i class="fas fa-play-circle"></i> 인프런에서 수강하기
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

# 하단 정보
current_date = datetime.now().strftime('%Y년 %m월 %d일')
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 2rem; margin-top: 2rem;">
    <p style="margin-bottom: 0.5rem;">
        <i class="fas fa-clock"></i> 마지막 업데이트: {current_date}
    </p>
    <p style="font-size: 0.9rem; opacity: 0.8;">
        Powered by OpenAI ChatGPT | Made with <i class="fas fa-heart" style="color: #e74c3c;"></i> by BigData Q&A Team
    </p>
</div>
""", unsafe_allow_html=True)
