# 사이드바 설정
with st.sidebar:
    # 인프런 SQL 강의 광고 배너
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 5px 15px rgba(255, 193, 7, 0.3);
        text-align: center;
        animation: pulse 2s infinite;
        position: relative;
        overflow: hidden;
    ">
        <style>
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.02); }
                100% { transform: scale(1); }
            }
            @keyframes sparkle {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
        <div style="position: absolute; top: 10px; right: 10px; font-size: 1.5rem; animation: sparkle 3s linear infinite;">✨</div>
        <div style="position: absolute; bottom: 10px; left: 10px; font-size: 1.5rem; animation: sparkle 3s linear infinite reverse;">⭐</div>
        
        <h3 style="color: #333; margin: 0 0 0.5rem 0; font-size: 1.1rem; font-weight: 700;">
            🎯 SQL 마스터 되기!
        </h3>
        <p style="color: #444; font-size: 0.9rem; margin: 0.5rem 0; font-weight: 500;">
            데이터 분석의 시작<br>
            <strong>실무 SQL 완전정복</strong>
        </p>
        <div style="
            background: white;
            color: #FF6B00;
            padding: 0.5rem 1rem;
            border-radius: 25px;
            display: inline-block;
            margin: 0.5rem 0;
            font-weight: 700;
            font-size: 0.85rem;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        ">
            🔥 인프런 인기 강의
        </div>
        <a href="https://inf.run/R9Te3" target="_blank" style="
            display: inline-block;
            background: #FF6B00;
            color: white;
            padding: 0.7rem 1.5rem;
            border-radius: 25px;
            text-decoration: none;
            font-weight: 700;
            margin-top: 0.5rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(255, 107, 0, 0.3);
        " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(255, 107, 0, 0.4)';" 
           onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 15px rgba(255, 107, 0, 0.3)';">
            수강하러 가기 →
        </a>
        <p style="color: #666; font-size: 0.75rem; margin-top: 0.5rem; margin-bottom: 0;">
            <em>* 커리어 성장의 필수 스킬</em>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
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
