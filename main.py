import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# CSS: 한글 폰트
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# 탭 제목
# ─────────────────────────────────────────────
st.set_page_config(page_title="극지식물 생육 대시보드", layout="wide")
st.title("토양재배와 수경재배 환경이 극지식물의 생육 특성에 어떤 영향을 미칠까?")

# ─────────────────────────────────────────────
# CSV 읽기 함수 (인코딩 이중 시도)
# ─────────────────────────────────────────────
@st.cache_data
def read_csv(file):
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()

# ─────────────────────────────────────────────
# 열 이름 표준화
# ─────────────────────────────────────────────
def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + pd.to_timedelta(
                parts[1].astype(int), unit="h"
            )
        df["time"] = parsed
    return df

# ─────────────────────────────────────────────
# 사이드바: CSV 업로드 + 생육 측정 항목 선택
# ─────────────────────────────────────────────
st.sidebar.header("📁 데이터 업로드")
env_file = st.sidebar.file_uploader("환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 데이터 CSV", type="csv")

growth_options = [
    "잎 수",
    "지상부 길이",
    "지하부 길이",
    "지상부 질량",
    "지하부 질량",
    "생중량",
]
selected_growth = st.sidebar.multiselect(
    "🌱 생육 측정 항목 선택 (탭2·탭3에서 사용)",
    growth_options,
    default=["잎 수", "지상부 길이", "지하부 길이", "지상부 질량", "지하부 질량"],
)

# ─────────────────────────────────────────────
# 데이터 로딩 (한 번만)
# ─────────────────────────────────────────────
env_df = pd.DataFrame()
growth_df = pd.DataFrame()
school_name = "아직 업로드하지 않음"

if env_file is not None:
    with st.spinner("환경 데이터를 읽는 중..."):
        raw = read_csv(env_file)
        if not raw.empty:
            env_df = std_cols(raw)
            # 파일명에서 학교명 추출 (앞부분만)
            fn = os.path.basename(env_file.name)
            school_name = fn.split("_")[0] if "_" in fn else fn.replace(".csv", "")
if growth_file is not None:
    with st.spinner("생육 데이터를 읽는 중..."):
        raw = read_csv(growth_file)
        if not raw.empty:
            growth_df = std_cols(raw)

st.sidebar.markdown(f"**대시보드 실행 학교:** `{school_name}`")
st.sidebar.markdown("---")
st.sidebar.info(
    "✅ CSV를 업로드하면 자동으로 열 이름을 표준화하고\n"
    "   인코딩(utf-8-sig / cp949)을 시도합니다.\n"
    "   학교마다 열 이름이 달라도 동작하도록 설계되어 있습니다."
)

# ─────────────────────────────────────────────
# 도움말: 성장 측정 항목 열 이름 매핑
# ─────────────────────────────────────────────
GROWTH_COL_MAP = {
    "잎 수": "잎 수",
    "지상부 길이": "지상부 길이",
    "지하부 길이": "지하부 길이",
    "지상부 질량": "지상부 질량",
    "지하부 질량": "지하부 질량",
    "생중량": "생중량",
}

# ─────────────────────────────────────────────
# 그래프 공용: 추세선 추가 헬퍼
# ─────────────────────────────────────────────
def add_trendline(fig, x_vals, y_vals, color="red", dash="dash"):
    valid = pd.concat([x_vals, y_vals], axis=1).dropna()
    if len(valid) < 2:
        return
    x = valid.iloc[:, 0].values
    y = valid.iloc[:, 1].values
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.array([x.min(), x.max()])
    y_line = slope * x_line + intercept
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            line=dict(color=color, dash=dash, width=2),
            name=f"추세선 (r 계산)",
            hoverinfo="skip",
        )
    )

# ─────────────────────────────────────────────
# 상관계수 표시 헬퍼
# ─────────────────────────────────────────────
def corr_text(x_series, y_series):
    v = pd.concat([x_series, y_series], axis=1).dropna()
    if len(v) < 2:
        return " 데이터가 충분하지 않아 상관계수를 계산할 수 없습니다."
    r = v.iloc[:, 0].corr(v.iloc[:, 1])
    direction = "양의" if r > 0 else "음의"
    strength = (
        "매우 강한" if abs(r) >= 0.8
        else "강한" if abs(r) >= 0.6
        else "중간" if abs(r) >= 0.4
        else "약한" if abs(r) >= 0.2
        else "매우 약한"
    )
    return f"  상관계수 r = {r:.3f} ({direction} {strength} 상관관계)"

# ─────────────────────────────────────────────
# 탭 구성
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔬 가설 및 주제 소개", "📊 데이터 비교 분석", "🎯 재배 적합성 검토"])

# ═════════════════════════════════════════════
# 탭 1: 가설 및 주제 소개
# ═════════════════════════════════════════════
with tab1:
    try:
        st.markdown("## 📋 연구 주제")
        st.info(
            "극지식물은 낮은 온도, 적은 수분, 강한 자외선 등 가혹한 환경에 적응한 식물입니다. "
            "이 대시보드에서는 토양재배와 수경재배 환경에서 키운 극지식물의 생육 차이를 비교 분석합니다."
        )

        st.markdown("## 🧪 가설")
        st.markdown("""
**가설 (1)**  
상대적으로 높은 기온과 제어된 광조건은 식물의 대사 활동을 촉진하여  
잎 수(평균 32.17장)를 토양재배(평균 21.73장)보다 유의미하게 증가시키는 원인이 될 것이다.

> 💡 **평균**: 여러 개체의 값을 모두 더한 뒤 개체 수로 나눈 값입니다.  
>    예: 잎 수가 30장, 40장, 20장인 개체 3개의 평균은 (30+40+20)÷3 = 30장입니다.
""")

        st.markdown("""
**가설 (2)**  
높은 상대습도 환경에서는 지하부의 길이가 짧고,  
낮은 상대습도 환경에서는 지하부의 길이가 길 것이다.

> 💡 **상대습도**: 공기 중에 포함된 수증기량이 그 온도에서 최대로 포함할 수 있는 수증기량의 몇 %인지를 나타냅니다.  
>    상대습도가 높으면 식물이 물을 덜 빼앗기므로 뿌리를 길게 뻗을 필요가 줄어듭니다.
""")

        st.markdown("""
**가설 (3)**  
상대습도가 낮은 환경 조건에서 증산 작용을 조절하기 위해  
잎이 수경재배 특유의 두껍고 조밀한 형태적 구조를 형성할 것이다.

> 💡 **증산 작용**: 식물의 잎에서 수분이 수증기 형태로 빠져나가는 현상입니다.  
>    습도가 낮으면 수분 손실이 커지므로 잎을 두껍고 조밀하게 만들어 이를 조절합니다.
""")

        if env_df.empty and growth_df.empty:
            st.info("📁 사이드바에서 CSV 파일을 업로드하면 탭2·탭3의 분석이 활성화됩니다.")
        else:
            st.markdown("## 📁 업로드된 데이터 미리보기")
            col1, col2 = st.columns(2)
            with col1:
                if not env_df.empty:
                    st.subheader(f"📊 환경 데이터 — {school_name}")
                    st.info(f"총 {len(env_df)}개 행 • 열: {', '.join(env_df.columns)}")
                    st.dataframe(env_df.head(10), width="stretch")
                else:
                    st.info("환경 CSV가 업로드되지 않았습니다.")
            with col2:
                if not growth_df.empty:
                    st.subheader(f"🌱 생육 결과 데이터 — {school_name}")
                    st.info(f"총 {len(growth_df)}개 개체(행) • 열: {', '.join(growth_df.columns)}")
                    st.dataframe(growth_df.head(10), width="stretch")
                else:
                    st.info("생육 CSV가 업로드되지 않았습니다.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ═════════════════════════════════════════════
# 탭 2: 환경 데이터 및 생육 데이터 비교 분석
# ═════════════════════════════════════════════
with tab2:
    try:
        st.markdown("## 📊 환경 데이터 시계열 및 생육 데이터 비교")

        # ── 2-1. 환경 데이터 시계열 (온도·습도·CO₂) ──────────────────
        if not env_df.empty and "time" in env_df.columns:
            st.markdown("### 🌡️ 환경 변수 시계열 (전체 기간)")
            env_cols = [c for c in ["temperature", "humidity", "ph", "ec", "co2"] if c in env_df.columns]
            if env_cols:
                fig_env = make_subplots(
                    rows=len(env_cols),
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.04,
                    subplot_titles=[c.replace("temperature", "온도(℃)").replace("humidity", "습도(%)").replace("ph", "pH").replace("ec", "EC").replace("co2", "CO₂(ppm)") for c in env_cols],
                )
                for i, col in enumerate(env_cols, 1):
                    s = env_df[col].dropna()
                    if s.empty:
                        continue
                    fig_env.add_trace(
                        go.Scatter(
                            x=env_df["time"],
                            y=env_df[col],
                            mode="lines",
                            name=col,
                            line=dict(width=1.2),
                        ),
                        row=i,
                        col=1,
                    )
                fig_env.update_layout(
                    height=220 * len(env_cols) + 60,
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    showlegend=False,
                    margin=dict(l=40, r=20, t=40, b=20),
                )
                fig_env.update_xaxes(showgrid=True, gridcolor="lightgrey")
                fig_env.update_yaxes(showgrid=True, gridcolor="lightgrey")
                st.plotly_chart(fig_env, width="stretch")
            else:
                st.info("환경 CSV에 표시할 환경 변수(온도·습도·pH·EC·CO₂) 열이 없습니다.")
        else:
            st.info("환경 CSV를 업로드하면 환경 변수 시계열 그래프가 표시됩니다.")

        # ── 2-2. 생육 데이터 분포: Box plot ──────────────────────────
        if not growth_df.empty:
            st.markdown("### 🌱 생육 지표 분포 (개체별 상자 그림)")
            gc = [GROWTH_COL_MAP[g] for g in selected_growth if GROWTH_COL_MAP[g] in growth_df.columns]
            if gc:
                fig_box = make_subplots(
                    rows=1,
                    cols=len(gc),
                    shared_yaxes=False,
                    subplot_titles=gc,
                    horizontal_spacing=0.08,
                )
                colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]
                for i, col in enumerate(gc, 1):
                    vals = growth_df[col].dropna()
                    if vals.empty:
                        continue
                    unit = ""
                    if col == "잎 수":
                        unit = "장"
                    elif "길이" in col:
                        unit = "mm"
                    elif "질량" in col or col == "생중량":
                        unit = "g"
                    fig_box.add_trace(
                        go.Box(
                            y=vals.values,
                            name=col,
                            fillcolor=colors[i % len(colors)],
                            boxpoints="suspectedoutliers",
                            marker=dict(size=4),
                            line=dict(width=1.5),
                        ),
                        row=1,
                        col=i,
                    )
                    fig_box.update_yaxes(title_text=f"{col} ({unit})", row=1, col=i)
                fig_box.update_layout(
                    height=380,
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    showlegend=False,
                    margin=dict(l=40, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_box, width="stretch")

                # 생육 지표 요약 통계
                st.markdown("#### 📋 생육 지표 요약 통계")
                stats_cols = ["지표", "평균", "표준편차", "최소", "최대", "개체 수"]
                stats_rows = []
                for col in gc:
                    vals = growth_df[col].dropna()
                    if len(vals) < 1:
                        continue
                    stats_rows.append([
                        col,
                        f"{vals.mean():.2f}",
                        f"{vals.std(ddof=1):.2f}" if len(vals) > 1 else "N/A",
                        f"{vals.min():.2f}",
                        f"{vals.max():.2f}",
                        str(len(vals)),
                    ])
                if stats_rows:
                    st.dataframe(pd.DataFrame(stats_rows, columns=stats_cols), width="stretch")
                else:
                    st.info("표시할 생육 지표 데이터가 충분하지 않습니다.")
            else:
                st.info("선택한 생육 측정 항목 중 실제 데이터가 있는 열이 없습니다. 사이드바에서 항목을 선택해주세요.")

            # ── 2-3. 개체별 생중량 상위·하위 강조 ────────────────────
            if "생중량" in growth_df.columns:
                st.markdown("### 🏆 개체별 생중량 비교 (상위 5 / 하위 5)")
                gw = growth_df[["개체번호", "생중량"]].dropna().sort_values("생중량", ascending=False)
                if len(gw) >= 2:
                    top5 = gw.head(5)
                    bot5 = gw.tail(5).iloc[::-1]
                    fig_bar = go.Figure()
                    # 상위
                    fig_bar.add_trace(
                        go.Bar(
                            x=top5["개체번호"].astype(str).tolist(),
                            y=top5["생중량"].tolist(),
                            name="상위 5개체",
                            marker_color="#2E86AB",
                            text=top5["생중량"].round(2),
                            textposition="outside",
                        )
                    )
                    # 하위
                    fig_bar.add_trace(
                        go.Bar(
                            x=bot5["개체번호"].astype(str).tolist(),
                            y=bot5["생중량"].tolist(),
                            name="하위 5개체",
                            marker_color="#A23B72",
                            text=bot5["생중량"].round(2),
                            textposition="outside",
                        )
                    )
                    fig_bar.update_layout(
                        barmode="group",
                        title="개체별 생중량(g) — 상위 5 / 하위 5",
                        yaxis_title="생중량 (g)",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        height=380,
                        margin=dict(l=40, r=20, t=50, b=20),
                    )
                    st.plotly_chart(fig_bar, width="stretch")

                    # 생중량 통계 카드
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("평균 생중량", f"{gw['생중량'].mean():.2f} g")
                    c2.metric("최대 생중량", f"{gw['생중량'].max():.2f} g (개체 {gw.loc[gw['생중량'].idxmax(), '개체번호']})")
                    c3.metric("최소 생중량", f"{gw['생중량'].min():.2f} g (개체 {gw.loc[gw['생중량'].idxmin(), '개체번호']})")
                    c4.metric("생체중 개체 수", str(len(gw)))
                else:
                    st.info("생중량 데이터가 충분하지 않아 비교할 수 없습니다.")
            else:
                st.info("생육 CSV에 '생중량' 열이 없습니다. (지상부 질량 + 지하부 질량으로 자동 계산됩니다.)")

            # ── 2-4. 환경 vs 생육 산점도 + 추세선 ────────────────────
            if not env_df.empty and not growth_df.empty:
                st.markdown("### 🔬 환경 변수 × 생육 지표 산점도")
                # 환경 변수가 1개라도 있고 생육 지표가 1개라도 있으면 표시
                env_avail = [c for c in ["temperature", "humidity", "ph", "ec", "co2"] if c in env_df.columns]
                grow_avail = [GROWTH_COL_MAP[g] for g in selected_growth if GROWTH_COL_MAP[g] in growth_df.columns]
                if env_avail and grow_avail:
                    # 생육 데이터와 환경 데이터의 시간 범위가 다를 수 있으므로,
                    # 환경 데이터는 일평균으로 요약해 생육 지표와 대략적 비교
                    if "time" in env_df.columns:
                        env_daily = (
                            env_df.set_index("time")[env_avail]
                            .resample("D")
                            .mean()
                            .dropna()
                            .reset_index()
                        )
                    else:
                        env_daily = env_df[env_avail].dropna().reset_index(drop=True)

                    env_col_disp = {
                        "temperature": "평균 온도 (℃)",
                        "humidity": "평균 습도 (%)",
                        "ph": "pH",
                        "ec": "EC",
                        "co2": "CO₂ (ppm)",
                    }

                    for gc2 in grow_avail:
                        # 생육 데이터는 평균 1개 값으로 비교 (개체별 편차가 있으므로 평균 사용)
                        g_val = growth_df[gc2].dropna().mean()
                        if env_daily.empty or pd.isna(g_val):
                            continue
                        st.markdown(f"#### {gc2} vs 환경 변수")
                        fig_scatter = make_subplots(
                            rows=len(env_avail),
                            cols=1,
                            shared_xaxes=False,
                            subplot_titles=[env_col_disp.get(c, c) for c in env_avail],
                            vertical_spacing=0.05,
                        )
                        for i, ec2 in enumerate(env_avail, 1):
                            ev = env_daily[ec2].dropna()
                            if ev.empty:
                                continue
                            xv = ev.values
                            yv = np.full(len(xv), g_val)  # 생육 평균값을 수평선으로 표시
                            fig_scatter.add_trace(
                                go.Scatter(
                                    x=xv,
                                    y=yv,
                                    mode="lines+markers",
                                    name=gc2,
                                    line=dict(color="#C44E52", width=2),
                                    marker=dict(size=4),
                                    hovertemplate=f"{gc2}: {g_val:.2f}<extra></extra>",
                                ),
                                row=i,
                                col=1,
                            )
                            # 해당 환경의 분포를 히스토그램으로 함께 표시
                            fig_scatter.add_trace(
                                go.Histogram(
                                    x=xv,
                                    name=f"{env_col_disp.get(ec2, ec2)} 분포",
                                    opacity=0.4,
                                    marker_color="#4C72B0",
                                    hovertemplate=f"{env_col_disp.get(ec2, ec2)}<extra></extra>",
                                ),
                                row=i,
                                col=1,
                            )
                        fig_scatter.update_layout(
                            height=200 * len(env_avail) + 40,
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02),
                            margin=dict(l=40, r=20, t=40, b=20),
                        )
                        st.plotly_chart(fig_scatter, width="stretch")
                        st.info(
                            f"💡 위 그래프는 환경 데이터의 시계열 분포를 보여주고, "
                            f"빨간 선은 `{gc2}`의 전체 개체 평균값({g_val:.2f})을 나타냅니다. "
                            f"환경 데이터는 일별 평균으로 요약했습니다."
                        )
                else:
                    st.info("환경 변수 또는 생육 측정 항목 열이 부족하여 산점도를 표시할 수 없습니다.")
        else:
            st.info("생육 CSV를 업로드하면 생육 분포 그래프와 개체 비교가 표시됩니다.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ═════════════════════════════════════════════
# 탭 3: 재배 목적에 따른 토양·수경재배 적합성 검토
# ═════════════════════════════════════════════
with tab3:
    try:
        st.markdown("## 🎯 재배 목적에 따른 토양·수경재배 적합성 검토")

        st.info(
            "이 탭에서는 업로드된 생육 데이터를 바탕으로 각 재배 방식의 특성을 비교합니다. "
            "실제 CSV에 재배방식 열이 없는 경우, 전체 개체 데이터를 기준으로 분석합니다."
        )

        # ── 3-1. 생육 지표 간 상관관계 매트릭스 ──────────────────────
        if not growth_df.empty:
            gc3 = [GROWTH_COL_MAP[g] for g in selected_growth if GROWTH_COL_MAP[g] in growth_df.columns]
            if len(gc3) >= 2:
                st.markdown("### 🔗 생육 지표 간 상관관계")
                corr_df = growth_df[gc3].dropna()
                if len(corr_df) >= 2:
                    corr_mat = corr_df.corr()
                    fig_corr = px.imshow(
                        corr_mat,
                        text_auto=".2f",
                        color_continuous_scale="RdBu_r",
                        zmin=-1,
                        zmax=1,
                        labels=dict(x="생육 지표", y="생육 지표", color="r"),
                        aspect="auto",
                    )
                    fig_corr.update_layout(
                        title="생육 지표 상관계수 매트릭스 (r)",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        coloraxis_colorbar=dict(title="r"),
                        margin=dict(l=40, r=20, t=50, b=20),
                    )
                    st.plotly_chart(fig_corr, width="stretch")

                    # 주요 상관관계 설명
                    pairs = []
                    for i, c1 in enumerate(gc3):
                        for c2 in gc3[i + 1:]:
                            v = corr_df[[c1, c2]].dropna()
                            if len(v) >= 2:
                                r = v[c1].corr(v[c2])
                                pairs.append((c1, c2, r))
                    if pairs:
                        st.markdown("#### 📌 주요 상관관계 설명")
                        for c1, c2, r in sorted(pairs, key=lambda x: abs(x[2]), reverse=True):
                            direction = "정(+)의" if r > 0 else "부(-)의"
                            strength = (
                                "매우 강한" if abs(r) >= 0.8
                                else "강한" if abs(r) >= 0.6
                                else "중간" if abs(r) >= 0.4
                                else "약한" if abs(r) >= 0.2
                                else "매우 약한"
                            )
                            st.markdown(
                                f"- **{c1} ↔ {c2}**: r = {r:.3f} → {direction} {strength} 상관관계"
                            )
                else:
                    st.info("상관계수를 계산하기에 데이터가 충분하지 않습니다.")
            else:
                st.info("상관행렬을 만들려면 2개 이상의 생육 측정 항목을 선택해주세요.")

            # ── 3-2. 생육 지표 간 산점도 (쌍별) ────────────────────
            if len(gc3) >= 2:
                st.markdown("### 🔬 생육 지표 쌍별 산점도")
                n = len(gc3)
                rows_n = (n * (n - 1)) // 2
                if rows_n <= 12:
                    fig_pairs = make_subplots(
                        rows=rows_n,
                        cols=1,
                        shared_xaxes=False,
                        subplot_titles=None,
                        vertical_spacing=0.03,
                    )
                    idx = 0
                    for i in range(n):
                        for j in range(i + 1, n):
                            idx += 1
                            x_col = gc3[i]
                            y_col = gc3[j]
                            xy = growth_df[[x_col, y_col]].dropna()
                            if len(xy) < 2:
                                continue
                            fig_pairs.add_trace(
                                go.Scatter(
                                    x=xy[x_col],
                                    y=xy[y_col],
                                    mode="markers",
                                    marker=dict(size=8, color="#4C72B0", opacity=0.75),
                                    hovertemplate=f"{x_col}: %{{x:.1f}}<br>{y_col}: %{{y:.1f}}<extra></extra>",
                                ),
                                row=idx,
                                col=1,
                            )
                            add_trendline(fig_pairs, xy[x_col], xy[y_col])
                            fig_pairs.update_xaxes(title_text=x_col, row=idx, col=1)
                            fig_pairs.update_yaxes(title_text=y_col, row=idx, col=1)
                    fig_pairs.update_layout(
                        height=220 * min(rows_n, 8) + 40,
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        showlegend=False,
                        margin=dict(l=40, r=20, t=40, b=20),
                    )
                    st.plotly_chart(fig_pairs, width="stretch")
                else:
                    st.info("선택한 생육 지표가 많아 그래프가 너무 많아집니다. 항목을 줄여주세요.")

        # ── 3-3. 환경 데이터 요약 통계 ──────────────────────────────
        if not env_df.empty:
            st.markdown("### 🌡️ 환경 데이터 요약")
            env_cols = [c for c in ["temperature", "humidity", "ph", "ec", "co2"] if c in env_df.columns]
            if env_cols:
                unit_map = {"temperature": "℃", "humidity": "%", "ph": "", "ec": "mS/cm", "co2": "ppm"}
                env_stats = []
                for col in env_cols:
                    s = env_df[col].dropna()
                    if s.empty:
                        continue
                    env_stats.append([
                        col.replace("temperature", "온도").replace("humidity", "습도").replace("ph", "pH").replace("ec", "EC").replace("co2", "CO₂"),
                        f"{s.mean():.1f} {unit_map.get(col, '')}",
                        f"{s.min():.1f} {unit_map.get(col, '')}",
                        f"{s.max():.1f} {unit_map.get(col, '')}",
                        f"{s.std(ddof=1):.1f} {unit_map.get(col, '')}" if len(s) > 1 else "N/A",
                        str(len(s)),
                    ])
                if env_stats:
                    st.dataframe(pd.DataFrame(env_stats, columns=["변수", "평균", "최소", "최대", "표준편차", "측정 수"]), width="stretch")
                else:
                    st.info("환경 변수 데이터가 충분하지 않습니다.")
            else:
                st.info("환경 CSV에 온도·습도·pH·EC·CO₂ 열이 없습니다.")
        else:
            st.info("환경 CSV를 업로드하면 환경 데이터 요약 통계가 표시됩니다.")

        # ── 3-4. CSV 다운로드 ──────────────────────────────────────
        st.markdown("### 📥 분석 결과 CSV 다운로드")
        if not env_df.empty or not growth_df.empty:
            download_df = pd.DataFrame()
            download_label = ""
            if not env_df.empty:
                download_df = env_df.copy()
                download_label = "환경 데이터"
            if not growth_df.empty:
                if download_df.empty:
                    download_df = growth_df.copy()
                    download_label = "생육 데이터"
                else:
                    # 둘 다 있으면 생육 데이터를 함께 다운로드
                    download_label = "환경 + 생육 데이터"

            if not download_df.empty:
                csv_bytes = download_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label=f"📥 {download_label} 다운로드 (utf-8-sig)",
                    data=csv_bytes,
                    file_name=f"{school_name}_분석데이터.csv",
                    mime="text/csv",
                )
        else:
            st.info("다운로드할 데이터가 없습니다. CSV를 업로드해주세요.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
