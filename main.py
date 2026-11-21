import io
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


def std_cols(df):
    df = df.copy()
    # 중복 열 이름 처리: 같은 이름이 여러 번 나타나면 접미사 붙여 고유화
    seen = {}
    new_cols = []
    for c in df.columns:
        s = str(c).strip()
        if s in seen:
            seen[s] += 1
            new_cols.append(f"{s}_{seen[s]}")
        else:
            seen[s] = 0
            new_cols.append(s)
    df.columns = new_cols

    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k.endswith("_time") or k.endswith("_날짜"):
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
        col = df[c]
        if isinstance(col, pd.DataFrame):
            continue
        if not pd.api.types.is_numeric_dtype(col):
            num = pd.to_numeric(
                col.astype(str).str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"]
        if isinstance(t, pd.DataFrame):
            df["time"] = pd.to_datetime(t.iloc[:, 0].astype(str), errors="coerce", format="mixed")
        else:
            tt = t.astype(str).str.strip()
            parsed = pd.to_datetime(tt, errors="coerce", format="mixed")
            bad = parsed.isna() & tt.str.match(r"^\d{8}_\d{1,2}$")
            if bad.any():
                parts = tt[bad].str.split("_", expand=True)
                parsed[bad] = (
                    pd.to_datetime(parts[0], format="%Y%m%d")
                    + pd.to_timedelta(parts[1].astype(int), unit="h")
                )
            df["time"] = parsed
    return df


@st.cache_data
def read_csv(file):
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


def convert_csv(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def time_range_str(series):
    s = pd.to_datetime(series.dropna(), errors="coerce")
    s = s.dropna()
    if len(s) == 0:
        return ""
    return f"{s.min().strftime('%Y-%m-%d')} ~ {s.max().strftime('%Y-%m-%d')}"


def corr_plot(df, x_col, y_col, title):
    data = df[[x_col, y_col]].dropna()
    if len(data) < 2:
        return None, None
    x = data[x_col].values
    y = data[y_col].values
    r = np.corrcoef(x, y)[0, 1]
    m, b = np.polyfit(x, y, 1)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(size=8, color="#2ca02c", opacity=0.7),
            name="관측값",
            hovertemplate="%{x:.2f}, %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.sort(x),
            y=m * np.sort(x) + b,
            mode="lines",
            line=dict(color="#d62728", width=2),
            name=f"추세선 (r={r:.2f})",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        width=420,
        height=320,
        margin=dict(l=50, r=20, t=40, b=40),
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig, r


def group_summary(df, cols):
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) < 2:
            continue
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        rows.append({
            "측정 항목": col,
            "평균": f"{s.mean():.2f}",
            "최소": f"{s.min():.2f}",
            "최대": f"{s.max():.2f}",
            "표준편차": f"{s.std():.2f}",
            "변동계수(CV)": f"{s.std()/s.mean()*100:.1f}%",
        })
    return pd.DataFrame(rows)


def two_group_bar(df_a, df_b, col, name_a, name_b, color_a, color_b):
    a = df_a[["개체번호", col]].dropna()
    b = df_b[["개체번호", col]].dropna()
    if len(a) < 2 and len(b) < 2:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=a["개체번호"].astype(str),
            y=a[col],
            name=name_a,
            marker_color=color_a,
            hovertemplate="%{x}<br>%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=b["개체번호"].astype(str),
            y=b[col],
            name=name_b,
            marker_color=color_b,
            hovertemplate="%{x}<br>%{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{col} 비교: {name_a} vs {name_b}",
        xaxis_title="개체",
        yaxis_title=col,
        barmode="group",
        margin=dict(l=50, r=20, t=40, b=40),
        height=300,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
    )
    return fig


def boxplot_two_groups(df_a, df_b, col, name_a, name_b, color_a, color_b):
    a = df_a[col].dropna()
    b = df_b[col].dropna()
    if len(a) < 2 and len(b) < 2:
        return None
    if isinstance(a, pd.DataFrame):
        a = a.iloc[:, 0]
    if isinstance(b, pd.DataFrame):
        b = b.iloc[:, 0]
    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=a,
            name=name_a,
            fillcolor=color_a,
            marker=dict(color=color_a),
            boxpoints="outliers",
            hovertemplate=f"{name_a}<br>{col}: %{{y:.2f}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Box(
            y=b,
            name=name_b,
            fillcolor=color_b,
            marker=dict(color=color_b),
            boxpoints="outliers",
            hovertemplate=f"{name_b}<br>{col}: %{{y:.2f}}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{col} 분포 비교: {name_a} vs {name_b}",
        yaxis_title=col,
        width=520,
        height=340,
        margin=dict(l=50, r=20, t=40, b=40),
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
    )
    return fig


def scatter_two_groups(df_a, df_b, x_col, y_col, name_a, name_b, color_a, color_b, title):
    a = df_a[[x_col, y_col]].dropna()
    b = df_b[[x_col, y_col]].dropna()
    if len(a) < 2 and len(b) < 2:
        return None, None, None
    fig = go.Figure()
    if len(a) >= 2:
        fig.add_trace(
            go.Scatter(
                x=a[x_col],
                y=a[y_col],
                mode="markers",
                marker=dict(size=9, color=color_a, opacity=0.75),
                name=name_a,
                hovertemplate=f"{name_a}<br>{x_col}: %{{x:.2f}}<br>{y_col}: %{{y:.2f}}<extra></extra>",
            )
        )
    if len(b) >= 2:
        fig.add_trace(
            go.Scatter(
                x=b[x_col],
                y=b[y_col],
                mode="markers",
                marker=dict(size=9, color=color_b, opacity=0.75),
                name=name_b,
                hovertemplate=f"{name_b}<br>{x_col}: %{{x:.2f}}<br>{y_col}: %{{y:.2f}}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        width=440,
        height=340,
        margin=dict(l=50, r=20, t=40, b=40),
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    r_a = np.corrcoef(a[x_col], a[y_col])[0, 1] if len(a) >= 2 else None
    r_b = np.corrcoef(b[x_col], b[y_col])[0, 1] if len(b) >= 2 else None
    return fig, r_a, r_b


st.set_page_config(page_title="극지식물 재배 환경 비교 대시보드", layout="wide")

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

st.title("🌱 토양재배와 수경재배 환경이 극지식물의 생육 특성에 어떤 영향을 미칠까?")

with st.sidebar:
    st.header("📂 CSV 업로드")
    env_file = st.sidebar.file_uploader("환경 데이터 CSV (수경재배)", type="csv")
    soil_growth_file = st.sidebar.file_uploader("토양재배 생육 결과 CSV", type="csv")
    water_growth_file = st.sidebar.file_uploader("수경재배 생육 결과 CSV", type="csv")
    st.divider()
    st.info("📎 토양재배 생육 = 강화중 CSV, 수경재배 생육 = 강화여고 CSV 를 업로드하세요.")
    st.info("ℹ️ 환경 CSV는 수경재배 환경의 온도·습도·CO₂ 시계열입니다.")
    st.info("ℹ️ 매칭 파일(토양_수경_환경데이터_매칭.csv)은 기본으로 포함되어 토양·수경 환경 비교에 사용됩니다.")
    st.info("ℹ️ 인코딩(utf-8-sig / cp949)과 열 이름을 자동으로 처리합니다.")

with st.spinner("데이터를 읽어오는 중..."):
    env_df = read_csv(env_file) if env_file is not None else pd.DataFrame()
    soil_df = read_csv(soil_growth_file) if soil_growth_file is not None else pd.DataFrame()
    water_df = read_csv(water_growth_file) if water_growth_file is not None else pd.DataFrame()
    try:
        match_df = std_cols(pd.read_csv("/home/user/uploads/2_______________.csv"))
    except Exception:
        match_df = pd.DataFrame()

if not env_df.empty:
    env_df = std_cols(env_df)
if not soil_df.empty:
    soil_df = std_cols(soil_df)
if not water_df.empty:
    water_df = std_cols(water_df)

tabs = st.tabs([
    "📖 가설 및 주제 소개",
    "🌡️ 환경 변수 비교 및 생육 시각화",
    "🔍 재배 목적에 따른 적합성 검토",
])


with tabs[0]:
    try:
        st.header("연구 주제")
        st.markdown(
            """
극지식물은 저온·건조 등 혹독한 환경에서도 살아남는 식물로, **재배 환경(토양재배 vs 수경재배)**에 따라 생육 특성이 달라질 수 있습니다.<br><br>
이 대시보드는 **강화중(토양재배, 70개체)**과 **강화여고(수경재배, 24개체)**의 생육 데이터를 직접 비교하고, 두 재배 환경의 **온도·습도·pH·EC·CO₂** 차이를 매칭 파일로 살펴봅니다.
"""
        )

        st.header("연구 가설")
        st.markdown(
            """
**가설 (1)**<br>
상대적으로 높은 기온과 제어된 광조건은 식물의 대사 활동을 촉진하여 **잎 수(평균 32.17장)**를 토양재배(평균 21.73장)보다 유의미하게 증가시키는 원인이 될 것이다.<br><br>
**가설 (2)**<br>
높은 상대습도에서는 지하부의 길이가 짧고, 낮은 상대습도에서는 지하부의 길이가 길 것이다.<br><br>
**가설 (3)**<br>
상대습도가 낮은 환경 조건에서 증산 작용을 조절하기 위해 잎이 수경재배 특유의 **두껍고 조밀한 형태적 구조**를 형성할 것이다.
"""
        )

        st.header("데이터 요약")
        rows = []

        if not soil_df.empty and "개체번호" in soil_df.columns:
            s_total = len(soil_df["개체번호"].dropna())
            rows.append({"데이터": "토양재배 생육(강화중)", "개체 수": f"{s_total}개체"})
            if "time" in soil_df.columns:
                rows.append({"데이터": "토양재배 생육", "측정 기간": time_range_str(soil_df["time"])})
            s_cols = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"] if c in soil_df.columns]
            rows.append({"데이터": "토양재배 생육", "측정 항목": ", ".join(s_cols) if s_cols else "없음"})
        else:
            rows.append({"데이터": "토양재배 생육", "상태": "CSV를 업로드해 주세요."})

        if not water_df.empty and "개체번호" in water_df.columns:
            w_total = len(water_df["개체번호"].dropna())
            rows.append({"데이터": "수경재배 생육(강화여고)", "개체 수": f"{w_total}개체"})
            if "time" in water_df.columns:
                rows.append({"데이터": "수경재배 생육", "측정 기간": time_range_str(water_df["time"])})
            w_cols = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"] if c in water_df.columns]
            rows.append({"데이터": "수경재배 생육", "측정 항목": ", ".join(w_cols) if w_cols else "없음"})
        else:
            rows.append({"데이터": "수경재배 생육", "상태": "CSV를 업로드해 주세요."})

        if not env_df.empty and "time" in env_df.columns:
            rows.append({"데이터": "수경재배 환경(강화여고)", "측정 기간": time_range_str(env_df["time"])})
            e_cols = [c for c in ["temperature", "humidity", "ph", "ec", "co2"] if c in env_df.columns]
            rows.append({"데이터": "수경재배 환경", "측정 항목": ", ".join(e_cols) if e_cols else "없음"})
        else:
            rows.append({"데이터": "수경재배 환경", "상태": "환경 CSV를 업로드해 주세요."})

        if not match_df.empty:
            rows.append({"데이터": "매칭 파일(토양·수경 비교)", "행 수": f"{len(match_df)}행"})
        else:
            rows.append({"데이터": "매칭 파일", "상태": "읽을 수 없어요."})

        st.dataframe(pd.DataFrame(rows), width="stretch")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


with tabs[1]:
    try:
        st.header("1. 수경재배 환경 데이터 (강화여고 환경 CSV)")
        if env_df.empty:
            st.info("📎 환경 CSV(수경재배)를 업로드해 주세요.")
        else:
            env_num_cols = [c for c in ["temperature", "humidity", "ph", "ec", "co2"] if c in env_df.columns]
            if env_num_cols:
                env_summary = env_df[env_num_cols].describe().loc[["mean", "min", "max"]].T
                env_summary.columns = ["평균", "최소", "최대"]
                st.dataframe(env_summary, width="stretch")
            else:
                st.info("표시할 숫자 환경 열이 없어요.")

            if "time" in env_df.columns and "temperature" in env_df.columns:
                t_col = env_df["time"]
                temp_col = env_df["temperature"]
                if isinstance(t_col, pd.DataFrame):
                    t_col = t_col.iloc[:, 0]
                if isinstance(temp_col, pd.DataFrame):
                    temp_col = temp_col.iloc[:, 0]
                daily = (
                    pd.DataFrame({"time": t_col, "temperature": temp_col, "humidity": env_df["humidity"]})
                    .dropna(subset=["time", "temperature"])
                    .copy()
                )
                daily["date"] = daily["time"].dt.date
                if "humidity" in env_df.columns:
                    hum_col = env_df["humidity"]
                    if isinstance(hum_col, pd.DataFrame):
                        hum_col = hum_col.iloc[:, 0]
                    daily["humidity"] = hum_col
                daily = daily.groupby("date").agg({"temperature": "mean", "humidity": "mean"}).reset_index()
                if len(daily) >= 2:
                    temp_t = go.Scatter(
                        x=daily["date"],
                        y=daily["temperature"],
                        mode="lines",
                        name="온도",
                        line=dict(color="#e74c3c", width=2),
                        hovertemplate="%{x|%Y-%m-%d}<br>온도: %{y:.1f} °C<extra></extra>",
                    )
                    hum_t = go.Scatter(
                        x=daily["date"],
                        y=daily["humidity"],
                        mode="lines",
                        name="습도",
                        line=dict(color="#3498db", width=2),
                        hovertemplate="%{x|%Y-%m-%d}<br>습도: %{y:.1f} %%<extra></extra>",
                        yaxis="y2",
                    )
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=("일별 평균 온도", "일별 평균 습도"),
                        specs=[[{}, {"secondary_y": True}]],
                    )
                    fig.add_trace(temp_t, row=1, col=1)
                    fig.add_trace(hum_t, row=1, col=2, secondary_y=True)
                    fig.update_xaxes(tickformat="%Y-%m-%d", tickangle=45, row=1, col=1)
                    fig.update_xaxes(tickformat="%Y-%m-%d", tickangle=45, row=1, col=2)
                    fig.update_yaxes(title_text="온도 (°C)", row=1, col=1)
                    fig.update_yaxes(title_text="습도 (%)", row=1, col=2)
                    fig.update_layout(
                        margin=dict(l=50, r=30, t=40, b=40),
                        height=320,
                        hovermode="x unified",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    )
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("날짜별 데이터를 묶기에 측정값이 너무 적어요.")
            else:
                st.info("환경 데이터에 'time'과 'temperature' 열이 필요해요.")

        st.divider()
        st.header("2. 토양재배 vs 수경재배 환경 비교 (매칭 파일)")
        if not match_df.empty:
            s_env_cols = [c for c in ["강화중_temperature", "강화중_humidity", "강화중_ph", "강화중_ec"] if c in match_df.columns]
            w_env_cols = [c for c in ["강화여고_temperature", "강화여고_humidity", "강화여고_co2_참고용"] if c in match_df.columns]
            if s_env_cols or w_env_cols:
                rows_m = []
                if "강화중_temperature" in match_df.columns:
                    t_mean = match_df["강화중_temperature"].mean()
                    if isinstance(match_df["강화중_temperature"], pd.DataFrame):
                        t_mean = match_df["강화중_temperature"].iloc[:, 0].mean()
                    rows_m.append({"재배 환경": "토양재배(추정)", "온도(°C)": f"{t_mean:.1f}"})
                    if "강화중_humidity" in match_df.columns:
                        h_mean = match_df["강화중_humidity"].mean()
                        if isinstance(match_df["강화중_humidity"], pd.DataFrame):
                            h_mean = match_df["강화중_humidity"].iloc[:, 0].mean()
                        rows_m.append({"재배 환경": "토양재배(추정)", "습도(%)": f"{h_mean:.1f}"})
                if "강화중_ph" in match_df.columns and "강화중_ec" in match_df.columns:
                    ph_mean = match_df["강화중_ph"].mean()
                    ec_mean = match_df["강화중_ec"].mean()
                    if isinstance(match_df["강화중_ph"], pd.DataFrame):
                        ph_mean = match_df["강화중_ph"].iloc[:, 0].mean()
                    if isinstance(match_df["강화중_ec"], pd.DataFrame):
                        ec_mean = match_df["강화중_ec"].iloc[:, 0].mean()
                    rows_m.append({"재배 환경": "토양재배(추정)", "pH": f"{ph_mean:.2f}", "EC": f"{ec_mean:.2f}"})
                if "강화여고_temperature" in match_df.columns:
                    t_mean_w = match_df["강화여고_temperature"].mean()
                    if isinstance(match_df["강화여고_temperature"], pd.DataFrame):
                        t_mean_w = match_df["강화여고_temperature"].iloc[:, 0].mean()
                    rows_m.append({"재배 환경": "수경재배(추정)", "온도(°C)": f"{t_mean_w:.1f}"})

                if "강화여고_humidity" in match_df.columns:
                    h_mean_w = match_df["강화여고_humidity"].mean()
                    if isinstance(match_df["강화여고_humidity"], pd.DataFrame):
                        h_mean_w = match_df["강화여고_humidity"].iloc[:, 0].mean()
                    rows_m.append({"재배 환경": "수경재배(추정)", "습도(%)": f"{h_mean_w:.1f}"})
                if "강화여고_co2_참고용" in match_df.columns:
                    co2_mean = match_df["강화여고_co2_참고용"].mean()
                    if isinstance(match_df["강화여고_co2_참고용"], pd.DataFrame):
                        co2_mean = match_df["강화여고_co2_참고용"].iloc[:, 0].mean()
                    rows_m.append({"재배 환경": "수경재배(추정)", "CO₂(ppm)": f"{co2_mean:.0f}"})
                st.dataframe(pd.DataFrame(rows_m), width="stretch")
            else:
                st.info("매칭 파일의 환경 열을 확인할 수 없어요.")

            def get_series(df, col):
                s = df[col]
                return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s

            if "강화중_temperature" in match_df.columns and "강화중_humidity" in match_df.columns:
                s_t = get_series(match_df, "강화중_temperature")
                s_h = get_series(match_df, "강화중_humidity")
                s_data = pd.DataFrame({"t": s_t, "h": s_h}).dropna()
                if len(s_data) >= 2:
                    fig_s = go.Figure()
                    fig_s.add_trace(
                        go.Scatter(
                            x=s_data["t"], y=s_data["h"],
                            mode="markers",
                            marker=dict(size=10, color="#8e44ad", opacity=0.75),
                            name="토양재배(추정)",
                            hovertemplate="토양재배(추정)<br>온도: %{x:.1f}°C<br>습도: %{y:.1f}%<extra></extra>",
                        )
                    )
                    fig_s.update_layout(
                        title="토양재배(추정) 환경 산점도 - 온도 vs 습도",
                        xaxis_title="온도 (°C)", yaxis_title="습도 (%)",
                        width=440, height=340,
                        margin=dict(l=50, r=20, t=40, b=40),
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    )
                    st.plotly_chart(fig_s, width="stretch")

            if "강화여고_temperature" in match_df.columns and "강화여고_humidity" in match_df.columns:
                w_t = get_series(match_df, "강화여고_temperature")
                w_h = get_series(match_df, "강화여고_humidity")
                w_data = pd.DataFrame({"t": w_t, "h": w_h}).dropna()
                if len(w_data) >= 2:
                    fig_w = go.Figure()
                    fig_w.add_trace(
                        go.Scatter(
                            x=w_data["t"], y=w_data["h"],
                            mode="markers",
                            marker=dict(size=10, color="#16a085", opacity=0.75),
                            name="수경재배(추정)",
                            hovertemplate="수경재배(추정)<br>온도: %{x:.1f}°C<br>습도: %{y:.1f}%<extra></extra>",
                        )
                    )
                    fig_w.update_layout(
                        title="수경재배(추정) 환경 산점도 - 온도 vs 습도",
                        xaxis_title="온도 (°C)", yaxis_title="습도 (%)",
                        width=440, height=340,
                        margin=dict(l=50, r=20, t=40, b=40),
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    )
                    st.plotly_chart(fig_w, width="stretch")

            if (
                "강화중_temperature" in match_df.columns and "강화중_humidity" in match_df.columns
                and "강화여고_temperature" in match_df.columns and "강화여고_humidity" in match_df.columns
            ):
                s2_t = get_series(match_df, "강화중_temperature")
                s2_h = get_series(match_df, "강화중_humidity")
                w2_t = get_series(match_df, "강화여고_temperature")
                w2_h = get_series(match_df, "강화여고_humidity")
                s2 = pd.DataFrame({"t": s2_t, "h": s2_h}).dropna()
                w2 = pd.DataFrame({"t": w2_t, "h": w2_h}).dropna()
                if len(s2) >= 2 and len(w2) >= 2:
                    combined = pd.DataFrame({
                        "temperature": pd.concat([s2["t"], w2["t"]]),
                        "humidity": pd.concat([s2["h"], w2["h"]]),
                        "재배환경": ["토양재배(추정)"] * len(s2) + ["수경재배(추정)"] * len(w2),
                    })
                    fig_cmp = go.Figure()
                    mask_s = combined["재배환경"] == "토양재배(추정)"
                    mask_w = combined["재배환경"] == "수경재배(추정)"
                    fig_cmp.add_trace(
                        go.Scatter(
                            x=combined.loc[mask_s, "temperature"],
                            y=combined.loc[mask_s, "humidity"],
                            mode="markers",
                            marker=dict(size=10, color="#8e44ad", opacity=0.75),
                            name="토양재배(추정)",
                            hovertemplate="토양재배(추정)<br>온도: %{x:.1f}°C<br>습도: %{y:.1f}%<extra></extra>",
                        )
                    )
                    fig_cmp.add_trace(
                        go.Scatter(
                            x=combined.loc[mask_w, "temperature"],
                            y=combined.loc[mask_w, "humidity"],
                            mode="markers",
                            marker=dict(size=10, color="#16a085", opacity=0.75),
                            name="수경재배(추정)",
                            hovertemplate="수경재배(추정)<br>온도: %{x:.1f}°C<br>습도: %{y:.1f}%<extra></extra>",
                        )
                    )
                    fig_cmp.update_layout(
                        title="토양재배 vs 수경재배 환경 산점도 (온도 vs 습도)",
                        xaxis_title="온도 (°C)", yaxis_title="습도 (%)",
                        width=460, height=340,
                        margin=dict(l=50, r=20, t=40, b=40),
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    )
                    st.plotly_chart(fig_cmp, width="stretch")
            else:
                st.info("토양·수경 양쪽 온도·습도 열이 모두 필요해요.")
        else:
            st.info("매칭 파일을 읽을 수 없어요.")

        st.divider()
        st.header("3. 생육 결과 비교: 토양재배 vs 수경재배")
        both_ready = not soil_df.empty and not water_df.empty
        if not both_ready:
            if soil_df.empty and water_df.empty:
                st.info("📎 토양재배 생육 CSV와 수경재배 생육 CSV를 모두 업로드해 주세요.")
            elif soil_df.empty:
                st.info("📎 토양재배 생육 CSV(강화중)를 업로드해 주세요.")
            else:
                st.info("📎 수경재배 생육 CSV(강화여고)를 업로드해 주세요.")
        else:
            if "개체번호" not in soil_df.columns or "개체번호" not in water_df.columns:
                st.info("두 생육 CSV 모두 '개체번호' 열이 필요해요.")
            else:
                def safe_col(df, col):
                    s = df[col]
                    return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s

                s_dfs = {}
                w_dfs = {}
                for col in ["잎 수", "지상부 길이", "지하부 길이", "생중량"]:
                    if col in soil_df.columns:
                        s_dfs[col] = safe_col(soil_df, col)
                    if col in water_df.columns:
                        w_dfs[col] = safe_col(water_df, col)

                st.subheader("생육 요약 비교 표")
                rows_g = []
                for col in ["잎 수", "지상부 길이", "지하부 길이", "생중량"]:
                    if col in s_dfs and col in w_dfs:
                        s = s_dfs[col].dropna()
                        w = w_dfs[col].dropna()
                        if len(s) >= 2 and len(w) >= 2:
                            rows_g.append({
                                "측정 항목": col,
                                "토양재배 평균": f"{s.mean():.2f}",
                                "토양재배 최소~최대": f"{s.min():.2f} ~ {s.max():.2f}",
                                "수경재배 평균": f"{w.mean():.2f}",
                                "수경재배 최소~최대": f"{w.min():.2f} ~ {w.max():.2f}",
                            })
                if rows_g:
                    st.dataframe(pd.DataFrame(rows_g), width="stretch")
                else:
                    st.info("두 집단에서 공통으로 존재하는 생육 측정 항목이 없어요.")

                st.subheader("개체별 생육 비교 (막대 그래프)")
                for col in ["잎 수", "지상부 길이", "지하부 길이", "생중량"]:
                    if col in s_dfs and col in w_dfs:
                        a = pd.DataFrame({"id": soil_df["개체번호"].astype(str), col: s_dfs[col]}).dropna()
                        b = pd.DataFrame({"id": water_df["개체번호"].astype(str), col: w_dfs[col]}).dropna()
                        if len(a) >= 2 or len(b) >= 2:
                            fig = go.Figure()
                            if len(a) >= 2:
                                fig.add_trace(
                                    go.Bar(
                                        x=a["id"], y=a[col],
                                        name="토양재배(강화중)",
                                        marker_color="#8e44ad",
                                        hovertemplate="%{x}<br>%{y:.2f}<extra></extra>",
                                    )
                                )
                            if len(b) >= 2:
                                fig.add_trace(
                                    go.Bar(
                                        x=b["id"], y=b[col],
                                        name="수경재배(강화여고)",
                                        marker_color="#16a085",
                                        hovertemplate="%{x}<br>%{y:.2f}<extra></extra>",
                                    )
                                )
                            fig.update_layout(
                                title=f"{col} 비교: 토양재배 vs 수경재배",
                                xaxis_title="개체", yaxis_title=col,
                                barmode="group",
                                margin=dict(l=50, r=20, t=40, b=40),
                                height=300,
                                font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            )
                            st.plotly_chart(fig, width="stretch")
                    else:
                        st.info(f"{col} 열이 두 CSV 모두에 필요해요.")

                st.divider()
                st.subheader("생육 분포 비교 (박스플롯)")
                for col in ["잎 수", "지상부 길이", "지하부 길이", "생중량"]:
                    if col in s_dfs and col in w_dfs:
                        s = s_dfs[col].dropna()
                        w = w_dfs[col].dropna()
                        if len(s) >= 2 or len(w) >= 2:
                            fig = go.Figure()
                            if len(s) >= 2:
                                fig.add_trace(
                                    go.Box(
                                        y=s, name="토양재배(강화중)",
                                        fillcolor="#8e44ad", marker=dict(color="#8e44ad"),
                                        boxpoints="outliers",
                                        hovertemplate=f"토양재배(강화중)<br>{col}: %{{y:.2f}}<extra></extra>",
                                    )
                                )
                            if len(w) >= 2:
                                fig.add_trace(
                                    go.Box(
                                        y=w, name="수경재배(강화여고)",
                                        fillcolor="#16a085", marker=dict(color="#16a085"),
                                        boxpoints="outliers",
                                        hovertemplate=f"수경재배(강화여고)<br>{col}: %{{y:.2f}}<extra></extra>",
                                    )
                                )
                            fig.update_layout(
                                title=f"{col} 분포 비교: 토양재배 vs 수경재배",
                                yaxis_title=col,
                                width=520, height=340,
                                margin=dict(l=50, r=20, t=40, b=40),
                                font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            )
                            st.plotly_chart(fig, width="stretch")
                    else:
                        st.info(f"{col} 열이 두 CSV 모두에 필요해요.")

                st.divider()
                st.subheader("상관관계 비교 (산점도 + 추세선)")
                for x_col, y_col, title in [
                    ("잎 수", "생중량", "잎 수 vs 생중량"),
                    ("지상부 길이", "생중량", "지상부 길이 vs 생중량"),
                    ("잎 수", "지상부 길이", "잎 수 vs 지상부 길이"),
                ]:
                    if x_col in s_dfs and y_col in s_dfs and x_col in w_dfs and y_col in w_dfs:
                        s_pair = pd.DataFrame({"x": s_dfs[x_col], "y": s_dfs[y_col]}).dropna()
                        w_pair = pd.DataFrame({"x": w_dfs[x_col], "y": w_dfs[y_col]}).dropna()
                        if len(s_pair) >= 2 or len(w_pair) >= 2:
                            fig = go.Figure()
                            if len(s_pair) >= 2:
                                fig.add_trace(
                                    go.Scatter(
                                        x=s_pair["x"], y=s_pair["y"],
                                        mode="markers",
                                        marker=dict(size=9, color="#8e44ad", opacity=0.75),
                                        name="토양재배(강화중)",
                                        hovertemplate=f"토양재배(강화중)<br>{x_col}: %{{x:.2f}}<br>{y_col}: %{{y:.2f}}<extra></extra>",
                                    )
                                )
                                m_s, b_s = np.polyfit(s_pair["x"], s_pair["y"], 1)
                                fig.add_trace(
                                    go.Scatter(
                                        x=np.sort(s_pair["x"]), y=m_s*np.sort(s_pair["x"])+b_s,
                                        mode="lines",
                                        line=dict(color="#8e44ad", width=2, dash="dot"),
                                        name=f"토양재배 추세선",
                                    )
                                )
                            if len(w_pair) >= 2:
                                fig.add_trace(
                                    go.Scatter(
                                        x=w_pair["x"], y=w_pair["y"],
                                        mode="markers",
                                        marker=dict(size=9, color="#16a085", opacity=0.75),
                                        name="수경재배(강화여고)",
                                        hovertemplate=f"수경재배(강화여고)<br>{x_col}: %{{x:.2f}}<br>{y_col}: %{{y:.2f}}<extra></extra>",
                                    )
                                )
                                m_w, b_w = np.polyfit(w_pair["x"], w_pair["y"], 1)
                                fig.add_trace(
                                    go.Scatter(
                                        x=np.sort(w_pair["x"]), y=m_w*np.sort(w_pair["x"])+b_w,
                                        mode="lines",
                                        line=dict(color="#16a085", width=2, dash="dot"),
                                        name=f"수경재배 추세선",
                                    )
                                )
                            fig.update_layout(
                                title=f"{title} - 토양재배 vs 수경재배",
                                xaxis_title=x_col, yaxis_title=y_col,
                                width=460, height=340,
                                margin=dict(l=50, r=20, t=40, b=40),
                                font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                            )
                            st.plotly_chart(fig, width="stretch")
                            r_s = np.corrcoef(s_pair["x"], s_pair["y"])[0, 1] if len(s_pair) >= 2 else None
                            r_w = np.corrcoef(w_pair["x"], w_pair["y"])[0, 1] if len(w_pair) >= 2 else None
                            r_s_str = f"{r_s:.2f}" if r_s is not None else "N/A"
                            r_w_str = f"{r_w:.2f}" if r_w is not None else "N/A"
                            st.caption(f"토양재배 r={r_s_str}, 수경재배 r={r_w_str} (1에 가까울수록 함께 증가)")
                    else:
                        st.info(f"{x_col}·{y_col} 열이 두 CSV 모두에 필요해요.")

        st.divider()
        gd_exp = st.expander("📄 생육 데이터 원본 + CSV 다운로드")
        with gd_exp:
            if not soil_df.empty:
                st.dataframe(soil_df, width="stretch")
                st.download_button(
                    "토양재배 생육 CSV 다운로드",
                    data=convert_csv(soil_df),
                    file_name="강화중_토양재배_생육결과.csv",
                    mime="text/csv",
                )
            if not water_df.empty:
                st.dataframe(water_df, width="stretch")
                st.download_button(
                    "수경재배 생육 CSV 다운로드",
                    data=convert_csv(water_df),
                    file_name="강화여고_수경재배_생육결과.csv",
                    mime="text/csv",
                )
            if soil_df.empty and water_df.empty:
                st.info("생육 데이터가 없어요.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


with tabs[2]:
    try:
        st.header("재배 목적에 따른 토양·수경재배 적합성 검토")

        st.subheader("가설 (1) 검증: 높은 기온·제어된 광조건 → 잎 수 증가")
        if (
            not soil_df.empty and "잎 수" in soil_df.columns
            and not water_df.empty and "잎 수" in water_df.columns
        ):
            s_leaf = soil_df["잎 수"].dropna()
            w_leaf = water_df["잎 수"].dropna()
            if isinstance(s_leaf, pd.DataFrame):
                s_leaf = s_leaf.iloc[:, 0]
            if isinstance(w_leaf, pd.DataFrame):
                w_leaf = w_leaf.iloc[:, 0]
            if len(s_leaf) >= 2 and len(w_leaf) >= 2:
                s_mean = s_leaf.mean()
                w_mean = w_leaf.mean()
                diff = w_mean - s_mean
                st.markdown(
                    f"""
- 토양재배(강화중, {len(s_leaf)}개체) 잎 수 평균: **{s_mean:.2f}장** (범위 {s_leaf.min():.0f}–{s_leaf.max():.0f}장)
- 수경재배(강화여고, {len(w_leaf)}개체) 잎 수 평균: **{w_mean:.2f}장** (범위 {w_leaf.min():.0f}–{w_leaf.max():.0f}장)
- 두 집단 간 평균 차이: **{diff:+.2f}장** ({'수경재배가 더 높음' if diff > 0 else '토양재배가 더 높음'})
- 문서에서 제시된 수경재배 잎 수 평균 32.17장과 실제 측정값({w_mean:.2f}장)이 {"일치" if abs(w_mean - 32.17) < 1 else "근사"}합니다.
- 문서 제시 토양재배 잎 수 평균 21.73장과 실제 측정값({s_mean:.2f}장)을 비교하면, 가설 (1)의 방향성(수경재배 잎 수가 더 많음)은 {"일치" if diff > 0 else "반대"}합니다.
"""
                )
            else:
                st.info("잎 수 데이터가 충분하지 않아요.")
        else:
            st.info("두 생육 CSV를 모두 업로드하고 잎 수 열이 있어야 가설 (1)을 검토할 수 있어요.")

        st.subheader("가설 (2) 검증: 상대습도 ↔ 지하부 길이")
        if (
            not soil_df.empty and "지하부 길이" in soil_df.columns
            and not water_df.empty and "지하부 길이" in water_df.columns
        ):
            s_root = soil_df["지하부 길이"].dropna()
            w_root = water_df["지하부 길이"].dropna()
            if isinstance(s_root, pd.DataFrame):
                s_root = s_root.iloc[:, 0]
            if isinstance(w_root, pd.DataFrame):
                w_root = w_root.iloc[:, 0]
            if len(s_root) >= 2 and len(w_root) >= 2:
                env_hum = None
                if not env_df.empty and "humidity" in env_df.columns:
                    env_hum = env_df["humidity"].dropna().mean()
                    if isinstance(env_df["humidity"], pd.DataFrame):
                        env_hum = env_df["humidity"].iloc[:, 0].dropna().mean()
                s_hum_match = None
                w_hum_match = None
                if not match_df.empty:
                    if "강화중_humidity" in match_df.columns:
                        s_hum_match = match_df["강화중_humidity"].dropna().mean()
                        if isinstance(match_df["강화중_humidity"], pd.DataFrame):
                            s_hum_match = match_df["강화중_humidity"].iloc[:, 0].dropna().mean()
                    if "강화여고_humidity" in match_df.columns:
                        w_hum_match = match_df["강화여고_humidity"].dropna().mean()
                        if isinstance(match_df["강화여고_humidity"], pd.DataFrame):
                            w_hum_match = match_df["강화여고_humidity"].iloc[:, 0].dropna().mean()
                s_root_mean = s_root.mean()
                w_root_mean = w_root.mean()
                root_diff = w_root_mean - s_root_mean
                st.markdown(
                    f"""
- 토양재배(추정) 습도 환경(매칭 파일): **{s_hum_match:.1f}%** (평균) — 높은 습도 환경
- 수경재배 환경(강화여고 환경 CSV): **{env_hum:.1f}%** (평균) — 상대적으로 낮은 습도 환경
- 토양재배(강화중, {len(s_root)}개체) 지하부 길이 평균: **{s_root_mean:.2f}mm** (범위 {s_root.min():.0f}–{s_root.max():.0f}mm)
- 수경재배(강화여고, {len(w_root)}개체) 지하부 길이 평균: **{w_root_mean:.2f}mm** (범위 {w_root.min():.0f}–{w_root.max():.0f}mm)
- 두 집단 간 지하부 길이 평균 차이: **{root_diff:+.2f}mm** ({'수경재배가 더 김' if root_diff > 0 else '토양재배가 더 김'})
- 가설 (2): "높은 습도 → 짧은 지하부, 낮은 습도 → 긴 지하부"
  - 실제 데이터에서 {'수경재배(낮은 습도)의 지하부가 더 길게 나타나 가설과 일치' if (root_diff > 0 and env_hum is not None and s_hum_match is not None and env_hum < s_hum_match) else '수경재배(낮은 습도)의 지하부가 더 짧게 나타나 가설과 불일치' if (root_diff <= 0 and env_hum is not None and s_hum_match is not None and env_hum < s_hum_match) else '습도 조건 직접 비교를 위해 매칭 파일의 습도 데이터 확인이 필요'}
"""
                )
            else:
                st.info("지하부 길이 데이터가 충분하지 않아요.")
        else:
            st.info("두 생육 CSV를 모두 업로드하고 지하부 길이 열이 있어야 가설 (2)을 검토할 수 있어요.")

        st.subheader("가설 (3) 검증: 저습도 조건에서 잎이 두껍고 조밀한 구조 형성")
        st.markdown(
            """
- 현재 데이터에는 잎의 **두께·밀도·단위면적당 질량(LMA)** 등 형태적 구조 측정값이 포함되어 있지 않아, 가설 (3)을 데이터로 직접 검증할 수 없습니다.
- 수경재배 환경(강화여고, 약 56–61% 습도)에서 자란 개체의 잎이 형태적으로 두꺼운지는 **현미경 관찰·잎 두께 측정·LMA 측정** 등 추가 실험이 필요합니다.
- 문서에서 제시된 "수경재배 특유의 두껍고 조밀한 형태적 구조"는 본 대시보드의 수치 데이터만으로는 확인이 어려우며, 정성적 관찰 결과로 이해해야 합니다.
"""
        )

        st.divider()
        st.header("종합: 재배 목적에 따른 적합성")
        if (
            not soil_df.empty and "잎 수" in soil_df.columns and "생중량" in soil_df.columns
            and not water_df.empty and "잎 수" in water_df.columns and "생중량" in water_df.columns
        ):
            s_cols_list = ["잎 수", "지상부 길이", "지하부 길이", "생중량"]
            rows_sum = []
            for col in s_cols_list:
                if col in soil_df.columns and col in water_df.columns:
                    s = safe_col(soil_df, col).dropna()
                    w = safe_col(water_df, col).dropna()
                    if len(s) >= 2 and len(w) >= 2:
                        rows_sum.append({
                            "재배 방식": "토양재배(강화중)",
                            "측정 항목": col,
                            "평균": f"{s.mean():.2f}",
                            "표준편차": f"{s.std():.2f}",
                            "CV": f"{s.std()/s.mean()*100:.1f}%",
                        })
                        rows_sum.append({
                            "재배 방식": "수경재배(강화여고)",
                            "측정 항목": col,
                            "평균": f"{w.mean():.2f}",
                            "표준편차": f"{w.std():.2f}",
                            "CV": f"{w.std()/w.mean()*100:.1f}%",
                        })
            if rows_sum:
                st.subheader("두 재배 방식 생육 지표 종합 비교")
                st.dataframe(pd.DataFrame(rows_sum), width="stretch")

        st.markdown(
            """
**수경재배가 유리한 목적**
- 잎 수 확보가 중요한 경우: 수경재배 환경에서 잎 수 평균이 높게 나타났는지 확인 (가설 1 방향)
- 생육 개체 간 편차가 작고 일정한 품질을 원하는 경우: 제어된 환경의 이점 활용

**토양재배가 유리한 목적**
- 자연 상태에 가까운 조건에서 스트레스 내성 개체를 확보하려는 경우
- 지하부 길이가 긴 개체가 필요한 경우 (가설 2 방향 확인 후 판단)
- 본 대시보드의 두 집단 비교 결과를 바탕으로 목적에 맞는 재배 방식 선택

**데이터 한계**
- 두 생육 데이터가 서로 다른 시기·개체 수로 측정되어 통계 검정은 제한적
- 가설 (3) 검증을 위해서는 잎 두께·LMA 등 형태 측정 데이터가 별도로 필요
- 환경 데이터(수경재배 환경 CSV)와 매칭 파일(토양·수경 비교)은 서로 다른 기간의 데이터
"""
        )

        st.divider()
        env_exp = st.expander("📄 환경 데이터 원본 + CSV 다운로드")
        with env_exp:
            if not env_df.empty:
                st.dataframe(env_df, width="stretch")
                st.download_button(
                    "환경 CSV 다운로드",
                    data=convert_csv(env_df),
                    file_name="강화여고_환경.csv",
                    mime="text/csv",
                )
            else:
                st.info("환경 데이터가 없어요.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
