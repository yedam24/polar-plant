import io
import os
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st


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
    s = series.dropna()
    if len(s) == 0:
        return ""
    return f"{s.min().strftime('%Y-%m-%d')} ~ {s.max().strftime('%Y-%m-%d')}"


def growth_bar(df, col, title, color):
    data = df[["개체번호", col]].dropna()
    if len(data) < 2:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=data["개체번호"].astype(str),
            y=data[col],
            name=title,
            marker_color=color,
            hovertemplate="%{x}<br>%{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="개체",
        yaxis_title=col,
        margin=dict(l=40, r=20, t=40, b=40),
        height=280,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
    )
    return fig


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


st.set_page_config(page_title="극지식물 생육 대시보드", layout="wide")

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


st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")

with st.sidebar:
    st.header("📂 CSV 업로드")
    env_file = st.sidebar.file_uploader("환경 데이터 CSV", type="csv")
    growth_file = st.sidebar.file_uploader("생육 결과 데이터 CSV", type="csv")
    st.divider()
    measure = st.multiselect(
        "생육 측정 항목 (그래프에 표시)",
        ["잎 수", "지상부 길이", "지하부 길이"],
        default=["잎 수", "지상부 길이", "지하부 길이"],
        help="복수 선택 가능. 그래프에 가로 막대 그래프로 나란히 표시됩니다.",
    )

    st.divider()
    st.info("📎 CSV를 업로드하면 각 탭에 분석 결과가 표시됩니다.")
    st.info("ℹ️ 인코딩(utf-8-sig / cp949)과 열 이름을 자동으로 처리합니다.")

with st.spinner("데이터를 읽어오는 중..."):
    env_df = read_csv(env_file) if env_file is not None else pd.DataFrame()
    growth_df = read_csv(growth_file) if growth_file is not None else pd.DataFrame()

school_name = ""
if growth_file is not None:
    base = os.path.basename(growth_file.name)
    school_name = base.replace(".csv", "").split("_")[0]

if not env_df.empty:
    env_df = std_cols(env_df)
if not growth_df.empty:
    growth_df = std_cols(growth_df)

tabs = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])


with tabs[0]:
    try:
        if env_df.empty and growth_df.empty:
            st.info("📎 사이드바에서 CSV 파일을 업로드해 주세요.")
        else:
            st.header("연구 배경 및 목적")
            st.markdown(
                """
- 극지식물은 저온·건조 등 혹독한 환경에서도 살아남는 식물로, 같은 조건에서도 **개체마다 생육 차이**가 나타납니다.
- 이 대시보드는 우리 학교가 측정한 **환경 데이터**(온도·습도·pH·EC·CO₂ 등)와 **개체별 생육 결과**(잎 수·길이·질량)를 연결해, 어떤 개체가 잘 자랐는지 비교합니다.
- 평균·표준편차·변동계수(CV)·상관계수 등 통계량을 함께 확인해, 단순한 숫자 차이를 넘어선 해석을 돕습니다.
"""
            )

            st.header("데이터 요약")
            rows = []
            if not env_df.empty and "time" in env_df.columns:
                rows.append(
                    {
                        "항목": "환경 데이터",
                        "내용": f"측정 기간: {time_range_str(env_df['time'])}",
                    }
                )
            if not env_df.empty:
                env_cols = [
                    c for c in ["temperature", "humidity", "ph", "ec", "co2"]
                    if c in env_df.columns
                ]
                rows.append(
                    {
                        "항목": "환경 측정 항목",
                        "내용": ", ".join(env_cols)
                        if env_cols
                        else "없음",
                    }
                )
            if not growth_df.empty:
                g_cols = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"] if c in growth_df.columns]
                rows.append(
                    {
                        "항목": "생육 측정 항목",
                        "내용": ", ".join(g_cols) if g_cols else "없음",
                    }
                )
            if not growth_df.empty and "개체번호" in growth_df.columns:
                rows.append(
                    {"항목": "총 개체 수", "내용": str(len(growth_df["개체번호"].dropna()))}
                )
            if rows:
                st.dataframe(pd.DataFrame(rows), width="stretch")
            else:
                st.info("데이터가 충분하지 않아 요약 표를 표시할 수 없어요.")

            total = 0
            avg_temp = None
            avg_hum = None
            avg_fw = None

            if not growth_df.empty and "개체번호" in growth_df.columns:
                total = len(growth_df["개체번호"].dropna())

            if not env_df.empty and "temperature" in env_df.columns:
                t = env_df["temperature"].dropna()
                if len(t) >= 2:
                    avg_temp = t.mean()

            if not env_df.empty and "humidity" in env_df.columns:
                h = env_df["humidity"].dropna()
                if len(h) >= 2:
                    avg_hum = h.mean()

            if not growth_df.empty and "생중량" in growth_df.columns:
                fw = growth_df["생중량"].dropna()
                if len(fw) >= 2:
                    avg_fw = fw.mean()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("총 개체수", f"{total}개" if total else "—")
            if avg_temp is not None:
                m2.metric("평균 온도", f"{avg_temp:.1f} °C")
            else:
                m2.metric("평균 온도", "—")
            if avg_hum is not None:
                m3.metric("평균 습도", f"{avg_hum:.1f} %")
            else:
                m3.metric("평균 습도", "—")
            if avg_fw is not None:
                m4.metric("평균 생중량", f"{avg_fw:.2f} g")
            else:
                m4.metric("평균 생중량", "—")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


with tabs[1]:
    try:
        if env_df.empty:
            st.info("📎 환경 CSV를 업로드해 주세요. 이 탭은 환경 데이터가 있어야 표시됩니다.")
        else:
            st.header("환경 데이터 요약")
            env_num_cols = [
                c for c in ["temperature", "humidity", "ph", "ec", "co2"]
                if c in env_df.columns
            ]
            if env_num_cols:
                env_summary = env_df[env_num_cols].describe().loc[["mean", "min", "max"]].T
                env_summary.columns = ["평균", "최소", "최대"]
                st.dataframe(env_summary, width="stretch")
            else:
                st.info("표시할 숫자 환경 열이 없어요. CSV 열 이름(time, temperature, humidity 등)을 확인해 주세요.")

            st.divider()
            st.header("날짜별 시계열")

            if "temperature" in env_df.columns and "time" in env_df.columns:
                daily = (
                    env_df[["time", "temperature", "humidity"]]
                    .dropna(subset=["time", "temperature"])
                    .copy()
                )
                daily["date"] = daily["time"].dt.date
                daily = daily.groupby("date").agg({"temperature": "mean", "humidity": "mean"}).reset_index()

                if len(daily) >= 2:
                    temp_t = go.Scatter(
                        x=daily["date"],
                        y=daily["temperature"],
                        mode="lines",
                        name="온도",
                        line=dict(color="#e74c3c", width=2),
                        hovertemplate="%(date)s<br>온도: %(y).1f °C<extra></extra>",
                    )
                    hum_t = go.Scatter(
                        x=daily["date"],
                        y=daily["humidity"],
                        mode="lines",
                        name="습도",
                        line=dict(color="#3498db", width=2),
                        hovertemplate="%(date)s<br>습도: %(y).1f %%<extra></extra>",
                        yaxis="y2",
                    )
                    fig = make_subplots(
                        rows=1,
                        cols=2,
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
                st.info("환경 데이터에 'time'과 'temperature' 열이 필요해요. CSV 열 이름을 확인해 주세요.")

            st.divider()
            env_exp = st.expander("📄 환경 데이터 원본 테이블 + CSV 다운로드")
            with env_exp:
                if not env_df.empty:
                    st.dataframe(env_df, width="stretch")
                    env_csv = convert_csv(env_df)
                    st.download_button(
                        "CSV 다운로드",
                        data=env_csv,
                        file_name=f"{school_name or 'data'}_환경.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("환경 데이터가 없어요.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


with tabs[2]:
    try:
        if growth_df.empty:
            st.info("📎 생육 CSV를 업로드해 주세요. 이 탭은 생육 데이터가 있어야 표시됩니다.")
        else:
            if "생중량" not in growth_df.columns:
                if {"지상부 질량", "지하부 질량"} <= set(growth_df.columns):
                    growth_df["생중량"] = growth_df["지상부 질량"] + growth_df["지하부 질량"]

            st.header("🥇 핵심 결과")

            if "개체번호" in growth_df.columns and "생중량" in growth_df.columns:
                fw = growth_df[["개체번호", "생중량"]].dropna()
                total_g = len(fw)
                avg_fw = fw["생중량"].mean() if len(fw) >= 2 else None
                if len(fw) >= 2:
                    max_row = fw.loc[fw["생중량"].idxmax()]
                    min_row = fw.loc[fw["생중량"].idxmin()]
                    max_id = max_row["개체번호"]
                    max_val = max_row["생중량"]
                    min_id = min_row["개체번호"]
                    min_val = min_row["생중량"]
                else:
                    max_id, max_val, min_id, min_val = None, None, None, None

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("총 개체수", f"{total_g}개")
                if avg_fw is not None:
                    c2.metric("평균 생중량", f"{avg_fw:.2f} g")
                else:
                    c2.metric("평균 생중량", "—")
                if max_id is not None:
                    c3.metric("생중량 최대 개체", f"개체 {max_id}  ({max_val:.2f} g)")
                else:
                    c3.metric("생중량 최대 개체", "—")
                if min_id is not None:
                    c4.metric("생중량 최소 개체", f"개체 {min_id}  ({min_val:.2f} g)")
                else:
                    c4.metric("생중량 최소 개체", "—")
            else:
                st.info("생육 데이터에 '개체번호'와 '생중량' 열이 필요해요.")

            st.divider()
            st.header("개체별 생육 비교 (막대 그래프)")
            st.caption("⭐ 평균 생중량이 가장 중요한 지표입니다. 개체번호는 분석에서 제외하고 값만 표시했어요.")

            bar_cols = []
            if "생중량" in growth_df.columns:
                bar_cols.append(("생중량", "생중량 (g)", "#27ae60"))
            for mc in ["잎 수", "지상부 길이", "지하부 길이"]:
                if mc in growth_df.columns:
                    bar_cols.append((mc, mc, "#2980b9"))

            if bar_cols:
                n = len(bar_cols)
                if n == 1:
                    figs = [growth_bar(growth_df, bar_cols[0][0], bar_cols[0][1], bar_cols[0][2])]
                    if figs[0] is not None:
                        st.plotly_chart(figs[0], width="stretch")
                else:
                    fig = make_subplots(
                        rows=1,
                        cols=n,
                        subplot_titles=[bc[1] for bc in bar_cols],
                    )
                    for i, (col, title, color) in enumerate(bar_cols):
                        data = growth_df[["개체번호", col]].dropna()
                        if len(data) >= 2:
                            fig.add_trace(
                                go.Bar(
                                    x=data["개체번호"].astype(str),
                                    y=data[col],
                                    name=title,
                                    marker_color=color,
                                    showlegend=False,
                                    hovertemplate="%{x}<br>%{y:.2f}<extra></extra>",
                                ),
                                row=1,
                                col=i + 1,
                            )
                    fig.update_layout(
                        margin=dict(l=40, r=20, t=40, b=40),
                        height=280,
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    )
                    for i in range(n):
                        fig.update_xaxes(title_text="개체", row=1, col=i + 1)
                        fig.update_yaxes(title_text=bar_cols[i][1], row=1, col=i + 1)
                    st.plotly_chart(fig, width="stretch")
            else:
                st.info("표시할 생육 측정 열이 없어요. CSV 열 이름을 확인해 주세요.")

            st.divider()
            st.header("생중량 분포")
            if "생중량" in growth_df.columns:
                fw_data = growth_df["생중량"].dropna()
                if len(fw_data) >= 4:
                    try:
                        fig = go.Figure()
                        fig.add_trace(
                            go.Box(
                                y=fw_data,
                                name="생중량",
                                fillcolor="#2ecc71",
                                marker=dict(color="#2ecc71"),
                                boxpoints="outliers",
                                jitter=0.3,
                                pointpos=-1.8,
                                hovertemplate="생중량: %{y:.2f} g<extra></extra>",
                            )
                        )
                        q1, q3 = fw_data.quantile([0.25, 0.75])
                        iqr = q3 - q1
                        upper = q3 + 1.5 * iqr
                        lower = q1 - 1.5 * iqr
                        fig.add_hline(y=upper, line_dash="dot", line_color="gray", annotation_text="상위 경계")
                        fig.add_hline(y=lower, line_dash="dot", line_color="gray", annotation_text="하위 경계")
                        fig.update_layout(
                            title="개체별 생중량 분포 (박스플롯)",
                            yaxis_title="생중량 (g)",
                            width=520,
                            height=340,
                            margin=dict(l=50, r=20, t=40, b=40),
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        )
                        st.plotly_chart(fig, width="stretch")
                    except Exception:
                        fig2, ax = plt.subplots(figsize=(6, 3.4))
                        ax.boxplot(fw_data.values, patch_artist=True, boxprops=dict(facecolor="#2ecc71"))
                        ax.set_title("개체별 생중량 분포")
                        ax.set_ylabel("생중량 (g)")
                        ax.set_xticklabels(["개체"])
                        st.pyplot(fig2)
                else:
                    st.info("분포 그래프를 그리기에 개체 수가 너무 적어요.")
            else:
                st.info("생중량 열이 없어 분포를 표시할 수 없어요.")

            st.divider()
            st.header("상관관계 분석 (산점도 + 추세선)")
            st.caption(
                "상관계수 r은 -1 ~ 1 사이의 값으로, 1에 가까울수록 함께 증가, -1에 가까울수록 한쪽이 증가할 때 다른 쪽이 감소하는関係을 나타냅니다."
            )

            if "잎 수" in growth_df.columns and "생중량" in growth_df.columns:
                fig1, r1 = corr_plot(growth_df, "잎 수", "생중량", "잎 수 vs 생중량")
                if fig1 is not None:
                    st.plotly_chart(fig1, width="stretch")
                    st.caption(f"잎 수–생중량 상관계수 r = {r1:.2f}")
            else:
                st.info("잎 수와 생중량 열이 모두 필요해요.")

            if "지상부 길이" in growth_df.columns and "생중량" in growth_df.columns:
                fig2, r2 = corr_plot(growth_df, "지상부 길이", "생중량", "지상부 길이 vs 생중량")
                if fig2 is not None:
                    st.plotly_chart(fig2, width="stretch")
                    st.caption(f"지상부 길이–생중량 상관계수 r = {r2:.2f}")
            else:
                st.info("지상부 길이와 생중량 열이 모두 필요해요.")

            st.divider()
            gd_exp = st.expander("📄 생육 데이터 원본 + CSV 다운로드")
            with gd_exp:
                if not growth_df.empty:
                    st.dataframe(growth_df, width="stretch")
                    growth_csv = convert_csv(growth_df)
                    st.download_button(
                        "CSV 다운로드",
                        data=growth_csv,
                        file_name=f"{school_name or 'data'}_생육결과.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("생육 데이터가 없어요.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
