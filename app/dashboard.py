"""
app/dashboard.py
----------------
Full Streamlit dashboard for the Social Media Sentiment Analysis project.

Run with:
  streamlit run app/dashboard.py
"""

import os, sys, json, re, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.cleaner   import clean_text
from src.features  import vader_scores, vader_label

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "SentimentIQ Dashboard",
    page_icon   = "📊",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&family=DM+Mono&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #12151c !important;
    border-right: 1px solid #2a3045;
}
section[data-testid="stSidebar"] * { color: #e8eaf0 !important; }

/* ── Main background ── */
.stApp { background: #0a0c10; color: #e8eaf0; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #12151c;
    border: 1px solid #2a3045;
    border-radius: 12px;
    padding: 16px;
}
[data-testid="metric-container"] label { color: #8892aa !important; font-family: 'DM Mono'; font-size: 12px; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-family: 'Syne'; font-size: 28px; font-weight: 800; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 12px; }

/* ── Cards ── */
.dash-card {
    background: #12151c;
    border: 1px solid #2a3045;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
}
.card-title { font-family: 'Syne'; font-size: 16px; font-weight: 700; color: #e8eaf0; margin-bottom: 4px; }
.card-sub   { font-family: 'DM Mono'; font-size: 11px; color: #4a5270; margin-bottom: 16px; }

/* ── Post items ── */
.post-card {
    background: #1a1e28;
    border: 1px solid #2a3045;
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 8px;
}
.post-meta  { font-family: 'DM Mono'; font-size: 11px; color: #4a5270; margin-bottom: 4px; }
.post-text  { font-size: 13px; color: #c8cfe0; line-height: 1.5; margin-bottom: 6px; }

/* ── Badges ── */
.badge-pos { background:#0d2818; color:#22c55e; border:1px solid #166534; border-radius:6px; padding:3px 10px; font-family:'DM Mono'; font-size:10px; }
.badge-neg { background:#280d0d; color:#ef4444; border:1px solid #991b1b; border-radius:6px; padding:3px 10px; font-family:'DM Mono'; font-size:10px; }
.badge-neu { background:#2a1d08; color:#f59e0b; border:1px solid #92400e; border-radius:6px; padding:3px 10px; font-family:'DM Mono'; font-size:10px; }

/* ── Confidence bars ── */
.conf-label { font-family:'DM Mono'; font-size:11px; color:#8892aa; }

/* ── Header ── */
.dashboard-header {
    background: linear-gradient(135deg, #12151c 0%, #1a1e28 100%);
    border: 1px solid #2a3045;
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 20px;
}
.header-title { font-family:'Syne'; font-size:28px; font-weight:800; color:#e8eaf0; margin:0; }
.header-sub   { font-family:'DM Mono'; font-size:12px; color:#4a5270; margin-top:4px; }
.live-pill    { display:inline-flex; align-items:center; gap:6px; background:#0d2818; border:1px solid #166534; border-radius:20px; padding:4px 12px; font-family:'DM Mono'; font-size:11px; color:#22c55e; }

button[data-testid="baseButton-secondary"] { background:#1a1e28 !important; border-color:#2a3045 !important; color:#e8eaf0 !important; }
.stTextArea textarea { background:#1a1e28 !important; border-color:#2a3045 !important; color:#e8eaf0 !important; border-radius:10px !important; }
.stSelectbox div[data-baseweb="select"] { background:#1a1e28 !important; border-color:#2a3045 !important; }
.stTabs [data-baseweb="tab-list"] { background:#12151c; border-radius:10px; border:1px solid #2a3045; }
.stTabs [data-baseweb="tab"] { color:#8892aa; }
.stTabs [aria-selected="true"] { color:#e8eaf0 !important; background:#1a1e28 !important; border-radius:8px; }
hr { border-color: #2a3045; }
</style>
""", unsafe_allow_html=True)

# ── Dark plot theme ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0a0c10", "axes.facecolor":  "#12151c",
    "axes.edgecolor":   "#2a3045", "axes.labelcolor": "#8892aa",
    "xtick.color":      "#8892aa", "ytick.color":     "#8892aa",
    "text.color":       "#e8eaf0", "grid.color":      "#2a3045",
    "grid.alpha": 0.4, "font.family": "sans-serif",
})
PALETTE = {"positive": "#22c55e", "neutral": "#f59e0b", "negative": "#ef4444"}
COLORS  = list(PALETTE.values())


# ── Data loading ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_data(show_spinner=False)
def load_data():
    path = os.path.join(ROOT, "data", "social_media_posts.csv")
    if not os.path.exists(path):
        # Generate on the fly
        sys.path.insert(0, os.path.join(ROOT, "data"))
        from generate_dataset import generate
        df = generate()
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["sentiment"] = df["sentiment"].str.lower().str.strip()
    return df


@st.cache_resource(show_spinner=False)
def load_predictor():
    try:
        from src.predictor import SentimentPredictor
        return SentimentPredictor()
    except Exception:
        return None


# ── Helpers ───────────────────────────────────────────────────────────────────
def badge_html(sentiment: str) -> str:
    cls = {"positive": "badge-pos", "negative": "badge-neg", "neutral": "badge-neu"}.get(sentiment, "badge-neu")
    label = sentiment.upper()
    return f'<span class="{cls}">{label}</span>'


def sentiment_color(s): return PALETTE.get(s, "#8892aa")


def make_fig(): return plt.subplots(figsize=(8, 4))


def dark_legend(ax, **kw):
    leg = ax.legend(facecolor="#12151c", edgecolor="#2a3045", labelcolor="#e8eaf0", **kw)
    return leg


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📊 SentimentIQ")
    st.markdown('<div style="font-family:DM Mono;font-size:11px;color:#4a5270;margin-bottom:16px">v2.0 · ML-Powered</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown("**Filters**")
    df_all = load_data()

    platform_opts  = ["All"] + sorted(df_all["platform"].unique().tolist())
    sentiment_opts = ["All", "positive", "neutral", "negative"]
    brand_opts     = ["All"] + sorted(df_all["brand"].unique().tolist())

    sel_platform  = st.selectbox("Platform",  platform_opts)
    sel_sentiment = st.selectbox("Sentiment", sentiment_opts)
    sel_brand     = st.selectbox("Brand",     brand_opts)
    sel_days      = st.slider("Last N days", 7, 90, 30)

    st.divider()
    st.markdown("**Model Info**")
    predictor = load_predictor()
    if predictor:
        meta = predictor.model_info()
        st.markdown(f"""
        <div style="font-family:DM Mono;font-size:11px;line-height:2">
        <span style="color:#4a5270">model</span> &nbsp;&nbsp;&nbsp;&nbsp; <span style="color:#3b82f6">{meta.get('model_name','LR')[:15]}</span><br>
        <span style="color:#4a5270">accuracy</span> &nbsp; <span style="color:#22c55e">{meta.get('accuracy',0):.2%}</span><br>
        <span style="color:#4a5270">f1 macro</span> &nbsp; <span style="color:#22c55e">{meta.get('f1_macro',0):.4f}</span><br>
        <span style="color:#4a5270">trained</span> &nbsp;&nbsp; <span style="color:#e8eaf0">{meta.get('trained_at','—')[:10]}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Model not trained yet.\nRun: `python src/train_model.py`", icon="⚠️")

    st.divider()
    st.caption("📁 [GitHub Repo](https://github.com) | Built with Streamlit")


# ── Apply filters ─────────────────────────────────────────────────────────────
df = df_all.copy()
cutoff = pd.Timestamp.now() - pd.Timedelta(days=sel_days)
df = df[df["timestamp"] >= cutoff]
if sel_platform  != "All": df = df[df["platform"]  == sel_platform]
if sel_sentiment != "All": df = df[df["sentiment"] == sel_sentiment]
if sel_brand     != "All": df = df[df["brand"]     == sel_brand]


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="dashboard-header">
  <p class="header-title">📊 Social Media Sentiment Dashboard</p>
  <p class="header-sub">AI-powered analysis of {len(df):,} posts across {df['platform'].nunique() if len(df)>0 else 0} platforms · Last updated {datetime.now().strftime('%H:%M:%S')}</p>
  <span class="live-pill">● LIVE ANALYSIS</span>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠  Overview", "🔍  Live Analyzer", "📋  Post Feed", "📈  Brand Monitor", "📖  About"
])


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 1 — OVERVIEW
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    if len(df) == 0:
        st.warning("No data matches current filters.")
    else:
        vc = df["sentiment"].value_counts()
        total = len(df)
        pos_p = vc.get("positive", 0) / total
        neg_p = vc.get("negative", 0) / total
        neu_p = vc.get("neutral",  0) / total

        # ── Metrics ──────────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("✅ Positive",  f"{pos_p:.1%}", f"+{pos_p-0.55:.1%}")
        c2.metric("❌ Negative",  f"{neg_p:.1%}", f"{neg_p-0.23:.1%}")
        c3.metric("⚪ Neutral",   f"{neu_p:.1%}", f"+{neu_p-0.22:.1%}")
        c4.metric("📝 Total Posts", f"{total:,}",  f"+{total//10:,}")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 1: Trend + Donut ──────────────────────────────────────────────
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown('<div class="dash-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Sentiment Trend</div><div class="card-sub">Daily rolling average</div>', unsafe_allow_html=True)
            daily = (df.set_index("timestamp")
                       .resample("D")["sentiment"]
                       .value_counts(normalize=True)
                       .unstack(fill_value=0) * 100)
            fig, ax = plt.subplots(figsize=(9, 3.5))
            for sent in ["positive", "neutral", "negative"]:
                if sent in daily.columns:
                    ax.plot(daily.index, daily[sent], label=sent.capitalize(),
                            color=PALETTE[sent], linewidth=2, marker="o", markersize=3)
                    ax.fill_between(daily.index, daily[sent], alpha=0.08, color=PALETTE[sent])
            ax.set_ylabel("Percentage (%)")
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3); ax.spines[["top","right"]].set_visible(False)
            dark_legend(ax, loc="upper left")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="dash-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Distribution</div><div class="card-sub">Current period</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(4, 4))
            vals    = [vc.get(s, 0) for s in ["positive", "neutral", "negative"]]
            cols    = [PALETTE[s] for s in ["positive", "neutral", "negative"]]
            wedges, texts, autotexts = ax.pie(
                vals, colors=cols, autopct="%1.1f%%", startangle=90,
                wedgeprops={"width": 0.6, "edgecolor": "#0a0c10", "linewidth": 2},
                pctdistance=0.78,
            )
            for at in autotexts:
                at.set_fontsize(11); at.set_color("#e8eaf0"); at.set_fontweight("bold")
            ax.set_facecolor("#12151c")
            patches = [mpatches.Patch(color=cols[i], label=["Positive","Neutral","Negative"][i]) for i in range(3)]
            ax.legend(handles=patches, loc="lower center", facecolor="#12151c",
                      edgecolor="#2a3045", labelcolor="#e8eaf0", fontsize=10,
                      bbox_to_anchor=(0.5, -0.08), ncol=3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Row 2: Platform + Hourly ──────────────────────────────────────────
        col3, col4 = st.columns(2)

        with col3:
            st.markdown('<div class="dash-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Platform Breakdown</div><div class="card-sub">Posts by source</div>', unsafe_allow_html=True)
            plat = df.groupby(["platform","sentiment"]).size().unstack(fill_value=0)
            fig, ax = plt.subplots(figsize=(7, 3.5))
            x = np.arange(len(plat))
            w = 0.25
            for i, sent in enumerate(["positive", "neutral", "negative"]):
                if sent in plat.columns:
                    ax.bar(x + i*w, plat[sent], width=w, label=sent.capitalize(),
                           color=PALETTE[sent], alpha=0.9, edgecolor="#0a0c10")
            ax.set_xticks(x + w); ax.set_xticklabels(plat.index)
            ax.set_ylabel("Posts")
            ax.grid(axis="y", alpha=0.3); ax.spines[["top","right"]].set_visible(False)
            dark_legend(ax)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="dash-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Posting Volume by Hour</div><div class="card-sub">When users are most active</div>', unsafe_allow_html=True)
            df["hour"] = df["timestamp"].dt.hour
            hourly = df.groupby(["hour","sentiment"]).size().unstack(fill_value=0)
            fig, ax = plt.subplots(figsize=(7, 3.5))
            for sent in ["positive", "negative"]:
                if sent in hourly.columns:
                    ax.bar(hourly.index, hourly[sent], label=sent.capitalize(),
                           color=PALETTE[sent], alpha=0.75, width=0.8)
            ax.set_xlabel("Hour of Day"); ax.set_ylabel("Posts")
            ax.grid(axis="y", alpha=0.3); ax.spines[["top","right"]].set_visible(False)
            dark_legend(ax)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Word frequency ────────────────────────────────────────────────────
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Top Keywords by Sentiment</div><div class="card-sub">Most frequent words after cleaning</div>', unsafe_allow_html=True)
        col5, col6, col7 = st.columns(3)
        for col_ui, sent, title in zip([col5, col6, col7],
                                       ["positive","negative","neutral"],
                                       ["✅ Positive","❌ Negative","⚪ Neutral"]):
            sub = df[df["sentiment"] == sent]["text"].astype(str)
            words = " ".join(sub).lower()
            words = re.sub(r"[^a-z\s]", " ", words).split()
            stopset = {"the","a","an","is","are","was","and","or","for","in","to","it",
                       "of","with","my","i","this","that","be","on","at","have","had","has"}
            freq = Counter(w for w in words if len(w) > 3 and w not in stopset).most_common(10)
            with col_ui:
                st.markdown(f"**{title}**")
                if freq:
                    labels_, vals_ = zip(*freq)
                    fig, ax = plt.subplots(figsize=(4, 3.5))
                    bars = ax.barh(list(labels_)[::-1], list(vals_)[::-1],
                                   color=PALETTE[sent], alpha=0.85, edgecolor="#0a0c10")
                    ax.spines[["top","right","left","bottom"]].set_visible(False)
                    ax.set_facecolor("#12151c"); ax.grid(axis="x", alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()
        st.markdown('</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 2 — LIVE ANALYZER
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 🔍 Live Sentiment Analyzer")
    st.markdown("Enter any social media text below. The model will classify it instantly.")

    examples = [
        "Absolutely love this product! Best purchase ever, highly recommend!",
        "Worst experience of my life. Never buying from them again.",
        "The product is okay, does what it promises. Nothing special.",
        "Delivery was super fast but the packaging was slightly damaged.",
    ]

    st.markdown("**Quick examples:**")
    ecols = st.columns(4)
    for i, ex in enumerate(examples):
        if ecols[i].button(f"Ex {i+1}", key=f"ex_{i}", use_container_width=True):
            st.session_state["analyzer_text"] = ex

    user_text = st.text_area(
        "Enter text to analyze:",
        value=st.session_state.get("analyzer_text", ""),
        height=120,
        placeholder="Paste a tweet, review, comment, or any social media text here…",
        key="analyzer_input",
    )

    col_a, col_b = st.columns([1, 4])
    analyze_btn  = col_a.button("Analyze ↗", type="primary", use_container_width=True)
    clear_btn    = col_b.button("Clear", use_container_width=False)
    if clear_btn:
        st.session_state["analyzer_text"] = ""
        st.rerun()

    if analyze_btn and user_text.strip():
        with st.spinner("Classifying sentiment…"):
            time.sleep(0.4)

            # Use trained model if available, else VADER only
            if predictor:
                result = predictor.predict(user_text)
            else:
                vs    = vader_scores(user_text)
                vl    = vader_label(user_text)
                conf  = abs(vs["compound"])
                score_d = {"positive": vs["pos"], "neutral": vs["neu"], "negative": vs["neg"]}
                result = {"label": vl, "confidence": conf, "scores": score_d,
                          "emoji": {"positive":"😊","neutral":"😐","negative":"😠"}[vl],
                          "color": PALETTE[vl], "vader_scores": vs}

        label = result["label"]
        color = result["color"]
        conf  = result["confidence"]

        # ── Result card ───────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="dash-card" style="border-color:{color}44;border-left:3px solid {color}">
          <div style="display:flex;align-items:center;gap:16px;margin-bottom:16px">
            <span style="font-size:48px">{result['emoji']}</span>
            <div>
              <div style="font-family:Syne;font-size:22px;font-weight:800;color:{color}">{label.upper()}</div>
              <div style="font-family:DM Mono;font-size:12px;color:#8892aa">Confidence: {conf:.2%}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Confidence bars ───────────────────────────────────────────────────
        st.markdown("**Confidence Scores**")
        scores = result.get("scores", {})
        for sent_key in ["positive", "neutral", "negative"]:
            pct = scores.get(sent_key, 0.0)
            col_x, col_y = st.columns([1, 8])
            col_x.markdown(f'<div style="font-family:DM Mono;font-size:11px;color:#8892aa;padding-top:6px">{sent_key[:3].upper()}</div>', unsafe_allow_html=True)
            col_y.progress(float(pct), text=f"{pct:.1%}")

        # ── VADER comparison ──────────────────────────────────────────────────
        with st.expander("VADER Raw Scores"):
            vs = result.get("vader_scores", {})
            vcols = st.columns(4)
            vcols[0].metric("Compound", f"{vs.get('compound',0):.3f}")
            vcols[1].metric("Positive",  f"{vs.get('pos',0):.3f}")
            vcols[2].metric("Neutral",   f"{vs.get('neu',0):.3f}")
            vcols[3].metric("Negative",  f"{vs.get('neg',0):.3f}")

        # ── Clean text view ───────────────────────────────────────────────────
        with st.expander("Preprocessed Text"):
            cleaned = clean_text(user_text)
            st.code(cleaned, language=None)

    elif analyze_btn:
        st.warning("Please enter some text first.")


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 3 — POST FEED
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 📋 Post Feed")

    fc1, fc2, fc3 = st.columns([2, 2, 2])
    feed_sent  = fc1.selectbox("Sentiment", ["All","positive","neutral","negative"], key="feed_sent")
    feed_plat  = fc2.selectbox("Platform",  ["All"] + sorted(df["platform"].unique().tolist()), key="feed_plat")
    feed_n     = fc3.number_input("Show N posts", 5, 50, 20, 5)

    feed_df = df.copy()
    if feed_sent != "All": feed_df = feed_df[feed_df["sentiment"] == feed_sent]
    if feed_plat != "All": feed_df = feed_df[feed_df["platform"]  == feed_plat]
    feed_df = feed_df.sort_values("timestamp", ascending=False).head(int(feed_n))

    for _, row in feed_df.iterrows():
        sent     = row["sentiment"]
        color    = PALETTE.get(sent, "#8892aa")
        badge    = badge_html(sent)
        st.markdown(f"""
        <div class="post-card" style="border-left:3px solid {color}">
          <div class="post-meta">
            {row.get('brand','—')} &nbsp;·&nbsp; {row.get('platform','—')} &nbsp;·&nbsp; {str(row['timestamp'])[:16]}
            &nbsp;&nbsp;
            {badge}
          </div>
          <div class="post-text">{row['text']}</div>
          <div style="font-family:DM Mono;font-size:10px;color:#4a5270">
            ♥ {row.get('likes',0):,} &nbsp;&nbsp; 🔁 {row.get('retweets',0):,} &nbsp;&nbsp; 💬 {row.get('replies',0):,}
            &nbsp;&nbsp;&nbsp; Topic: {row.get('topic','—')}
          </div>
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 4 — BRAND MONITOR
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### 📈 Brand Sentiment Monitor")

    brand_df = df_all.copy()
    brand_summary = (brand_df.groupby(["brand","sentiment"])
                              .size().unstack(fill_value=0)
                              .reset_index())
    for col in ["positive","neutral","negative"]:
        if col not in brand_summary.columns:
            brand_summary[col] = 0
    brand_summary["total"] = brand_summary[["positive","neutral","negative"]].sum(axis=1)
    brand_summary["pos_%"] = (brand_summary["positive"] / brand_summary["total"] * 100).round(1)
    brand_summary["neg_%"] = (brand_summary["negative"] / brand_summary["total"] * 100).round(1)
    brand_summary["neu_%"] = (brand_summary["neutral"]  / brand_summary["total"] * 100).round(1)
    brand_summary["net"]   = brand_summary["pos_%"] - brand_summary["neg_%"]
    brand_summary          = brand_summary.sort_values("net", ascending=False)

    # ── Net sentiment score bar ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(brand_summary["brand"], brand_summary["net"],
                  color=[PALETTE["positive"] if v >= 0 else PALETTE["negative"]
                         for v in brand_summary["net"]],
                  edgecolor="#0a0c10", width=0.6)
    for bar, v in zip(bars, brand_summary["net"]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (1 if v >= 0 else -3),
                f"{v:.0f}", ha="center", va="bottom", fontsize=10)
    ax.axhline(0, color="#2a3045", linewidth=1)
    ax.set_ylabel("Net Sentiment Score (Pos% − Neg%)")
    ax.set_title("Brand Net Sentiment Score", pad=10)
    ax.grid(axis="y", alpha=0.3); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Stacked % bar ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(brand_summary))
    ax.bar(x, brand_summary["pos_%"], label="Positive", color=PALETTE["positive"], alpha=0.9)
    ax.bar(x, brand_summary["neu_%"], bottom=brand_summary["pos_%"], label="Neutral", color=PALETTE["neutral"], alpha=0.9)
    ax.bar(x, brand_summary["neg_%"], bottom=brand_summary["pos_%"]+brand_summary["neu_%"], label="Negative", color=PALETTE["negative"], alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(brand_summary["brand"], rotation=20, ha="right")
    ax.set_ylabel("Percentage (%)"); ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3); ax.spines[["top","right"]].set_visible(False)
    dark_legend(ax, loc="upper right")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Table ─────────────────────────────────────────────────────────────────
    display_cols = ["brand","total","pos_%","neu_%","neg_%","net"]
    st.dataframe(
        brand_summary[display_cols].rename(columns={
            "brand":"Brand","total":"Total Posts","pos_%":"Positive %",
            "neu_%":"Neutral %","neg_%":"Negative %","net":"Net Score"
        }).set_index("Brand"),
        use_container_width=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 5 — ABOUT
# ──────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("""
### 📖 About This Project

**Social Media Sentiment Analysis Dashboard** is a full-stack NLP + ML project that demonstrates
real-world social media analytics using Python, Scikit-learn, and Streamlit.

#### 🔧 Tech Stack
| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| NLP | VADER Sentiment, NLTK |
| Features | TF-IDF (bigrams, 8K vocab) |
| Model | Logistic Regression + Complement NB |
| Dashboard | Streamlit + Matplotlib + Seaborn |
| Data | Synthetic (3,100 posts) |

#### 🏗️ Architecture
```
Raw Text → Cleaner → TF-IDF + VADER → Classifier → Prediction → Dashboard
```

#### 📊 Model Performance
- Accuracy: ~91%
- F1 Macro: ~0.90
- Classes: Positive / Neutral / Negative

#### 👤 Author
Built as a portfolio project for placement and internship applications.
""")


# ──────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<div style="text-align:center;font-family:DM Mono;font-size:11px;color:#4a5270">'
    'SentimentIQ v2.0 · Built with Streamlit · Social Media Sentiment Analysis Dashboard'
    '</div>',
    unsafe_allow_html=True
)
