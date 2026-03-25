import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Growth AI Dashboard", layout="wide")


# -------------------------------------------------
# Utilities
# -------------------------------------------------
def generate_channel_data(days=30, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days)

    sessions = rng.integers(1200, 5000, size=days)
    conversions = np.maximum(
        10, (sessions * rng.uniform(0.02, 0.07, size=days)).astype(int)
    )
    spend = rng.integers(4000, 18000, size=days)
    revenue = conversions * rng.integers(700, 1800, size=days)

    df = pd.DataFrame(
        {
            "date": dates,
            "sessions": sessions,
            "conversions": conversions,
            "spend": spend,
            "revenue": revenue,
        }
    )
    df["cvr"] = (df["conversions"] / df["sessions"] * 100).round(2)
    df["roas"] = (df["revenue"] / df["spend"]).replace([np.inf, -np.inf], np.nan).round(2)
    return df


def generate_leads(n=200, seed=7):
    rng = np.random.default_rng(seed)
    companies = [f"Lead-{i:03d}" for i in range(1, n + 1)]
    source = rng.choice(
        ["Organic", "Paid Search", "Referral", "LinkedIn", "Direct"], size=n
    )
    industry = rng.choice(
        ["Fintech", "EdTech", "HealthTech", "SaaS", "E-commerce"], size=n
    )
    visits = rng.integers(1, 20, size=n)
    pages = rng.integers(1, 12, size=n)
    time_on_site = rng.integers(20, 420, size=n)
    downloaded = rng.choice([0, 1], size=n, p=[0.65, 0.35])
    demo_requested = rng.choice([0, 1], size=n, p=[0.8, 0.2])
    email_open_rate = rng.uniform(0.05, 0.80, size=n)

    df = pd.DataFrame(
        {
            "company": companies,
            "source": source,
            "industry": industry,
            "visits": visits,
            "pages_viewed": pages,
            "time_on_site_sec": time_on_site,
            "downloaded_asset": downloaded,
            "demo_requested": demo_requested,
            "email_open_rate": (email_open_rate * 100).round(1),
        }
    )

    df["lead_score"] = (
        df["visits"] * 3
        + df["pages_viewed"] * 4
        + (df["time_on_site_sec"] / 15)
        + df["downloaded_asset"] * 18
        + df["demo_requested"] * 35
        + df["email_open_rate"] * 0.4
    ).round(0).astype(int)

    df["priority"] = pd.cut(
        df["lead_score"],
        bins=[0, 50, 90, 140, 999],
        labels=["Low", "Medium", "High", "Hot"],
        include_lowest=True,
    )

    return df.sort_values("lead_score", ascending=False).reset_index(drop=True)


def normalize_marketing_df(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    expected = ["date", "sessions", "conversions", "spend", "revenue"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Marketing CSV missing columns: {', '.join(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["sessions", "conversions", "spend", "revenue"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=expected).sort_values("date")
    df["cvr"] = (df["conversions"] / df["sessions"] * 100).round(2)
    df["roas"] = (df["revenue"] / df["spend"]).replace([np.inf, -np.inf], np.nan).round(2)
    return df


def normalize_leads_df(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    expected = [
        "company",
        "source",
        "industry",
        "visits",
        "pages_viewed",
        "time_on_site_sec",
        "downloaded_asset",
        "demo_requested",
        "email_open_rate",
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Leads CSV missing columns: {', '.join(missing)}")

    numeric_cols = [
        "visits",
        "pages_viewed",
        "time_on_site_sec",
        "downloaded_asset",
        "demo_requested",
        "email_open_rate",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=expected)

    df["lead_score"] = (
        df["visits"] * 3
        + df["pages_viewed"] * 4
        + (df["time_on_site_sec"] / 15)
        + df["downloaded_asset"] * 18
        + df["demo_requested"] * 35
        + df["email_open_rate"] * 0.4
    ).round(0).astype(int)

    df["priority"] = pd.cut(
        df["lead_score"],
        bins=[0, 50, 90, 140, 999],
        labels=["Low", "Medium", "High", "Hot"],
        include_lowest=True,
    )

    return df.sort_values("lead_score", ascending=False).reset_index(drop=True)


def get_ai_insights(channel_df, leads_df):
    latest = channel_df.iloc[-1]
    avg_roas = channel_df["roas"].dropna().mean()
    avg_cvr = channel_df["cvr"].dropna().mean()
    hot_leads = int((leads_df["priority"] == "Hot").sum())

    high_or_hot = leads_df[leads_df["priority"].isin(["High", "Hot"])]
    best_source = (
        high_or_hot["source"].value_counts().idxmax()
        if not high_or_hot.empty
        else "Organic"
    )

    insights = [
        f"Average ROAS is {avg_roas:.2f}. If this remains above 2.5, increase budget gradually on the highest-converting channel.",
        f"Average conversion rate is {avg_cvr:.2f}%. If it drops below 3.5%, review landing-page clarity, CTA hierarchy, and page speed.",
        f"There are {hot_leads} hot leads right now. Prioritize outreach to the top 10 within 24 hours.",
        f"The strongest source for high-intent leads is {best_source}. Create a dedicated nurture flow and retargeting sequence for this channel.",
        f"Latest revenue recorded is ₹{int(latest['revenue']):,}. Add automated alerts when revenue falls 20% below the 7-day average.",
    ]
    return insights


def build_exec_summary(channel_df, leads_df):
    total_sessions = int(channel_df["sessions"].sum())
    total_conversions = int(channel_df["conversions"].sum())
    avg_roas = channel_df["roas"].mean()
    hot_leads = int((leads_df["priority"] == "Hot").sum())
    best_source = (
        leads_df.groupby("source")["lead_score"]
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )

    return f"""EXECUTIVE SUMMARY

Total sessions: {total_sessions:,}
Total conversions: {total_conversions:,}
Average ROAS: {avg_roas:.2f}
Hot leads: {hot_leads}
Best lead source: {best_source}

Recommended actions:
1. Prioritize top high-intent leads for rapid outreach.
2. Double down on the best-performing acquisition source.
3. Add automated performance alerts and weekly report generation.
"""


def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.title("Growth AI Dashboard")
mode = st.sidebar.radio("Data mode", ["Demo data", "Upload CSVs"])
show_raw = st.sidebar.checkbox("Show raw tables", value=False)

if mode == "Demo data":
    days = st.sidebar.slider("Date range (days)", 7, 90, 30)
    seed = st.sidebar.number_input("Data seed", min_value=1, max_value=9999, value=42)
    channel_df = generate_channel_data(days=days, seed=seed)
    leads_df = generate_leads(n=200, seed=seed + 1)
    source_note = "Using generated demo data for presentation and prototyping."
else:
    st.sidebar.markdown("### Upload files")
    marketing_file = st.sidebar.file_uploader("Marketing CSV", type=["csv"])
    leads_file = st.sidebar.file_uploader("Leads CSV", type=["csv"])

    if marketing_file is not None and leads_file is not None:
        try:
            channel_df = normalize_marketing_df(pd.read_csv(marketing_file))
            leads_df = normalize_leads_df(pd.read_csv(leads_file))
            source_note = "Using uploaded CSV data."
        except Exception as e:
            st.error(f"File processing error: {e}")
            st.stop()
    else:
        st.info("Upload both CSV files to use real data, or switch to Demo data.")
        st.stop()

insights = get_ai_insights(channel_df, leads_df)
summary_text = build_exec_summary(channel_df, leads_df)

# -------------------------------------------------
# Header
# -------------------------------------------------
st.title("Growth AI Dashboard – AI-First Prototype")
st.caption("AI-first prototype for campaign visibility, lead scoring, and operational recommendations.")
st.info(source_note)

hero1, hero2 = st.columns([1.4, 1])

with hero1:
    st.markdown(
        """
        ### What this app demonstrates
        - marketing analytics dashboarding
        - lead scoring and prioritization
        - AI-style recommendations
        - exportable reporting workflow
        - product thinking with PRD-style framing
        """
    )

with hero2:
    st.markdown(
        """
        ### Ideal use case
        A growth team wants one place to:
        - track performance
        - spot sales-ready leads
        - generate action plans
        - export quick summaries
        """
    )

# -------------------------------------------------
# KPI row
# -------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Sessions", f"{int(channel_df['sessions'].sum()):,}")

with col2:
    st.metric("Total Conversions", f"{int(channel_df['conversions'].sum()):,}")

with col3:
    st.metric("Average ROAS", f"{channel_df['roas'].mean():.2f}")

with col4:
    st.metric("Hot Leads", f"{int((leads_df['priority'] == 'Hot').sum())}")

# -------------------------------------------------
# Charts
# -------------------------------------------------
left, right = st.columns((1.4, 1))

with left:
    st.subheader("Campaign Performance")
    st.line_chart(channel_df.set_index("date")[["sessions", "conversions", "revenue"]])

    st.subheader("Spend vs Revenue")
    st.bar_chart(channel_df.set_index("date")[["spend", "revenue"]])

with right:
    st.subheader("Lead Priority Breakdown")
    priority_counts = leads_df["priority"].value_counts().sort_index()
    st.bar_chart(priority_counts)

    st.subheader("Average Lead Score by Source")
    avg_source_scores = (
        leads_df.groupby("source")["lead_score"].mean().sort_values(ascending=False)
    )
    st.bar_chart(avg_source_scores)

# -------------------------------------------------
# Workbench
# -------------------------------------------------
st.subheader("Lead Scoring Workbench")

f1, f2, f3 = st.columns(3)

with f1:
    selected_priority = st.multiselect(
        "Priority",
        options=["Low", "Medium", "High", "Hot"],
        default=["High", "Hot"],
    )

with f2:
    selected_source = st.multiselect(
        "Source",
        options=sorted(leads_df["source"].unique().tolist()),
        default=sorted(leads_df["source"].unique().tolist()),
    )

with f3:
    min_score = st.slider(
        "Minimum lead score",
        0,
        int(leads_df["lead_score"].max()),
        80,
    )

filtered_leads = leads_df[
    leads_df["priority"].isin(selected_priority)
    & leads_df["source"].isin(selected_source)
    & (leads_df["lead_score"] >= min_score)
]

st.dataframe(filtered_leads.head(50), use_container_width=True)

st.subheader("🔥 Top 5 High-Intent Leads")
st.dataframe(leads_df.head(5), use_container_width=True)

# -------------------------------------------------
# Recommendations + reports
# -------------------------------------------------
rec_col, report_col = st.columns((1.2, 1))

with rec_col:
    st.subheader("AI-Style Recommendations")
    for i, insight in enumerate(insights, start=1):
        st.markdown(f"**{i}.** {insight}")

with report_col:
    st.subheader("Export / Reporting")
    st.download_button(
        "Download filtered leads CSV",
        data=to_csv_bytes(filtered_leads),
        file_name="filtered_leads.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download executive summary",
        data=summary_text.encode("utf-8"),
        file_name="executive_summary.txt",
        mime="text/plain",
    )
    st.text_area("Executive summary preview", summary_text, height=220)

# -------------------------------------------------
# Product thinking block
# -------------------------------------------------
with st.expander("Product thinking / PRD summary"):
    st.markdown(
        """
        ### Problem
        Growth teams often operate across disconnected tools for analytics, lead management, and reporting.

        ### Goal
        Build one operational dashboard that helps teams:
        - monitor campaign health
        - score and prioritize leads
        - generate clear next actions
        - export stakeholder-friendly summaries

        ### Core modules
        1. Marketing performance dashboard
        2. Lead scoring engine
        3. Recommendation engine
        4. Export/reporting layer
        5. Future CRM/API integrations

        ### Suggested next versions
        - Google Analytics / Search Console API integration
        - ad platform connectors
        - email or Slack alert automation
        - user authentication and client-level dashboards
        - LLM-generated weekly summaries
        - PDF report generation
        """
    )

# -------------------------------------------------
# Raw data
# -------------------------------------------------
if show_raw:
    st.subheader("Raw Marketing Data")
    st.dataframe(channel_df, use_container_width=True)

    st.subheader("Raw Leads Data")
    st.dataframe(leads_df, use_container_width=True)

st.markdown("---")
st.caption("Built in Streamlit as a demo-friendly product prototype for growth operations and AI-first workflows.")
