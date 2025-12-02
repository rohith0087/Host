import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

FILE_PATH = "Practice_with_pets.xlsx"

# --------------------------
# 1. LOAD & MERGE DATA
# --------------------------
@st.cache_data
def load_data():
    df_clinics = pd.read_excel(FILE_PATH, sheet_name="Sheet1")
    df_da = pd.read_excel(FILE_PATH, sheet_name="CREDIT RISK INDEX SCORE")

    # make join key consistent
    df_clinics = df_clinics.rename(columns={"postalCode": "Postal Code"})

    # merge on Postal Code
    df = df_clinics.merge(
        df_da,
        on="Postal Code",
        how="left",
        suffixes=("", "_DA"),
    )
    return df


df = load_data()

# --------------------------
# 2. PAGE CONFIG + GLOBAL CSS
# --------------------------
st.set_page_config(
    page_title="Pet Insurance Middleware — Market Intelligence",
    layout="wide",
)

st.markdown(
    """
    <style>
    .method-card {
        background: #020617;
        padding: 1rem 1.2rem;
        border-radius: 0.9rem;
        border: 1px solid rgba(16,185,129,0.35);
        margin-bottom: 0.9rem;
    }
    .method-badge {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.7rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        background: rgba(16,185,129,0.12);
        color: #6ee7b7;
        border: 1px solid rgba(16,185,129,0.35);
        margin-bottom: 0.4rem;
    }
    .method-title {
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 0.2rem;
        color: #e5e7eb;
    }
    .method-body {
        font-size: 0.85rem;
        color: #cbd5f5;
    }
    .method-math {
        font-family: monospace;
        font-size: 0.8rem;
        background: #020617;
        border-radius: 0.6rem;
        border: 1px solid rgba(148,163,184,0.4);
        padding: 0.6rem 0.8rem;
        color: #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🐾 Pet Insurance  — Market Intelligence Dashboard")
st.caption(
    "Use this app to find **where to launch**, which **clinics to partner with**, "
    "and how much **pet spend potential** exists in each area."
)

# --------------------------
# 3. SIDEBAR FILTERS
# --------------------------
st.sidebar.header("Filters")

provinces = sorted(df["province"].dropna().unique())
segments = sorted(df["Segment Label"].dropna().unique())
statuses = sorted(df["practice_status"].dropna().unique())

selected_provinces = st.sidebar.multiselect(
    "Province", provinces, default=provinces
)
selected_segments = st.sidebar.multiselect(
    "Market Segment", segments, default=segments
)
selected_status = st.sidebar.multiselect(
    "Practice Status", statuses, default=statuses
)

min_pets = float(df["Estimated Pets"].min() if df["Estimated Pets"].notna().any() else 0)
max_pets = float(df["Estimated Pets"].max() if df["Estimated Pets"].notna().any() else 0)
min_spend = float(df["Pet Spend Potential"].min() if df["Pet Spend Potential"].notna().any() else 0)
max_spend = float(df["Pet Spend Potential"].max() if df["Pet Spend Potential"].notna().any() else 0)

pets_range = st.sidebar.slider(
    "Estimated Pets range", min_pets, max_pets, (min_pets, max_pets)
)
spend_range = st.sidebar.slider(
    "Pet Spend Potential range", min_spend, max_spend, (min_spend, max_spend)
)

# apply filters
filtered = df.copy()
if selected_provinces:
    filtered = filtered[filtered["province"].isin(selected_provinces)]
if selected_segments:
    filtered = filtered[filtered["Segment Label"].isin(selected_segments)]
if selected_status:
    filtered = filtered[filtered["practice_status"].isin(selected_status)]

filtered = filtered[filtered["Estimated Pets"].between(pets_range[0], pets_range[1])]
filtered = filtered[filtered["Pet Spend Potential"].between(spend_range[0], spend_range[1])]

# --------------------------
# 4. KPI STRIP
# --------------------------
total_clinics = filtered["org_id"].nunique()
active_clinics = filtered[filtered["practice_status"] == "Active"]["org_id"].nunique()
invalid_postal = filtered["Postal Code"].isna().sum()
total_pets = filtered["Estimated Pets"].fillna(0).sum()
total_spend = filtered["Pet Spend Potential"].fillna(0).sum()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Clinics", f"{total_clinics}")
col2.metric("Active Clinics", f"{active_clinics}")
col3.metric("Rows with Missing Postal Code", f"{invalid_postal}")
col4.metric("Estimated Pets in Market", f"{int(total_pets):,}")
col5.metric("Total Pet Spend Potential ($)", f"{int(total_spend):,}")

st.markdown("---")

# --------------------------
# 5. TABS
# --------------------------
tab_overview, tab_clinics, tab_segments, tab_launch, tab_methodology = st.tabs(
    ["Overview", "Clinic Explorer", "Segment Insights", "Launch Planner", "Methodology"]
)

# ---------- Overview ----------
with tab_overview:
    st.subheader("Clinics by Province")
    if not filtered.empty:
        clinics_by_province = (
            filtered.groupby("province")["org_id"]
            .nunique()
            .reset_index(name="Clinic Count")
            .sort_values("Clinic Count", ascending=False)
        )
        fig1 = px.bar(
            clinics_by_province,
            x="province",
            y="Clinic Count",
            title="Number of Clinics by Province",
        )
        st.plotly_chart(fig1, width="stretch")
    else:
        st.info("No data for current filter selection.")

    st.subheader("Pet Spend Potential by Segment")
    if not filtered.empty:
        spend_by_segment = (
            filtered.groupby("Segment Label")["Pet Spend Potential"]
            .sum()
            .reset_index()
            .sort_values("Pet Spend Potential", ascending=False)
        )
        fig2 = px.bar(
            spend_by_segment,
            x="Segment Label",
            y="Pet Spend Potential",
            title="Total Pet Spend Potential by Segment",
        )
        st.plotly_chart(fig2, width="stretch")
    else:
        st.info("No data for current filter selection.")

# ---------- Clinic Explorer ----------
with tab_clinics:
    st.subheader("High-Value Clinics (Density vs Spend)")

    if not filtered.empty:
        fig3 = px.scatter(
            filtered,
            x="PET DENSITY SCORE",
            y="LOCAL SPEND ($)",
            size="Estimated Pets",
            color="Segment Label",
            hover_data=[
                "Full Name",
                "city",
                "province",
                "Postal Code",
                "practice_status",
            ],
            title="Pet Density vs Local Spend (circle size = Estimated Pets)",
        )
        st.plotly_chart(fig3, width="stretch")

        st.markdown("### Clinic Table (sorted by Pet Spend Potential)")
        display_cols = [
            "Full Name",
            "city",
            "province",
            "Postal Code",
            "practice_status",
            "Segment Label",
            "Estimated Pets",
            "Pet Spend Potential",
            "PET DENSITY SCORE",
            "LOCAL SPEND ($)",
        ]
        table = (
            filtered[display_cols]
            .drop_duplicates()
            .sort_values("Pet Spend Potential", ascending=False)
        )
        st.dataframe(table, use_container_width=True)

        csv = table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download filtered clinics as CSV",
            data=csv,
            file_name="filtered_clinics.csv",
            mime="text/csv",
        )
    else:
        st.info("No data for current filter selection.")

# ---------- Segment Insights ----------
with tab_segments:
    st.subheader("Segment ROI vs Spend")

    if not filtered.empty:
        seg_summary = (
            filtered.groupby("Segment Label")
            .agg(
                Pet_Spend=("Pet Spend Potential", "mean"),
                ROI=("Marketing ROI", "mean"),
                Pets=("Estimated Pets", "mean"),
            )
            .reset_index()
        )

        fig4 = px.scatter(
            seg_summary,
            x="Pet_Spend",
            y="ROI",
            size="Pets",
            color="Segment Label",
            hover_name="Segment Label",
            title="Average Pet Spend vs Marketing ROI by Segment",
        )
        st.plotly_chart(fig4, width="stretch")

        st.markdown("### Segment Summary Table")
        st.dataframe(
            seg_summary.sort_values("ROI", ascending=False),
            use_container_width=True,
        )
    else:
        st.info("No data for current filter selection.")

# ---------- Launch Planner ----------
with tab_launch:
    st.subheader("Launch Planner — Best Postal Codes to Start")

    if not filtered.empty:
        launch_cols = [
            "Postal Code",
            "province",
            "Estimated Pets",
            "Pet Spend Potential",
            "Marketing ROI",
        ]
        launch_df = (
            filtered[launch_cols]
            .dropna(subset=["Postal Code"])
            .groupby(["Postal Code", "province"], as_index=False)
            .agg(
                Estimated_Pets=("Estimated Pets", "sum"),
                Pet_Spend=("Pet Spend Potential", "sum"),
                ROI=("Marketing ROI", "mean"),
            )
        )

        def normalize(series: pd.Series) -> pd.Series:
            if series.max() == series.min():
                return pd.Series(1.0, index=series.index)
            return (series - series.min()) / (series.max() - series.min())

        launch_df["Score_Pets"] = normalize(launch_df["Estimated_Pets"])
        launch_df["Score_Spend"] = normalize(launch_df["Pet_Spend"])
        launch_df["Score_ROI"] = normalize(launch_df["ROI"])

        # weighted composite score
        launch_df["Launch_Score"] = (
            0.4 * launch_df["Score_Pets"]
            + 0.4 * launch_df["Score_Spend"]
            + 0.2 * launch_df["Score_ROI"]
        )

        launch_df = launch_df.sort_values("Launch_Score", ascending=False)

        top_n = st.slider("How many top postal codes to show?", 5, 50, 15)
        top_launch = launch_df.head(top_n)

        fig5 = px.bar(
            top_launch,
            x="Postal Code",
            y="Launch_Score",
            color="province",
            hover_data=["Estimated_Pets", "Pet_Spend", "ROI"],
            title="Top Launch Postal Codes (by composite score)",
        )
        st.plotly_chart(fig5, width="stretch")

        st.markdown("### Launch Plan Table")
        st.dataframe(top_launch, use_container_width=True)

        csv_launch = top_launch.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download launch plan as CSV",
            data=csv_launch,
            file_name="launch_postal_codes.csv",
            mime="text/csv",
        )
    else:
        st.info("No data for current filter selection.")

# ---------- Methodology ----------
# ---------- Methodology ----------
with tab_methodology:
    st.subheader("Methodology — from census data to launch decisions")

    st.markdown(
        """
        This page explains how we go from **DA-level census data** to:
        - a **Credit Risk Index** (financial strength),
        - a **Pet Density Index** (pet ownership intensity),
        - **Estimated pets, pet spend, and ROI**, and
        - final **segments & launch scores**.
        
        Use the controls below to tweak a hypothetical DA and see how the scores react.
        """,
    )

    # -------------------------
    # 1. INPUT PLAYGROUND
    # -------------------------
    st.markdown("### 1. Play with a hypothetical DA")

    c_in1, c_in2 = st.columns(2)

    with c_in1:
        st.markdown("**Economic profile**")
        median_income = st.slider(
            "Median household income ($)",
            min_value=20000,
            max_value=200000,
            value=80000,
            step=5000,
        )
        emp_rate = st.slider(
            "% with employment income",
            min_value=20.0,
            max_value=100.0,
            value=65.0,
            step=1.0,
        )
        lim_at = st.slider(
            "% low income (LIM-AT)",
            min_value=0.0,
            max_value=50.0,
            value=10.0,
            step=1.0,
        )

        st.markdown("**Population & households**")
        population = st.slider(
            "Population in DA",
            min_value=200,
            max_value=8000,
            value=2500,
            step=50,
        )
        avg_hh_size = st.slider(
            "Average household size",
            min_value=1.0,
            max_value=4.5,
            value=2.5,
            step=0.1,
        )

    with c_in2:
        st.markdown("**Housing & cost stress**")
        renter_rate = st.slider(
            "% renter households",
            min_value=0.0,
            max_value=100.0,
            value=30.0,
            step=1.0,
        )
        shelter_stress = st.slider(
            "% spending >30% of income on shelter",
            min_value=0.0,
            max_value=60.0,
            value=20.0,
            step=1.0,
        )

        st.markdown("**Built form & age**")
        median_home_value = st.slider(
            "Median home value ($)",
            min_value=150000,
            max_value=1500000,
            value=600000,
            step=25000,
        )
        pop_density = st.slider(
            "Population density (people/km²)",
            min_value=10,
            max_value=20000,
            value=2500,
            step=100,
        )
        median_age = st.slider(
            "Median age",
            min_value=20.0,
            max_value=60.0,
            value=40.0,
            step=1.0,
        )

    # -------------------------
    # 2. CALCULATE SCORES FROM INPUTS
    # -------------------------
    def normalize(val, benchmark, scale):
        return max(-2.0, min(2.0, (val - benchmark) / scale))

    # --- Credit Risk Index components ---
    base_credit = 50.0
    score = base_credit

    inc_contrib = 0.0
    if median_income > 0:
        inc_contrib = min(20.0, (median_income - 70000) / 2000.0)
        score += inc_contrib

    emp_contrib = min(15.0, (emp_rate - 60.0) * 0.5)
    score += emp_contrib

    renter_contrib = -(renter_rate - 30.0) * 0.3
    score += renter_contrib

    shelter_contrib = -(shelter_stress - 20.0) * 0.5
    score += shelter_contrib

    lim_contrib = -(lim_at - 10.0) * 0.5
    score += lim_contrib

    credit_raw = score
    credit_index = max(0, min(100, int(credit_raw)))

    # --- Pet Density Index components ---
    pd_base = 0.55
    pd_score = pd_base

    pd_inc = normalize(median_income, 70000, 30000) * 0.10
    pd_score += pd_inc

    pd_home = normalize(median_home_value, 500000, 300000) * 0.20
    pd_score += pd_home

    pd_hh = normalize(avg_hh_size, 2.4, 1.0) * 0.15
    pd_score += pd_hh

    pd_rent = -normalize(renter_rate, 30, 20) * 0.20
    pd_score += pd_rent

    pd_dens = -normalize(pop_density, 4000, 2000) * 0.10
    pd_score += pd_dens

    pd_shelter = -normalize(shelter_stress, 20, 10) * 0.10
    pd_score += pd_shelter

    pd_age = normalize(median_age, 40, 10) * 0.05
    pd_score += pd_age

    pet_density_index_raw = pd_score
    pet_density_index = max(0.20, min(0.80, pet_density_index_raw))

    # --- Business metrics ---
    households = population / avg_hh_size if avg_hh_size > 0 else 0.0
    est_pets = int(round(households * pet_density_index))

    # income factor for spend
    income_factor = median_income / 70000.0 if median_income > 0 else 1.0
    income_factor = max(0.5, min(2.0, income_factor))
    base_spend_per_pet = 800.0
    avg_spend_per_pet = base_spend_per_pet * income_factor
    pet_spend_potential = est_pets * avg_spend_per_pet
    roi_value = pet_spend_potential / households if households > 0 else 0.0

    # Segment score (for story)
    credit_norm = credit_index / 100.0
    pet_norm = (pet_density_index - 0.20) / 0.60
    segment_score = 0.5 * credit_norm + 0.5 * pet_norm

    if segment_score >= 0.75:
        seg_label = "Segment 1 – High-value pet market"
    elif segment_score >= 0.55:
        seg_label = "Segment 2 – Stable pet market"
    elif segment_score >= 0.35:
        seg_label = "Segment 3 – Emerging / value-focused"
    else:
        seg_label = "Segment 4 – High-risk / budget-constrained"

    # -------------------------
    # 3. KEY METRICS SNAPSHOT
    # -------------------------
    st.markdown("### 2. What this hypothetical DA looks like")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Credit Risk Index (0–100)", f"{credit_index}")
    k2.metric("Pet Density Index", f"{pet_density_index:0.2f}")
    k3.metric("Households", f"{int(households):,}")
    k4.metric("Estimated pets", f"{est_pets:,}")
    k5.metric("Pet spend potential ($/yr)", f"{int(pet_spend_potential):,}")

    st.caption(f"Segment classification: **{seg_label}**  (Score ≈ {segment_score:0.2f})")

    # -------------------------
    # 4. CONTRIBUTION CHARTS
    # -------------------------
    st.markdown("### 3. What is driving these scores?")

    c_chart1, c_chart2 = st.columns(2)

    # Credit Risk contributions chart
    with c_chart1:
        credit_factors = [
            "Base (50)",
            "Income",
            "Employment",
            "Renters",
            "Shelter >30%",
            "Low income",
        ]
        credit_values = [
            base_credit,
            inc_contrib,
            emp_contrib,
            renter_contrib,
            shelter_contrib,
            lim_contrib,
        ]
        fig_credit = px.bar(
            x=credit_factors,
            y=credit_values,
            title="Credit Risk – contribution by factor (points)",
            labels={"x": "Factor", "y": "Points added/removed"},
        )
        fig_credit.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=300)
        st.plotly_chart(fig_credit, use_container_width=True)
        st.caption(
            f"Total before clamping ≈ {credit_raw:0.1f} → final Credit Risk Index = {credit_index} (0–100)."
        )

    # Pet density contributions chart
    with c_chart2:
        pd_factors = [
            "Base (0.55)",
            "Income",
            "Home value",
            "HH size",
            "% renter",
            "Density",
            "Shelter >30%",
            "Median age",
        ]
        pd_values = [
            pd_base,
            pd_inc,
            pd_home,
            pd_hh,
            pd_rent,
            pd_dens,
            pd_shelter,
            pd_age,
        ]
        fig_pd = px.bar(
            x=pd_factors,
            y=pd_values,
            title="Pet Density – contribution by factor (Δ index)",
            labels={"x": "Factor", "y": "Change vs base"},
        )
        fig_pd.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=300)
        st.plotly_chart(fig_pd, use_container_width=True)
        st.caption(
            f"Raw Pet Density Index ≈ {pet_density_index_raw:0.2f} → clamped to {pet_density_index:0.2f} (0.20–0.80)."
        )

    st.markdown("---")



    # -------------------------
    # 6. TECHNICAL DETAILS (FORMULAS)
    # -------------------------
    st.markdown("### 5. Technical appendix — exact formulas")

    with st.expander("Credit Risk Index (0–100)"):
        st.markdown(
            """
            ```text
            score = 50.0

            # 1. Income (Median HH income)
            if income > 0:
                score += min(20, (income - 70,000) / 2,000)

            # 2. Employment (% with employment income)
            score += min(15, (emp_rate - 60) * 0.5)

            # 3. Tenure mix (% renter)
            score -= (renter_rate - 30) * 0.3

            # 4. Housing cost stress (% spending >30% on shelter)
            score -= (shelter_stress - 20) * 0.5

            # 5. Low income (% LIM-AT)
            score -= (lim_at - 10) * 0.5

            CreditRiskIndex = clamp( score, 0, 100 )
            ``` 
            """
        )

    with st.expander("Pet Density Index (0.20–0.80)"):
        st.markdown(
            """
            ```text
            normalize(x, benchmark, scale) = clamp( (x - benchmark) / scale , -2 , +2 )

            PetDensityIndex = 0.55
              + 0.10 * normalize(median_income,   70,000, 30,000)
              + 0.20 * normalize(home_value,      500,000, 300,000)
              + 0.15 * normalize(avg_hh_size,     2.4,    1.0)
              - 0.20 * normalize(renter_rate,     30,     20)
              - 0.10 * normalize(pop_density,     4,000,  2,000)
              - 0.10 * normalize(shelter_stress,  20,     10)
              + 0.05 * normalize(median_age,      40,     10)

            PetDensityIndex = clamp( PetDensityIndex, 0.20, 0.80 )
            ``` 
            """
        )

    with st.expander("Business metrics, Segment Score & Launch Score"):
        st.markdown(
            """
            ```text
            households     = population / avg_hh_size
            estimated_pets = households * PetDensityIndex

            income_factor  = clamp( median_income / 70,000 , 0.5 , 2.0 )
            base_spend_pet = 800
            avg_spend_pet  = base_spend_pet * income_factor

            PetSpendPotential = estimated_pets * avg_spend_pet
            MarketingROI      = PetSpendPotential / households

            credit_norm  = CreditRiskIndex / 100
            pet_norm     = (PetDensityIndex - 0.20) / 0.60
            SegmentScore = 0.5 * credit_norm + 0.5 * pet_norm
            # Thresholds:
            #   >= 0.75   → Segment 1 (High-value)
            #   0.55–0.75 → Segment 2 (Stable)
            #   0.35–0.55 → Segment 3 (Emerging / value-focused)
            #   < 0.35    → Segment 4 (High-risk / budget-constrained)

            # Postal-code LaunchScore (in Launch tab):
            #   Normalize Estimated_Pets, Pet_Spend, ROI to 0–1
            #   LaunchScore = 0.4 * Score_Pets
            #                + 0.4 * Score_Spend
            #                + 0.2 * Score_ROI
            ``` 
            """
        )
