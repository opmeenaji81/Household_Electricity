import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Household Electricity Analysis",
    page_icon=None,
    layout="wide"
)


# -----------------------------
# Load Data
# -----------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("household_power_consumption.csv")

    # Convert numeric columns
    numeric_columns = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert date
    df["Date"] = pd.to_datetime(df["Date"])

    # Create total submetering
    df["Total_Submetering"] = (
        df["Sub_metering_1"]
        + df["Sub_metering_2"]
        + df["Sub_metering_3"]
    )

    # Create hour
    df["Hour"] = pd.to_datetime(
        df["Time"],
        format="%H:%M:%S"
    ).dt.hour

    # Create day
    df["Day"] = df["Date"].dt.day_name()

    # Create month
    df["Month"] = df["Date"].dt.month

    # Energy outside sub-metering
    df["Energy_Outside_Submetering"] = (
        df["Global_active_power"]
        - df["Total_Submetering"]
    )

    return df


df = load_data()


# -----------------------------
# Sidebar Navigation
# -----------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "EDA",
        "Time Analysis",
        "Insights"
    ]
)


# =========================================================
# PAGE 1 - OVERVIEW
# =========================================================

if page == "Overview":

    st.title("Household Electricity Consumption Analysis")

    st.write(
        "Interactive analysis of household electricity consumption "
        "using historical power consumption data."
    )

    st.divider()

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Records",
        f"{len(df):,}"
    )

    col2.metric(
        "Number of Columns",
        len(df.columns)
    )

    col3.metric(
        "Start Date",
        df["Date"].min().strftime("%d-%m-%Y")
    )

    col4.metric(
        "End Date",
        df["Date"].max().strftime("%d-%m-%Y")
    )

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.divider()

    st.subheader("Missing Values")

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    if len(missing) > 0:
        st.dataframe(
            missing.rename("Missing Values"),
            use_container_width=True
        )
    else:
        st.success("No missing values found.")


# =========================================================
# PAGE 2 - EDA
# =========================================================

elif page == "EDA":

    st.title("Exploratory Data Analysis")

    # ----------------------------------
    # Summary Statistics
    # ----------------------------------

    st.subheader("Summary Statistics")

    columns = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity"
    ]

    st.dataframe(
        df[columns].describe(),
        use_container_width=True
    )

    st.divider()

    # ----------------------------------
    # Voltage Distribution
    # ----------------------------------

    st.subheader("Distribution of Voltage")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        df["Voltage"].dropna(),
        bins=30
    )

    ax.set_xlabel("Voltage")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Voltage")

    st.pyplot(fig)

    st.divider()

    # ----------------------------------
    # Active Power vs Intensity
    # ----------------------------------

    st.subheader(
        "Global Active Power vs Global Intensity"
    )

    correlation = df[
        "Global_active_power"
    ].corr(
        df["Global_intensity"]
    )

    st.write(
        f"Correlation coefficient: **{correlation:.3f}**"
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(
        df["Global_active_power"],
        df["Global_intensity"],
        alpha=0.2,
        s=10
    )

    ax.set_xlabel("Global Active Power")
    ax.set_ylabel("Global Intensity")
    ax.set_title("Active Power vs Global Intensity")

    st.pyplot(fig)

    st.divider()

    # ----------------------------------
    # Active Power vs Voltage
    # ----------------------------------

    st.subheader(
        "Global Active Power vs Voltage"
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    hb = ax.hexbin(
        df["Voltage"],
        df["Global_active_power"],
        gridsize=40,
        mincnt=1
    )

    ax.set_xlabel("Voltage")
    ax.set_ylabel("Global Active Power")
    ax.set_title("Global Active Power vs Voltage")

    fig.colorbar(
        hb,
        ax=ax,
        label="Number of Observations"
    )

    st.pyplot(fig)

    st.divider()

    # ----------------------------------
    # Sub-metering
    # ----------------------------------

    st.subheader(
        "Average Power Consumption by Sub-metering Category"
    )

    submetering_avg = df[
        [
            "Sub_metering_1",
            "Sub_metering_2",
            "Sub_metering_3"
        ]
    ].mean()

    fig, ax = plt.subplots(figsize=(8, 5))

    submetering_avg.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Sub-metering Category")
    ax.set_ylabel("Average Power Consumption")
    ax.set_title(
        "Average Power Consumption by Sub-metering Category"
    )

    st.pyplot(fig)


# =========================================================
# PAGE 3 - TIME ANALYSIS
# =========================================================

elif page == "Time Analysis":

    st.title("Time-Based Analysis")

    # ----------------------------------
    # Hourly Active Power
    # ----------------------------------

    st.subheader(
        "Hourly Trend of Global Active Power"
    )

    hourly_power = df.groupby(
        "Hour"
    )["Global_active_power"].mean()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        hourly_power.index,
        hourly_power.values,
        marker="o"
    )

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Global Active Power")
    ax.set_title(
        "Hourly Trend of Global Active Power"
    )

    ax.set_xticks(range(24))
    ax.grid(alpha=0.3)

    st.pyplot(fig)

    st.divider()

    # ----------------------------------
    # Day of Week
    # ----------------------------------

    st.subheader(
        "Average Power Consumption by Day"
    )

    daily_power = df.groupby(
        "Day"
    )["Global_active_power"].mean()

    days_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    daily_power = daily_power.reindex(
        days_order
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        daily_power.index,
        daily_power.values
    )

    ax.set_xlabel("Day")
    ax.set_ylabel("Average Global Active Power")
    ax.set_title(
        "Average Power Consumption by Day of Week"
    )

    plt.xticks(rotation=45)

    st.pyplot(fig)

    st.divider()

    # ----------------------------------
    # Hourly Submetering
    # ----------------------------------

    st.subheader(
        "Hourly Trend of Total Sub-metering"
    )

    hourly_submetering = df.groupby(
        "Hour"
    )["Total_Submetering"].mean()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        hourly_submetering.index,
        hourly_submetering.values,
        marker="o"
    )

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Total Submetering")
    ax.set_title(
        "Hourly Trend of Total Submetering"
    )

    ax.set_xticks(range(24))
    ax.grid(alpha=0.3)

    st.pyplot(fig)

    st.divider()

    # ----------------------------------
    # Hourly Voltage
    # ----------------------------------

    st.subheader(
        "Hourly Voltage Fluctuation"
    )

    hourly_voltage = df.groupby(
        "Hour"
    )["Voltage"].mean()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        hourly_voltage.index,
        hourly_voltage.values,
        marker="o"
    )

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Voltage")
    ax.set_title(
        "Hourly Voltage Fluctuation"
    )

    ax.set_xticks(range(24))
    ax.grid(alpha=0.3)

    st.pyplot(fig)

    st.divider()

    # ----------------------------------
    # Monthly Power
    # ----------------------------------

    st.subheader(
        "Average Global Active Power by Month"
    )

    monthly_power = df.groupby(
        "Month"
    )["Global_active_power"].mean()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        monthly_power.index,
        monthly_power.values,
        marker="o"
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Average Global Active Power")
    ax.set_title(
        "Average Global Active Power by Month"
    )

    ax.set_xticks(range(1, 13))
    ax.grid(alpha=0.3)

    st.pyplot(fig)


# =========================================================
# PAGE 4 - INSIGHTS
# =========================================================

elif page == "Insights":

    st.title("Key Insights")

    # ----------------------------------
    # Minimum and Maximum Power
    # ----------------------------------

    min_idx = df[
        "Global_active_power"
    ].idxmin()

    max_idx = df[
        "Global_active_power"
    ].idxmax()

    min_row = df.loc[min_idx]
    max_row = df.loc[max_idx]

    st.subheader(
        "Minimum and Maximum Global Active Power"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write("Minimum")

        st.metric(
            "Minimum Power",
            f"{min_row['Global_active_power']:.3f}"
        )

        st.write(
            f"Date: {min_row['Date'].strftime('%d-%m-%Y')}"
        )

        st.write(
            f"Time: {min_row['Time']}"
        )

    with col2:

        st.write("Maximum")

        st.metric(
            "Maximum Power",
            f"{max_row['Global_active_power']:.3f}"
        )

        st.write(
            f"Date: {max_row['Date'].strftime('%d-%m-%Y')}"
        )

        st.write(
            f"Time: {max_row['Time']}"
        )

    st.divider()

    # ----------------------------------
    # Highest Submetering
    # ----------------------------------

    st.subheader(
        "Highest Average Sub-metering Category"
    )

    submetering_avg = df[
        [
            "Sub_metering_1",
            "Sub_metering_2",
            "Sub_metering_3"
        ]
    ].mean()

    highest_category = submetering_avg.idxmax()
    highest_value = submetering_avg.max()

    st.write(
        f"**{highest_category}** has the highest average "
        f"consumption: **{highest_value:.3f}**"
    )

    st.divider()

    # ----------------------------------
    # Energy Outside Submetering
    # ----------------------------------

    st.subheader(
        "Energy Consumed Outside Three Sub-metering Categories"
    )

    outside_energy = df[
        "Energy_Outside_Submetering"
    ].mean()

    st.metric(
        "Average Energy Outside Sub-metering",
        f"{outside_energy:.3f}"
    )

    st.divider()

    # ----------------------------------
    # Highest Hour
    # ----------------------------------

    st.subheader(
        "Hour with Highest Average Total Sub-metering"
    )

    hourly_submetering = df.groupby(
        "Hour"
    )["Total_Submetering"].mean()

    highest_hour = hourly_submetering.idxmax()
    highest_hour_value = hourly_submetering.max()

    st.metric(
        "Highest Hour",
        f"{highest_hour}:00"
    )

    st.write(
        f"Average Total Submetering: "
        f"**{highest_hour_value:.3f}**"
    )

    st.divider()

    # ----------------------------------
    # Unusual Observations
    # ----------------------------------

    st.subheader(
        "Unusual Total Sub-metering Observations"
    )

    mean = df["Total_Submetering"].mean()
    std = df["Total_Submetering"].std()

    high = mean + std
    low = mean - std

    unusual = df[
        (df["Total_Submetering"] > high)
        |
        (df["Total_Submetering"] < low)
    ]

    st.write(
        f"Mean: **{mean:.3f}**"
    )

    st.write(
        f"Standard Deviation: **{std:.3f}**"
    )

    st.write(
        f"Unusual observations: **{len(unusual):,}**"
    )

    st.dataframe(
        unusual[
            [
                "Date",
                "Time",
                "Total_Submetering"
            ]
        ].head(20),
        use_container_width=True
    )