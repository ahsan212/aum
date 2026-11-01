import pandas as pd
import streamlit as st
import altair as alt
from pathlib import Path

st.set_page_config(page_title="Smart Campus Digital Twin", page_icon="🏫", layout="wide")
DATA_PATH = Path(__file__).parent / "data" / "week4_test_dataset.csv"
REQUIRED_COLUMNS = ["timestamp", "room_id", "temperature_c", "occupancy_persons", "energy_kwh", "comfort_status", "data_source"]

@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        st.stop()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data.dropna(subset=["timestamp", "room_id"]).sort_values("timestamp")
    return data

df = load_data(DATA_PATH)

st.title("AI-Enhanced Digital Twin for Smart Campus Energy and Comfort Management")
st.warning("SIMULATED DATA — Values shown in this prototype are not live campus measurements.")

with st.sidebar:
    st.header("Filters")
    selected_room = st.selectbox("Room", sorted(df["room_id"].astype(str).unique()))
    room_df = df[df["room_id"].astype(str) == selected_room].copy()
    min_date, max_date = room_df["timestamp"].dt.date.min(), room_df["timestamp"].dt.date.max()
    selected_dates = st.date_input("Date period", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date = end_date = selected_dates
    st.caption("Change the room or dates to update all panels.")

mask=(room_df["timestamp"].dt.date >= start_date) & (room_df["timestamp"].dt.date <= end_date)
filtered=room_df.loc[mask].copy()
if filtered.empty:
    st.error("No records are available for the selected period.")
    st.stop()
latest=filtered.iloc[-1]

page=st.radio("Screen", ["Room overview", "Room detail", "Recommendations"], horizontal=True)

if page == "Room overview":
    st.subheader("Room overview")
    cols=st.columns(4)
    cols[0].metric("Temperature", f'{latest["temperature_c"]:.1f} °C')
    cols[1].metric("Occupancy", f'{int(latest["occupancy_persons"])} persons')
    cols[2].metric("Energy", f'{latest["energy_kwh"]:.3f} kWh / 15 min')
    cols[3].metric("Comfort status", str(latest["comfort_status"]))
    st.caption(f'Latest record: {latest["timestamp"]:%d %b %Y, %H:%M}')
    summary=(df.groupby("room_id",as_index=False).agg(temperature_c=("temperature_c","last"), occupancy_persons=("occupancy_persons","last"), energy_kwh=("energy_kwh","last"), comfort_status=("comfort_status","last")))
    summary.columns=["Room","Temperature (°C)","Occupancy (persons)","Energy (kWh/15 min)","Comfort status"]
    st.dataframe(summary, use_container_width=True, hide_index=True)

elif page == "Room detail":
    st.subheader(f"Room detail — {selected_room}")
    metric=st.selectbox("Chart variable", ["temperature_c", "occupancy_persons", "energy_kwh"])
    labels={"temperature_c":"Temperature (°C)","occupancy_persons":"Occupancy (persons)","energy_kwh":"Energy (kWh per 15 min)"}
    chart=alt.Chart(filtered).mark_line().encode(x=alt.X("timestamp:T",title="Date and time"),y=alt.Y(f"{metric}:Q",title=labels[metric]),tooltip=[alt.Tooltip("timestamp:T",title="Time"),alt.Tooltip(f"{metric}:Q",title=labels[metric])]).properties(height=420)
    st.altair_chart(chart,use_container_width=True)
    with st.expander("View filtered records"):
        st.dataframe(filtered, use_container_width=True, hide_index=True)

else:
    st.subheader("Recommendations panel")
    st.info("PLACEHOLDER — Recommendation logic is not connected yet.")
    c1,c2,c3=st.columns(3)
    c1.metric("Next-period energy forecast", "Not available", help="Reserved for the ML forecasting module")
    c2.metric("Fixed-schedule baseline", "Not available", help="Reserved for baseline comparison")
    c3.metric("Estimated saving", "Not available", help="Reserved for decision logic")
    st.markdown("### Placeholder recommendation")
    st.write("No recommendation is generated at this stage. This panel will later show the suggested HVAC action, reason, expected energy effect and comfort impact.")
    st.button("Apply recommendation", disabled=True)
    st.caption("The disabled button is intentional. Automatic control is outside the current prototype scope.")

st.divider()
st.caption("Prototype status: Streamlit skeleton | Data source: Simulated CSV | Predictions: Placeholder only")
