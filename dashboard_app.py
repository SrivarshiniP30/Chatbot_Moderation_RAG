import streamlit as st
import os
import re
import pandas as pd
import plotly.express as px
from config import Config # Import Config to get log file path

# --- Dashboard Configuration ---
# User will manually refresh the dashboard.

# --- Log Parsing Logic ---
def parse_moderation_log(log_file_path: str) -> dict:
    """
    Parses the moderation log file and extracts relevant metrics, including detailed block types.
    This version uses more robust regex patterns to accurately categorize all interaction types.
    """
    log_data = {
        "total_user_inputs": 0,             # Total distinct user inputs attempted
        "user_input_accepted": 0,
        "user_input_blocked_overall": 0,    # Sum of all blocked inputs (rules + LLM general)
        "user_input_blocked_hate_speech": 0, # Rule-based blocks
        "user_input_blocked_pii": 0,         # Rule-based blocks
        "user_input_blocked_jailbreak": 0,   # Rule-based blocks
        "user_input_blocked_llm_general": 0, # LLM-based blocks from input moderation (e.g., PROMPT INJECTION)

        "total_llm_generations": 0,         # Total times LLM was asked to generate a response (i.e., user input was accepted)
        "llm_output_accepted": 0,
        "llm_output_blocked": 0,            # LLM's own output was blocked by moderation ("wrong replies")

        "all_log_lines": []                 # Stores all log lines for raw display
    }

    try:
        if not os.path.exists(log_file_path):
            st.warning(f"Log file not found at: {log_file_path}. No data to display.")
            return log_data # Return empty if file not found

        with open(log_file_path, "r") as f:
            for line in f:
                log_data["all_log_lines"].append(line.strip())

                # --- User Input Processing ---
                # Pattern for user input accepted
                if re.search(r"INFO - User input accepted; sending to main LLM:", line):
                    log_data["total_user_inputs"] += 1
                    log_data["user_input_accepted"] += 1
                    log_data["total_llm_generations"] += 1 # LLM only generates if input is accepted
                # Pattern for user input blocked (rule-based or LLM-based)
                elif re.search(r"WARNING - User input was blocked by moderation:", line):
                    log_data["total_user_inputs"] += 1
                    log_data["user_input_blocked_overall"] += 1

                    if "Hate speech detected" in line:
                        log_data["user_input_blocked_hate_speech"] += 1
                    elif "Personal identifiable information detected" in line:
                        log_data["user_input_blocked_pii"] += 1
                    elif "Jailbreak attempt detected" in line:
                        log_data["user_input_blocked_jailbreak"] += 1
                    # This captures LLM-based input blocks that are not specific rule types
                    elif re.search(r"User input blocked \(LLM-based\):", line):
                        log_data["user_input_blocked_llm_general"] += 1
                # Fallback for older log formats (if any are still present)
                elif re.search(r"WARNING - User input blocked \(Rule-based\):", line):
                    log_data["total_user_inputs"] += 1
                    log_data["user_input_blocked_overall"] += 1
                    if "Hate speech detected" in line:
                        log_data["user_input_blocked_hate_speech"] += 1
                    elif "PII detected" in line:
                        log_data["user_input_blocked_pii"] += 1
                    elif "Jailbreak attempt detected" in line:
                        log_data["user_input_blocked_jailbreak"] += 1
                elif re.search(r"WARNING - User input blocked by LLM moderation:", line):
                    log_data["total_user_inputs"] += 1
                    log_data["user_input_blocked_overall"] += 1
                    log_data["user_input_blocked_llm_general"] += 1


                # --- LLM Output Processing ---
                # Pattern for LLM output accepted
                if re.search(r"INFO - Main LLM Response passed output moderation:", line):
                    log_data["llm_output_accepted"] += 1
                # Pattern for LLM output blocked
                elif re.search(r"WARNING - LLM output blocked by moderation:", line):
                    log_data["llm_output_blocked"] += 1

    except Exception as e:
        st.error(f"Error parsing log file: {e}. Please ensure the log file is accessible and not corrupted.")
    return log_data

# --- Streamlit UI for Dashboard ---
def main():
    st.set_page_config(page_title="Chatbot Moderation Dashboard", layout="wide")
    st.title("📊 Chatbot Moderation Analytics Dashboard")
    st.markdown("""
        This dashboard provides insights into user interactions and LLM moderation effectiveness.
        It's designed for administrators and privacy-conscious monitoring.
    """)
    st.markdown("---")

    # Manual Refresh Button
    if st.button("Refresh Dashboard Data", key="refresh_dashboard_button", use_container_width=True):
        st.experimental_rerun() # Forces a full rerun of the script to refresh data

    st.empty().write(f"Last updated: {pd.to_datetime('now').strftime('%Y-%m-%d %H:%M:%S')}")


    # Parse logs
    logs_data = parse_moderation_log(Config.LOG_FILE)

    # --- Metrics Cards ---
    st.header("Key Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total User Inputs (Attempted)", value=logs_data["total_user_inputs"])
    with col2:
        st.metric(label="Total LLM Generations (Attempted)", value=logs_data["total_llm_generations"])
    with col3:
        st.metric(label="Overall Input Block Rate",
                  value=f"{((logs_data['user_input_blocked_overall'] / logs_data['total_user_inputs']) * 100):.1f}%" if logs_data['total_user_inputs'] > 0 else "0.0%")

    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric(label="User Inputs Accepted", value=logs_data["user_input_accepted"])
    with col5:
        st.metric(label="User Inputs Blocked", value=logs_data["user_input_blocked_overall"])
    with col6:
        st.metric(label="LLM Outputs Blocked (Wrong Replies)", value=logs_data["llm_output_blocked"])

    st.markdown("---")

    # --- User Input Moderation Breakdown (Graph) ---
    st.header("User Input Moderation Breakdown")

    # Always include all categories, even if counts are 0, for comprehensive display
    input_overview_data = {
        "Category": ["Accepted", "Hate Speech Block", "PII Block", "Jailbreak Block", "LLM General Block"],
        "Count": [
            logs_data["user_input_accepted"],
            logs_data["user_input_blocked_hate_speech"],
            logs_data["user_input_blocked_pii"],
            logs_data["user_input_blocked_jailbreak"],
            logs_data["user_input_blocked_llm_general"]
        ]
    }

    input_overview_df = pd.DataFrame(input_overview_data)
    # Only plot if there's any data at all
    if input_overview_df["Count"].sum() > 0:
        fig_input_overview = px.pie(input_overview_df, values="Count", names="Category",
                                    title="Overall User Input Moderation Status",
                                    color="Category",
                                    color_discrete_map={
                                        "Accepted": "green",
                                        "Hate Speech Block": "red",
                                        "PII Block": "darkorange",
                                        "Jailbreak Block": "purple",
                                        "LLM General Block": "brown"
                                    })
        st.plotly_chart(fig_input_overview, use_container_width=True)
    else:
        st.info("No user input data to display in the pie chart yet.")


    # Detailed bar chart for BLOCKED User Inputs
    st.subheader("Breakdown of Blocked User Inputs")
    blocked_input_data = {
        "Block Type": ["Hate Speech", "PII", "Jailbreak", "LLM General"],
        "Count": [
            logs_data["user_input_blocked_hate_speech"],
            logs_data["user_input_blocked_pii"],
            logs_data["user_input_blocked_jailbreak"],
            logs_data["user_input_blocked_llm_general"]
        ]
    }
    blocked_input_df = pd.DataFrame(blocked_input_data)

    if blocked_input_df["Count"].sum() > 0:
        fig_blocked_inputs = px.bar(blocked_input_df, x="Block Type", y="Count",
                                    title="Blocked User Inputs by Type",
                                    color="Block Type",
                                    color_discrete_map={
                                         "Hate Speech": "red",
                                         "PII": "darkorange",
                                         "Jailbreak": "purple",
                                         "LLM General": "brown"
                                    },
                                    text="Count") # Display count on bars
        st.plotly_chart(fig_blocked_inputs, use_container_width=True)
    else:
        st.info("No blocked user input data to display in the bar chart yet.")


    st.markdown("---")

    # --- LLM Output Moderation Breakdown (Graph) ---
    st.header("LLM Output Moderation Breakdown")
    llm_output_data = {
        "Category": ["Accepted", "Blocked (Wrong Reply)"],
        "Count": [
            logs_data["llm_output_accepted"],
            logs_data["llm_output_blocked"]
        ]
    }
    llm_output_df = pd.DataFrame(llm_output_data)

    if llm_output_df["Count"].sum() > 0:
        # Set custom colors for accepted/blocked
        color_map_llm_output = {"Accepted": "green", "Blocked (Wrong Reply)": "red"}
        fig_llm_output = px.bar(llm_output_df, x="Category", y="Count",
                                 title="LLM Output Moderation Status",
                                 color="Category",
                                 color_discrete_map=color_map_llm_output,
                                 text="Count") # Display count on bars
        st.plotly_chart(fig_llm_output, use_container_width=True)
    else:
        st.info("No LLM output data to display in the chart yet.")


    st.markdown("---")

    # --- Raw Logs Viewer ---
    st.header("Raw Activity Logs")
    st.write(f"Displaying all {len(logs_data['all_log_lines'])} entries from `{Config.LOG_FILE}`.")
    st.code("\n".join(logs_data["all_log_lines"]), language="log") # REMOVED height argument

    st.markdown("---")
    st.caption("Dashboard powered by Streamlit and Plotly")

if __name__ == "__main__":
    main()

