import streamlit as st
import requests


API_URL = "http://localhost:8000"
QUERY_TIMEOUT_SECONDS = 180

st.set_page_config(page_title="NLP Multi-Table PostgreSQL Query", layout="wide")
st.title("NLP Intent Query on PostgreSQL")
st.caption("Ask in plain English, detect intent, and retrieve data from up to 4 tables.")


def load_metadata():
    try:
        resp = requests.get(f"{API_URL}/table-metadata", timeout=20)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"Could not load table metadata: {exc}")
        return None


metadata = load_metadata()
all_tables = metadata.get("all_tables", []) if metadata else []
default_allowed = metadata.get("allowed_tables", []) if metadata else []
max_tables = metadata.get("max_tables", 4) if metadata else 4
roles = metadata.get("roles", {}) if metadata else {}
model_reference = metadata.get("model_reference", {}) if metadata else {}
llm_ready = metadata.get("llm_ready", False) if metadata else False
llm_error = metadata.get("llm_error") if metadata else None

with st.sidebar:
    st.subheader("Model Reference")
    if model_reference:
        st.write("Provider:", model_reference.get("provider", "unknown"))
        st.write("Model:", model_reference.get("model", "unknown"))
        st.write("Base URL:", model_reference.get("base_url", "unknown"))
        st.code(model_reference.get("pull_command", ""), language="bash")
    st.write("LLM Ready:", llm_ready)
    if llm_error:
        st.warning(f"LLM issue: {llm_error}")

with st.expander("Prompt examples"):
    st.write("- show me 20 rows from magnetic_ngdr_backup")
    st.write("- count records in magnetic_ngdr_backup")
    st.write("- show data from district RAIPUR")
    st.write("- show records between 2023-01-01 and 2023-12-31")
    st.write("- show rows where magnetic_anomaly > 50")

col1, col2 = st.columns(2)
with col1:
    user_id = st.text_input("User ID", value="demo_user")
with col2:
    role_options = list(roles.keys()) if roles else ["viewer", "analyst", "admin"]
    role = st.selectbox("Role", options=role_options, index=0)

if role in roles:
    st.info(f"Role limit: max {roles[role]['max_limit']} rows per table, max {roles[role]['max_tables']} tables")

selected_tables = st.multiselect(
    f"Select up to {max_tables} tables for retrieval",
    options=all_tables,
    default=default_allowed,
    max_selections=max_tables,
)

prompt = st.text_area(
    "Enter prompt",
    height=120,
    placeholder="e.g., count records in magnetic_ngdr_backup where magnetic_anomaly > 50",
)

limit = st.slider("Requested row limit", min_value=1, max_value=500, value=50)

if st.button("Run NLP Query", use_container_width=True):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    elif not selected_tables:
        st.warning("Please select at least one table.")
    else:
        payload = {
            "prompt": prompt,
            "limit": limit,
            "selected_tables": selected_tables,
            "role": role,
            "user_id": user_id.strip() or "anonymous",
        }

        with st.spinner("Detecting intent and fetching data..."):
            try:
                resp = requests.post(
                    f"{API_URL}/query-data",
                    json=payload,
                    timeout=QUERY_TIMEOUT_SECONDS,
                )
                resp.raise_for_status()
                result = resp.json()

                st.subheader("Detected Intent")
                st.write(result.get("intent", "unknown"))

                parsed = result.get("parsed", {})
                st.write("Parsed details:", parsed)
                if parsed.get("sql"):
                    st.subheader("Generated SQL")
                    st.code(parsed["sql"], language="sql")
                if parsed.get("model_reference"):
                    st.subheader("Model Used")
                    st.json(parsed["model_reference"])

                tables = result.get("tables", [])
                st.write("Tables used:", tables)

                results = result.get("results", {})
                if not results:
                    st.warning("No data returned.")
                else:
                    for table_name in tables:
                        st.markdown(f"### {table_name}")
                        table_data = results.get(table_name, [])
                        if table_data:
                            if isinstance(table_data, dict):
                                st.json(table_data)
                            else:
                                st.dataframe(table_data, use_container_width=True)
                        else:
                            st.info("No rows returned for this table.")

            except requests.RequestException as exc:
                st.error(f"Query failed: {exc}")
            except Exception as exc:
                st.error(f"Frontend render failed: {exc}")

st.divider()
st.subheader("Audit Logs")
if role != "admin":
    st.caption("Select role = admin to view audit logs.")
else:
    log_limit = st.slider("Audit log rows", min_value=10, max_value=500, value=100, step=10)
    if st.button("Load Audit Logs"):
        try:
            resp = requests.get(
                f"{API_URL}/audit-logs",
                params={"role": role, "limit": log_limit},
                timeout=30,
            )
            resp.raise_for_status()
            logs = resp.json()
            st.dataframe(logs, use_container_width=True)
        except requests.RequestException as exc:
            st.error(f"Failed to load audit logs: {exc}")

if metadata and metadata.get("message"):
    st.info(metadata["message"])
