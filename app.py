import os, re, json
import pandas as pd
import duckdb
import sqlglot
import streamlit as st
from huggingface_hub import InferenceClient  # Official client for router

# --------------------------
# Safe getters (ENV or empty)
# --------------------------
def safe_get_env(name: str, default=None):
    return os.environ.get(name, default)

DEFAULT_MODEL = safe_get_env("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
DEFAULT_TOKEN = safe_get_env("HF_TOKEN", "")

# --------------------------
# UI
# --------------------------
st.set_page_config(page_title="LLM → SQL", page_icon="🧠", layout="wide")
st.title("🧠→🗄️  LLM SQL Generator (Hugging Face Router)")

with st.sidebar:
    st.subheader("LLM Settings")
    model = st.text_input("HF Model", value=DEFAULT_MODEL, help="e.g. Qwen/Qwen2.5-7B-Instruct")
    token = st.text_input("HF Token (paste here)", type="password", value=DEFAULT_TOKEN)
    temp = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.slider("Max new tokens", 64, 2048, 512, 64)
    dialect = st.selectbox("SQL Dialect", ["postgres", "duckdb", "mysql", "sqlite", "bigquery", "snowflake"], index=0)
    st.caption("Paste your Hugging Face token here (required for model access).")

# --------------------------
# Prompting
# --------------------------
SQL_SYSTEM_PROMPT = """You are a senior data analyst who writes correct, executable SQL.
Follow the user's SQL dialect strictly. Return ONLY the SQL code (inside a code block) unless asked.
Prefer explicit column lists over SELECT * when reasonable.
If the question is ambiguous, add concise SQL comments at the top with assumptions.
"""

FEWSHOTS = [
    {
        "schema": """
        Table customers(id INT PRIMARY KEY, name TEXT, country TEXT);
        Table orders(id INT PRIMARY KEY, customer_id INT, order_date DATE, total NUMERIC);
        """,
        "dialect": "postgres",
        "question": "Monthly revenue for 2025 (completed only), month as YYYY-MM",
        "sql": """-- Completed orders only; Postgres date_trunc and to_char
SELECT to_char(date_trunc('month', order_date), 'YYYY-MM') AS month,
       SUM(total) AS revenue
FROM orders
WHERE order_date >= DATE '2025-01-01'
  AND order_date <  DATE '2026-01-01'
GROUP BY 1
ORDER BY 1;"""
    }
]

def build_messages(schema: str, question: str, dialect: str):
    messages = [{"role": "system", "content": SQL_SYSTEM_PROMPT}]
    for ex in FEWSHOTS:
        messages.append({
            "role": "user",
            "content": f"SQL DIALECT: {ex['dialect']}\nSCHEMA:\n{ex['schema'].strip()}\n"
                       f"QUESTION: {ex['question']}\nReturn only SQL."
        })
        messages.append({"role": "assistant", "content": f"```sql\n{ex['sql'].strip()}\n```"})
    messages.append({
        "role": "user",
        "content": f"SQL DIALECT: {dialect}\nSCHEMA:\n{schema.strip()}\n\nQUESTION: {question.strip()}\nReturn only SQL."
    })
    return messages

def extract_sql(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    return m.group(1).strip() if m else text.strip()

def validate_sql(sql: str, dialect: str):
    try:
        sqlglot.parse_one(sql, read=dialect)
        return True, None
    except Exception as e:
        return False, str(e)

def call_hf_chat(model_id: str, hf_token: str, messages: list,
                 max_new_tokens=512, temperature=0.2, top_p=0.95, timeout=60):
    """
    Uses the new Hugging Face Router via InferenceClient.
    No manual URL required; handles chat completions automatically.
    """
    if not hf_token:
        raise RuntimeError("Missing HF token. Paste it in the sidebar.")

    client = InferenceClient(model=model_id, token=hf_token.strip(), timeout=timeout)
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_new_tokens,
        top_p=top_p,
    )
    return response.choices[0].message.content

# --------------------------
# Main UI
# --------------------------
st.markdown("**Paste your schema (DDL or plain description), enter a question, and generate SQL.**")
schema = st.text_area(
    "Schema",
    height=220,
    placeholder=(
        "Table customers(customer_id BIGINT, first_name TEXT, last_name TEXT, email TEXT, country TEXT, signup_date DATE);\n"
        "Table products(product_id BIGINT, product_name TEXT, category TEXT, unit_price NUMERIC);\n"
        "Table orders(order_id BIGINT, customer_id BIGINT, order_ts TIMESTAMPTZ, status TEXT, total_amount NUMERIC);\n"
        "Table order_items(order_item_id BIGINT, order_id BIGINT, product_id BIGINT, quantity INT, item_price NUMERIC);\n"
        "-- customers.customer_id = orders.customer_id\n"
        "-- orders.order_id = order_items.order_id\n"
        "-- order_items.product_id = products.product_id"
    )
)
question = st.text_area("Question", height=120, placeholder="Monthly revenue for 2025 for completed orders only, as YYYY-MM.")
if st.button("Generate SQL"):
    try:
        messages = build_messages(schema, question, dialect)
        raw = call_hf_chat(model, token, messages, max_new_tokens=max_tokens, temperature=temp)
        sql = extract_sql(raw)
        ok, err = validate_sql(sql, dialect)
        st.session_state.setdefault("history", [])
        st.session_state["history"].append({"question": question, "sql": sql, "ok": ok, "err": err})
    except Exception as e:
        st.error(f"Generation error: {e}")

if st.session_state.get("history"):
    st.subheader("Results")
    for i, item in enumerate(reversed(st.session_state["history"]), 1):
        st.markdown(f"**Q{i}:** {item['question']}")
        st.code(item["sql"], language="sql")
        if item["ok"]:
            st.success("SQL looks valid for this dialect ✔")
        else:
            st.error(f"Validation error: {item['err']}")

st.divider()
st.subheader("Optional: run SQL on uploaded CSVs (DuckDB)")
st.caption("Upload CSVs and reference them by filename (without .csv) in your SQL. Use dialect 'duckdb' if you execute here.")
files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
if files:
    con = duckdb.connect(database=":memory:")
    for f in files:
        df = pd.read_csv(f)
        table_name = os.path.splitext(f.name)[0]
        con.register(table_name, df)
        st.write(f"Registered table: `{table_name}` (rows={len(df)})")

    query = st.text_area("SQL to run (DuckDB)", height=160, placeholder="SELECT * FROM your_table LIMIT 5;")
    if st.button("Run SQL"):
        try:
            res = con.execute(query).fetchdf()
            st.dataframe(res)
        except Exception as e:
            st.error(f"Execution error: {e}")
