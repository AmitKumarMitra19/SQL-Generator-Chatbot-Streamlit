import os, re, json, requests
import sqlglot
import duckdb
import pandas as pd
import streamlit as st

# --------------------------
# Secrets/ENV-safe getters
# --------------------------
def safe_get_secret(name: str, default=None):
    # Try Streamlit secrets if available, else ENV, else default
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, default)

# Defaults (can be overridden by user input below)
DEFAULT_MODEL = safe_get_secret("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
DEFAULT_TOKEN = safe_get_secret("HF_TOKEN", None)

# --------------------------
# UI
# --------------------------
st.set_page_config(page_title="LLM → SQL", page_icon="🧠", layout="wide")
st.title("🧠→🗄️  LLM SQL Generator (Open-source LLM via HF)")

with st.sidebar:
    st.subheader("LLM Settings (no secrets required)")
    model = st.text_input("HF Model", value=DEFAULT_MODEL, help="e.g. Qwen/Qwen2.5-7B-Instruct")
    token = st.text_input("HF Token (paste here)", type="password", value=DEFAULT_TOKEN if DEFAULT_TOKEN else "")
    temp = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.slider("Max new tokens", 64, 2048, 512, 64)
    dialect = st.selectbox("SQL Dialect", ["postgres", "duckdb", "mysql", "sqlite", "bigquery", "snowflake"], index=0)
    st.caption("Tip: Paste your Hugging Face token here at runtime if you don't want to use Streamlit Secrets.")

HF_API_URL = st.text_input(
    "HF Inference API URL",
    value=f"https://api-inference.huggingface.co/models/{model}",
    help="Advanced: set your own endpoint (e.g., a public Space) if you want to avoid tokens."
)

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
        "question": "Total revenue by country for 2024?",
        "sql": """-- Assumption: revenue measured by orders.total; 2024 calendar year
SELECT c.country, SUM(o.total) AS total_revenue
FROM customers c
JOIN orders o ON o.customer_id = c.id
WHERE o.order_date >= DATE '2024-01-01'
  AND o.order_date <  DATE '2025-01-01'
GROUP BY c.country
ORDER BY total_revenue DESC;"""
    }
]

def build_chat_prompt(schema: str, question: str, dialect: str) -> str:
    parts = [f"[SYSTEM]\n{SQL_SYSTEM_PROMPT}\n[/SYSTEM]"]
    for ex in FEWSHOTS:
        parts.append(
            "[USER]\n"
            f"SQL DIALECT: {ex['dialect']}\nSCHEMA:\n{ex['schema'].strip()}\n"
            f"QUESTION: {ex['question']}\nReturn only SQL.\n[/USER]"
        )
        parts.append(f"[ASSISTANT]\n```sql\n{ex['sql'].strip()}\n```\n[/ASSISTANT]")
    parts.append(
        "[USER]\n"
        f"SQL DIALECT: {dialect}\nSCHEMA:\n{schema.strip()}\n\n"
        f"QUESTION: {question.strip()}\nReturn only SQL.\n[/USER]"
    )
    return "\n".join(parts)

def call_hf_inference(api_url: str, hf_token: str, prompt: str,
                      max_new_tokens=512, temperature=0.2, top_p=0.95, timeout=60):
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "return_full_text": False
        }
    }
    resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
    # Nice errors
    try:
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}") from e
    data = resp.json()
    if isinstance(data, list) and data and "generated_text" in data[0]:
        return data[0]["generated_text"]
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"Inference error: {data['error']}")
    return json.dumps(data)

def extract_sql(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    return m.group(1).strip() if m else text.strip()

def validate_sql(sql: str, dialect: str):
    try:
        sqlglot.parse_one(sql, read=dialect)
        return True, None
    except Exception as e:
        return False, str(e)

st.markdown("**Paste your schema (DDL or plain description), enter a question, and generate SQL.**")
schema = st.text_area("Schema", height=220, placeholder="Table users(id INT, name TEXT, created_at TIMESTAMP);\nTable events(id INT, user_id INT, type TEXT, ts TIMESTAMP);")
question = st.text_area("Question", height=120, placeholder="Monthly active users in 2025, by month.")

if st.button("Generate SQL"):
    try:
        if not token and "api-inference.huggingface.co" in HF_API_URL:
            st.warning("You are calling the HF Inference API without a token. Many models require auth and may return 401.")
        prompt = build_chat_prompt(schema, question, dialect)
        raw = call_hf_inference(HF_API_URL, token, prompt, max_new_tokens=max_tokens, temperature=temp)
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
