# AI Data Mapper — Faster Asset (3-tab UX)
# Tabs: 1) Connect to DB  2) Prepare Mapping  3) AI Mapping & Export
# Changes requested:
# - Keep 3 tabs (no structure change) + sidebar-only User Manual
# - Download file names readable: ai_mappings_<targetdb>_<targettable>_<YYYYMMDD-HHMMSS>.json/xlsx
# Kept UX improvements:
# - Multiselect defaults to [] and shows "N of M selected"
# - No "None" printed anywhere (sanitized text/dataframes; removed tqdm)
# - Auto-advance to next tab after successful step; unlock tabs after a run to allow going back
# - Model fixed to gpt-5

import os, io, re, json, warnings, datetime
import pandas as pd
import streamlit as st
import pymssql
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from openai import OpenAI

warnings.filterwarnings("ignore") 
st.set_page_config(page_title="AI Data Mapper — Faster Asset", layout="wide")

# =============== Sidebar: User Manual only ===============
with st.sidebar:
    st.header("📘 User Manual")
    st.markdown("""
**What it does**  
Matches **source** columns to your **target** columns and drafts **SQL transforms**.

---

### 1) Connect to DB
- Enter **Host/IP**, **Port** (often `1433`), **User**, **Password**.
- Click **Connect**. You’ll move to the next tab on success.
- Optional: **Run LLM Health Check** to verify OpenAI key & quota.

### 2) Prepare Mapping
1. **Load DB list** → pick **Source DB** and **Target DB**  
2. **Load table lists** → choose your **Target table**  
3. **Preview target table** (shows rows)  
4. **Select target columns** to map (starts empty)  
5. **Prepare source columns** (builds search space)

### 3) AI Mapping & Export
- Click **Run column near-matching & AI mapping**  
- Review per-column results; download **JSON** or **Excel**

---

### Troubleshooting
- **DB connect fails:** verify network/firewall, SQL auth, port 1433.  
- **DB list error:** press “Rerun” from the Streamlit menu.  
- **OpenAI 429/quota:** add credit or set `OPENAI_API_KEY`.  
- **App stops:** read terminal logs; then “Rerun”.
""")

# =================== CSS ===================
st.markdown("""
<style>
:root{ --card-bg:#fff; --muted:#64748b; --ok:#16a34a; --warn:#d97706; --err:#dc2626; }
.step-card{ background:var(--card-bg); border:1px solid #e5e7eb; border-radius:14px; padding:16px;
            box-shadow:0 8px 24px rgba(0,0,0,.04); margin:14px 0; }
.step-head{ display:flex; align-items:center; gap:10px; margin-bottom:8px;}
.step-num{ background:#111827; color:#fff; font-weight:700; border-radius:10px; padding:2px 10px;}
.badge{ font-size:.85rem; padding:2px 8px; border-radius:999px; border:1px solid #e5e7eb; color:#111827;}
.badge.ok{ background:#ecfdf5; border-color:#bbf7d0; color:#065f46;}
.badge.warn{ background:#fffbeb; border-color:#fde68a; color:#92400e;}
.badge.err{ background:#fef2f2; border-color:#fecaca; color:#7f1d1d;}
.result-card{ border:1px solid #e5e7eb; border-radius:12px; padding:12px; margin:8px 0; background:#fafafa;}
.result-title{ font-weight:700;}
.code{ background:#0b1220; color:#e6edf3; padding:10px; border-radius:8px; overflow:auto;
       font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Courier New", monospace;}
button[role="tab"].locked { opacity:.45; pointer-events:none; }
</style>
""", unsafe_allow_html=True)

# =================== UI helpers ===================
def step_card(num:int, title:str, right_badge:str="", badge_class:str=""):
    st.markdown(f"""
<div class="step-card"><div class="step-head">
  <div class="step-num">{num}</div>
  <div style="font-weight:800;font-size:1.05rem;">{title}</div>
  <div style="flex:1"></div>
  {f'<span class="badge {badge_class}">{right_badge}</span>' if right_badge else ''}
</div>
""", unsafe_allow_html=True)
def step_card_end(): st.markdown("</div>", unsafe_allow_html=True)

def result_card(title:str, confidence:float|None=None):
    pill = ""
    if confidence is not None:
        try:
            c = float(confidence)
            cls = "ok" if c>=8 else "warn" if c>=5 else "err"
            pill = f'<span class="badge {cls}">{c:.1f}/10</span>'
        except: pass
    st.markdown(f"<div class='result-card'><div class='result-title'>{title} {pill}</div>", unsafe_allow_html=True)
def result_card_end(): st.markdown("</div>", unsafe_allow_html=True)

def nn(x):
    if x is None: return ""
    if isinstance(x, float) and (x!=x): return ""
    s = str(x)
    return "" if s == "None" else s

def df_clean(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty: return pd.DataFrame()
    return df.replace({None:""}).fillna("")

def _switch_tab_js(label: str):
    st.components.v1.html(f"""
    <script>
      const wanted = `{label}`.trim();
      const tabs = window.parent.document.querySelectorAll('button[role="tab"]');
      for (const t of tabs) {{ if (t.innerText.trim() === wanted) {{ t.click(); break; }} }}
    </script>
    """, height=0, scrolling=False)

def lock_earlier_tabs_upto(label: str):
    st.components.v1.html(f"""
    <script>
      const tabs = window.parent.document.querySelectorAll('button[role="tab"]');
      let lock = true;
      for (const t of tabs) {{
        const name = t.innerText.trim();
        if (name === `{label}`.trim()) lock = false;
        if (lock) t.classList.add('locked');
      }}
    </script>
    """, height=0, scrolling=False)

def unlock_all_tabs():
    st.components.v1.html("""
    <script>
      const tabs = window.parent.document.querySelectorAll('button[role="tab"]');
      for (const t of tabs) { t.classList.remove('locked'); }
    </script>
    """, height=0, scrolling=False)

def safe_status(ph, text: str):
    if text and text.strip(): ph.markdown(text)

# =================== Config ===================
DEFAULT_IP = ""
DEFAULT_PORT = 
DEFAULT_USER = ""
DEFAULT_PASSWORD = ""

TOP_N_EXAMPLES = 50
TOP_K_PREFILTER = 10
USE_DESC_ONLY_FOR_EMBEDDING = True

# =================== OpenAI (gpt-5) ===================
def get_openai_key(): return os.getenv("OPENAI_API_KEY")
@st.cache_resource(show_spinner=False)
def get_openai_client():
    key = get_openai_key()
    return OpenAI(api_key=key) if key else None

def llm_health_check():
    client = get_openai_client()
    if not client: return False, "No OpenAI API key found."
    try:
        r = client.chat.completions.create(model="gpt-5", messages=[{"role":"user","content":"ping"}])
        _ = r.choices[0].message.content
        return True, "OpenAI reachable."
    except Exception as e:
        return False, str(e)

# =================== DB helpers ===================
@st.cache_resource(show_spinner=False)
def get_db_conn(host: str, port: int, user: str, password: str):
    return pymssql.connect(host=host, port=int(port), user=user, password=password,
                           login_timeout=10, timeout=30, as_dict=True)

def run_sql(_conn, sql: str, params=None):
    with _conn.cursor() as cur:
        cur.execute(sql, params) if params else cur.execute(sql)
        try: rows = cur.fetchall()
        except pymssql.ProgrammingError: rows = []
    return rows

@st.cache_data(show_spinner=False, ttl=900)
def list_databases(_conn):
    return [r["name"] for r in run_sql(_conn, "SELECT name FROM sys.databases;")]

@st.cache_data(show_spinner=False, ttl=900)
def list_tables(_conn, dbname):
    return run_sql(_conn, f"SELECT TABLE_SCHEMA, TABLE_NAME FROM [{dbname}].INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE';")

@st.cache_data(show_spinner=False, ttl=900)
def list_columns(_conn, dbname):
    return run_sql(_conn, f"SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE, IS_NULLABLE FROM [{dbname}].INFORMATION_SCHEMA.COLUMNS;")

@st.cache_data(show_spinner=False, ttl=900)
def sample_rows(_conn, dbname, schema, table, top_n=50):
    sql = f"SELECT {'TOP '+str(int(top_n)) if str(top_n).lower()!='all' else ''} * FROM [{dbname}].[{schema}].[{table}];"
    return run_sql(_conn, sql)

# =================== Cache paths ===================
def safe_filename(name: str) -> str:
    return re.sub(r'[<>:\"/\\\\|?*]', '_', name or "")

def get_desc_cache_path(db_name: str, schema_name: str, table_name: str, role: str) -> str:
    base = os.path.join(".", "temp", "descriptions", role, safe_filename(db_name), safe_filename(schema_name))
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, safe_filename(f"{table_name}.json"))

# =================== Embedding & LLM utils ===================
@st.cache_resource(show_spinner=False)
def get_embedder(): return SentenceTransformer("all-MiniLM-L6-v2")

def getNearTextsEncodingsUpdated(pivot_texts, texts, src, model):
    tgt = model.encode(pivot_texts)
    scores = cosine_similarity(tgt, src)[0]
    top_idx = scores.argsort()[::-1][:TOP_K_PREFILTER]
    return [texts[i] for i in top_idx], [float(scores[i]) for i in top_idx]

def _extract_json_block(text: str) -> str:
    m = re.search(r'(\{.*\}|\[.*\])', text or "", flags=re.S)
    return m.group(1) if m else "[]"

def _parse_llm_json(raw: str):
    try:
        return json.loads(_extract_json_block(raw))
    except Exception:
        return []

def genColumnDescriptionsDF(db_name: str, schema_name: str, table_name: str, df: pd.DataFrame, role: str = "target"):
    cache_file = get_desc_cache_path(db_name, schema_name, table_name, role)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f: return json.load(f)
        except: pass

    cols_payload = []
    if isinstance(df, pd.DataFrame) and not df.empty:
        for col in df.columns:
            vals, seen = [], set()
            for v in list(df[col].values)[:200]:
                sv = str(v)[:40]
                if sv in ("", "NONE", "None", "NaT"): continue
                if sv not in seen:
                    seen.add(sv); vals.append(sv)
                if len(vals) >= 10: break
            cols_payload.append({"table_name": table_name, "column_name": col, "sample_data": vals})

    if not cols_payload:
        with open(cache_file, "w", encoding="utf-8") as f: json.dump([], f, indent=2, ensure_ascii=False)
        return []

    sample_output = [{"column_name": "<name>", "description": "<1–2 sentences, max 35 words>"}]
    prompt = f"""
You are a data engineer documenting database columns in the fleet management domain.
For each column object in the input list, write a concise definition of the data’s business meaning.

Input (array):
{json.dumps(cols_payload, ensure_ascii=False)}

Return ONLY raw JSON as:
{json.dumps(sample_output, ensure_ascii=False)}

Rules:
- No repetition of names/values; focus on meaning.
- If unclear, add a cautious placeholder (“likely …; confirm with SME”).
- Max 35 words per description.
""".strip()

    client = get_openai_client()
    try:
        if not client:
            out = [{"column name": c, "description": ""} for c in (df.columns if isinstance(df, pd.DataFrame) else [])]
        else:
            resp = client.chat.completions.create(model="gpt-5", messages=[{"role": "user", "content": prompt}])
            raw = resp.choices[0].message.content or "[]"
            parsed = _parse_llm_json(raw)
            out = []
            if isinstance(parsed, list):
                by_name = { (c.get("column_name") or ""): c for c in parsed if isinstance(c, dict) }
                for col in df.columns:
                    desc = nn((by_name.get(col, {}) or {}).get("description", ""))
                    out.append({"column name": col, "description": desc})
            else:
                out = [{"column name": c, "description": ""} for c in df.columns]
        with open(cache_file, "w", encoding="utf-8") as f: json.dump(out, f, indent=2, ensure_ascii=False)
        return out
    except Exception as e:
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump([{"error": f"Description generation failed: {nn(e)}"}], f, indent=2, ensure_ascii=False)
        except: pass
        return [{"error": f"Description generation failed: {nn(e)}"}]

def validateAIMappings(response, all_near_objs):
    try:
        out, near_tables, near_cols = [], {o["TABLE_NAME"] for o in all_near_objs}, {o["COLUMN_NAME"] for o in all_near_objs}
        for m in response:
            rels = m.get("possible_relationships", [])
            filt = [r for r in rels if (r.get("source_table_name") in near_tables and r.get("source_table_column") in near_cols)]
            mm = dict(m); mm["possible_relationships"] = filt; out.append(mm)
        return out
    except Exception:
        return response

# =================== Session defaults ===================
for k, v in {
    "conn": None,
    "databases": [],
    "source_db": None, "target_db": None, "target_table": None,
    "tables_source": [], "tables_target": [],
    "df_target_preview": None, "src_objects": {},
    "near_matches_df": None, "ai_mappings": {},
    "selected_target_columns": [],
    "step1_done": False, "step2_done": False, "run_done": False
}.items(): st.session_state.setdefault(k, v)

# =================== Tabs (3) ===================
label1, label2, label3 = "🔌 Connect to DB", "🧭 Prepare Mapping", "🤖 AI Mapping & Export"
tab1, tab2, tab3 = st.tabs([label1, label2, label3])

# ---------- Tab 1 ----------
with tab1:
    step_card(1, "SQL Server connection",
              right_badge=("Done" if st.session_state.step1_done else "Pending"),
              badge_class=("ok" if st.session_state.step1_done else "warn"))
    c1, c2, c3, c4 = st.columns([1.2, 0.6, 1, 1])
    with c1: host = st.text_input("Host / IP", value=os.environ.get("DB_HOST", DEFAULT_IP))
    with c2: port = st.number_input("Port", value=int(os.environ.get("DB_PORT", DEFAULT_PORT)), step=1)
    with c3: user = st.text_input("User", value=os.environ.get("DB_USER", DEFAULT_USER))
    with c4: password = st.text_input("Password", type="password", value=os.environ.get("DB_PASSWORD", DEFAULT_PASSWORD))

    cL, cR = st.columns([0.25, 0.75])
    with cL:
        if st.button("Connect"):
            try:
                st.session_state.conn = get_db_conn(host, port, user, password)
                st.session_state.step1_done = True
                st.success("✅ Connected to SQL Server.")
                _switch_tab_js(label2)  # auto-advance
            except Exception as e:
                st.session_state.conn = None
                st.session_state.step1_done = False
                st.error(f"Connection failed: {nn(e)}")
    with cR:
        st.info(f"Connection status: **{'Connected' if st.session_state.conn else 'Not connected'}**  | Host: `{nn(host)}`  | Port: `{nn(port)}`")

    hcL, _ = st.columns([0.25, 0.75])
    with hcL:
        if st.button("Run LLM Health Check"):
            ok, msg = llm_health_check()
            (st.success if ok else st.error)(msg)
    step_card_end()

if st.session_state.step1_done:
    lock_earlier_tabs_upto(label2)

# ---------- Tab 2 ----------
with tab2:
    step_card(2, "Prepare search space & preview target",
              right_badge=("Ready" if st.session_state.step2_done else ("Connected" if st.session_state.conn else "Connect first")),
              badge_class=("ok" if st.session_state.step2_done else ("warn" if st.session_state.conn else "err")))

    if not st.session_state.conn:
        st.warning("Please connect in the **Connect to DB** tab first.")
        step_card_end()
    else:
        st.subheader("Load databases")
        if st.button("Load DB list"):
            try:
                st.session_state.databases = list_databases(st.session_state.conn)
                st.success(f"Loaded {len(st.session_state.databases)} database(s).")
            except Exception as e:
                st.error(f"Failed to list databases: {nn(e)}")

        if len(st.session_state.databases) == 0:
            st.info("No DBs loaded yet."); step_card_end()
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.session_state.source_db = st.selectbox("Source DB", st.session_state.databases, index=None, placeholder="Pick a source DB")
            with c2:
                st.session_state.target_db = st.selectbox("Target DB", st.session_state.databases, index=None, placeholder="Pick a target DB")

            if not (st.session_state.source_db and st.session_state.target_db):
                st.warning("Select both Source and Target DBs to continue."); step_card_end()
            else:
                st.subheader("Load tables (cached)")
                if st.button("Load table lists"):
                    try:
                        st.session_state.tables_source = list_tables(st.session_state.conn, st.session_state.source_db)
                        st.session_state.tables_target = list_tables(st.session_state.conn, st.session_state.target_db)
                        st.success(f"Source tables: {len(st.session_state.tables_source)} | Target tables: {len(st.session_state.tables_target)}")
                    except Exception as e:
                        st.error(f"Failed to load tables: {nn(e)}")

                if len(st.session_state.tables_target) == 0:
                    st.info("Target tables not loaded yet."); step_card_end()
                else:
                    tgt_names = [r["TABLE_NAME"] for r in st.session_state.tables_target]
                    st.session_state.target_table = st.selectbox("Target table", tgt_names, index=None, placeholder="Pick a target table")

                    if st.session_state.target_table is None:
                        st.warning("Pick a Target table."); step_card_end()
                    else:
                        st.subheader("Preview target (descriptions cached)")
                        if st.button("Preview target table"):
                            try:
                                tgt_row = next(r for r in st.session_state.tables_target if r["TABLE_NAME"] == st.session_state.target_table)
                                schema = tgt_row["TABLE_SCHEMA"]
                                data = sample_rows(st.session_state.conn, st.session_state.target_db, schema, st.session_state.target_table, top_n='all')
                                df_prev = df_clean(pd.DataFrame(data))
                                st.session_state.df_target_preview = df_prev
                                _ = genColumnDescriptionsDF(st.session_state.target_db, schema, st.session_state.target_table, df_prev, role="target")
                                st.success("Preview loaded and target descriptions cached.")
                            except Exception as e:
                                st.error(f"Failed to preview target table: {nn(e)}")

                        df_prev = df_clean(st.session_state.get("df_target_preview"))
                        if not df_prev.empty:
                            st.dataframe(df_prev, use_container_width=True, height=280)
                            st.caption(f"Rows: {len(df_prev)} | Columns: {len(df_prev.columns)}")

                            # Multiselect default [] with counts
                            st.session_state.selected_target_columns = st.multiselect(
                                "Select target columns for AI mapping (starts empty)",
                                options=list(df_prev.columns),
                                default=[],
                                help="Select one or more columns to map."
                            )
                            st.info(f"**Selected:** {len(st.session_state.selected_target_columns)} of {len(df_prev.columns)} column(s)")

                        st.subheader("Prepare source columns (cached)")
                        if st.button("Prepare source columns"):
                            try:
                                all_cols = list_columns(st.session_state.conn, st.session_state.source_db)
                                src_tables = list_tables(st.session_state.conn, st.session_state.source_db)

                                src_objects = {}
                                progress = st.progress(0); status = st.empty()
                                total = max(len(src_tables), 1)

                                for i, t in enumerate(src_tables):
                                    tname, tschema = t["TABLE_NAME"], t["TABLE_SCHEMA"]
                                    rows = sample_rows(st.session_state.conn, st.session_state.source_db, tschema, tname, top_n=TOP_N_EXAMPLES*3)
                                    df_src = df_clean(pd.DataFrame(rows))

                                    table_column_descriptions = genColumnDescriptionsDF(
                                        st.session_state.source_db, tschema, tname, df_src, role="source"
                                    )
                                    for d in table_column_descriptions:
                                        if isinstance(d, dict): d["description"] = nn(d.get("description",""))

                                    tcols = [c for c in all_cols if c["TABLE_NAME"] == tname]
                                    key = f"{tschema}.{tname}"
                                    src_objects[key] = {
                                        "TABLE_SCHEMA": tschema, "TABLE_NAME": tname,
                                        "columns": tcols, "data": rows,
                                        "column_descriptions": table_column_descriptions
                                    }
                                    progress.progress(int((i+1) * 100 / total))
                                    safe_status(status, f"**Prepared:** {i+1}/{total} tables")

                                status.empty()
                                st.session_state.src_objects = src_objects
                                st.success(f"Prepared {len(src_objects)} source tables.")
                                st.session_state.step2_done = True
                                _switch_tab_js(label3)  # auto-advance to AI tab
                            except Exception as e:
                                st.error(f"Failed to prepare source columns: {nn(e)}")
                                st.session_state.step2_done = False

                        step_card_end()

if st.session_state.step2_done:
    lock_earlier_tabs_upto(label3)

# ---------- Tab 3 ----------
with tab3:
    step_card(3, "AI Mapping & Export",
              right_badge=("Ready" if (st.session_state.get("src_objects") and st.session_state.get("df_target_preview") is not None) else "Pending"),
              badge_class=("ok" if (st.session_state.get("src_objects") and st.session_state.get("df_target_preview") is not None) else "warn"))

    df_target = df_clean(st.session_state.get("df_target_preview"))
    src_objs = st.session_state.get("src_objects")

    if not (isinstance(src_objs, dict) and len(src_objs) > 0):
        st.warning("Source columns are not prepared yet. Please use **Prepare Mapping** tab.")
        step_card_end()
        st.stop()
    if df_target.empty:
        st.warning("Target preview not loaded yet. Please use **Prepare Mapping** tab.")
        step_card_end()
        st.stop()

    progress = st.progress(0); status = st.empty(); live_results = st.container()

    if st.button("Run column near-matching & AI mapping"):
        # Use *exactly* the selected target columns
        selected_cols = st.session_state.get("selected_target_columns") or []
        if len(selected_cols) == 0:
            st.error("Select at least one target column in the previous tab.")
            st.stop()

        target_table = st.session_state.target_table
        tgt_cols = [c for c in list(df_target.columns) if c in selected_cols]
        st.success(f"Mapping **{len(tgt_cols)}** selected column(s).")

        # Build source text universe
        source_texts, source_meta = [], []
        for _, sobj in src_objs.items():
            tschema, tname = sobj["TABLE_SCHEMA"], sobj["TABLE_NAME"]
            descs = sobj.get('column_descriptions', []) or []
            desc_map = {d.get("column name",""): nn(d.get("description","")) for d in descs if isinstance(d, dict)}
            for col in sobj["columns"]:
                colname = col["COLUMN_NAME"]; dtype = nn(col.get("DATA_TYPE",""))
                desc = desc_map.get(colname, "")
                if USE_DESC_ONLY_FOR_EMBEDDING:
                    txt = f"{tschema}.{tname}.{colname} :: {dtype} :: {desc}"
                else:
                    rows = [nn(r.get(colname,"")) for r in sobj["data"][:TOP_N_EXAMPLES]]
                    txt = f"{tschema}.{tname}.{colname} :: {dtype} :: {desc} :: " + " ".join(rows)
                source_texts.append(txt)
                source_meta.append({"TABLE_SCHEMA": tschema, "TABLE_NAME": tname,
                                    "COLUMN_NAME": colname, "DATA_TYPE": dtype, "DESC": desc})

        model = get_embedder()
        source_texts_encodings = model.encode(source_texts)

        ai_mappings, near_rows = {}, []
        client = get_openai_client()

        # Target desc cache
        tgt_row = next(r for r in st.session_state.tables_target if r["TABLE_NAME"] == target_table)
        target_schema = tgt_row["TABLE_SCHEMA"]
        tgt_desc_list = genColumnDescriptionsDF(st.session_state.target_db, target_schema, target_table, df_target, role="target")
        tgt_desc_map = {d.get("column name",""): nn(d.get("description","")) for d in (tgt_desc_list or []) if isinstance(d, dict)}
        tgt_cols_meta_all = list_columns(st.session_state.conn, st.session_state.target_db)

        total = max(len(tgt_cols),1); done = 0
        mapped_ok, mapped_err = [], []
        quota_exhausted = False

        for tgt_col in tgt_cols:
            safe_status(status, f"**Generating mapping for:** `{tgt_col}` …")

            tgt_meta = next((m for m in tgt_cols_meta_all if m["TABLE_NAME"] == target_table and m["COLUMN_NAME"] == tgt_col), {})
            tgt_dtype = nn(tgt_meta.get("DATA_TYPE", "nvarchar"))
            tgt_desc = tgt_desc_map.get(tgt_col, "")

            ex_vals = [nn(v) for v in df_target[tgt_col].tolist()[:TOP_N_EXAMPLES]]
            tgt_text = (f"{target_table}.{tgt_col} :: {tgt_dtype} :: {tgt_desc}"
                        if USE_DESC_ONLY_FOR_EMBEDDING else
                        f"{target_table}.{tgt_col} :: {tgt_dtype} :: {tgt_desc} :: " + " ".join(ex_vals))

            near_texts, scores = getNearTextsEncodingsUpdated([tgt_text], source_texts, source_texts_encodings, model)

            all_near_objs = []
            for ttxt, s in zip(near_texts, scores):
                idx = source_texts.index(ttxt)
                meta = source_meta[idx]
                try:
                    key = f"{meta['TABLE_SCHEMA']}.{meta['TABLE_NAME']}"
                    rows_src = src_objs[key]["data"][:TOP_N_EXAMPLES]
                    rows_src_vals = [nn(r.get(meta["COLUMN_NAME"], "")) for r in rows_src]
                except Exception:
                    rows_src_vals = []
                all_near_objs.append({
                    "TABLE_NAME": f"{meta['TABLE_SCHEMA']}.{meta['TABLE_NAME']}",
                    "COLUMN_NAME": meta['COLUMN_NAME'],
                    "DATA_TYPE": meta['DATA_TYPE'],
                    "top_n_data": rows_src_vals,
                    "score": f"{s:.4f}",
                    "description": meta['DESC']
                })
                near_rows.append({"column": tgt_col, "matches": f"{meta['TABLE_SCHEMA']}.{meta['TABLE_NAME']}.{meta['COLUMN_NAME']}|{s:.4f}"})

            if client and not quota_exhausted:
                sample_output = [{
                    "target_table_name": target_table,
                    "target_table_column_name": tgt_col,
                    "target_table_column_description": "",
                    "transformation_sql_script": "",
                    "possible_relationships": [
                        {"source_table_name": "", "source_table_column": "",
                         "source_table_column_description": "", "score": "", "reasoning":""}
                    ]
                }]

                close_txt = ""
                for o in all_near_objs:
                    close_txt += (
                        f"\nSource Table name: {o['TABLE_NAME']}"
                        f"\nSource Column name: {o['COLUMN_NAME']} \nDatatype: {o['DATA_TYPE']}"
                        f"\nColumn Description: {o['description']}\n"
                        f"\nTop rows data: {o['top_n_data']}\n"
                    )

                tgt_txt = (
                    f"Target Table name: {target_table}"
                    f"\nTarget Column name: {tgt_col} \nDatatype: {tgt_dtype}"
                    f"\nColumn Description: {tgt_desc}"
                    f"\nTop rows data: {ex_vals}"
                )

                prompt = f"""
Role: You are a Data Engineer expert in source→target mapping and SQL transformations.

Task: Given the target column & sample values, and a shortlist of likely source columns (with samples),
produce accurate mappings and a transformation SQL (if needed). Return ONLY raw JSON in this format:

{sample_output}

Target:
{tgt_txt}

Candidate sources:
{close_txt}

Rules:
- Prefer single-source columns when obvious; allow multi-source if needed.
- Cover type conversion & null handling in SQL when relevant.
- Use real column and table names.
- Add a 1–10 confidence score per relationship.
- Do not provide any other explanation or information, just the mapping in the specified format.
- Not even ```json or any other extra characters. Just raw json format
""".strip()

                try:
                    resp = client.chat.completions.create(
                        model="gpt-5", stream=False,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    raw = resp.choices[0].message.content or "[]"
                    response = _parse_llm_json(raw)
                    response = validateAIMappings(response, all_near_objs)
                    ai_mappings[tgt_col] = response
                    mapped_ok.append(tgt_col)
                except Exception as e:
                    msg = str(e)
                    if "insufficient_quota" in msg or "You exceeded your current quota" in msg:
                        quota_exhausted = True
                        st.error("OpenAI quota exhausted. Stopping further calls.")
                    ai_mappings[tgt_col] = [{"note": f"LLM error: {nn(msg)}"}]
                    mapped_err.append(tgt_col)
            else:
                ai_mappings[tgt_col] = [{"note": "OpenAI key not set; skipping LLM mapping."}]
                mapped_err.append(tgt_col)

            with live_results:
                for m in (ai_mappings.get(tgt_col) or []):
                    conf = None
                    try:
                        conf_vals = [float(x.get("score", 0)) for x in m.get("possible_relationships", [])
                                     if str(x.get("score","")).replace('.','',1).isdigit()]
                        conf = max(conf_vals) if conf_vals else None
                    except: pass
                    result_card(f"🎯 {tgt_col}", conf)
                    rels = m.get("possible_relationships", [])
                    if rels:
                        st.dataframe(pd.DataFrame(rels).replace({None:""}).fillna(""),
                                     use_container_width=True, height=180)
                    sql_txt = m.get("transformation_sql_script", "")
                    if sql_txt:
                        st.markdown(f"<div class='code'><pre>{nn(sql_txt)}</pre></div>", unsafe_allow_html=True)
                    result_card_end()

            done += 1
            progress.progress(int(done * 100 / total))
            safe_status(status, f"**Processing:** {done}/{total}")

        st.session_state.ai_mappings = ai_mappings
        st.session_state.near_matches_df = pd.DataFrame(near_rows)
        status.empty()
        st.success(f"Completed. **Mapped {len(mapped_ok)} / {len(tgt_cols)}** column(s)."
                   + (f" Skipped/errored: {', '.join(mapped_err)}" if mapped_err else ""))

        # After a run, allow going back to earlier tabs to re-select DBs if needed
        st.session_state.run_done = True
        unlock_all_tabs()

    # ========== Exports with readable names ==========
    if st.session_state.get("ai_mappings"):
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        td = safe_filename(st.session_state.target_db or "targetdb")
        tt = safe_filename(st.session_state.target_table or "table")
        json_name = f"ai_mappings_{td}_{tt}_{ts}.json"
        xlsx_name = f"ai_mappings_{td}_{tt}_{ts}.xlsx"

        c1, c2 = st.columns([1,1])
        with c1:
            st.download_button(
                "⬇️ Export AI mappings (JSON)",
                data=json.dumps(st.session_state.ai_mappings, indent=2),
                file_name=json_name,
                mime="application/json"
            )
        with c2:
            objs = []
            for col, mappings in st.session_state.ai_mappings.items():
                for mapping in (mappings or []):
                    base = {
                        "Target table name": mapping.get('target_table_name', st.session_state.target_table),
                        "Target column name": col,
                        "Target column description": mapping.get('target_table_column_description', "")
                    }
                    for rel in mapping.get('possible_relationships', []):
                        r = base.copy()
                        r["Source table name"] = nn(rel.get('source_table_name', ""))
                        r["Source column name"] = nn(rel.get('source_table_column', ""))
                        r["Source column description"] = nn(rel.get('source_table_column_description', ""))
                        r["Mapping score"] = nn(rel.get('score', ""))
                        r["Reasoning"] = nn(rel.get('reasoning', ""))
                        objs.append(r)
            excel_df = pd.DataFrame(objs or [{"note":"no results"}]).replace({None:""}).fillna("")
            buff = io.BytesIO()
            with pd.ExcelWriter(buff, engine='xlsxwriter') as w:
                sheet = (st.session_state.target_table or "Results")[:31]
                excel_df.to_excel(w, sheet_name=sheet, index=False)
            st.download_button("⬇️ Download as Excel", data=buff.getvalue(),
                               file_name=xlsx_name,
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    step_card_end()