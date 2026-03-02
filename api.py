import os
import re
import json
from datetime import datetime
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import psycopg2
import requests
from psycopg2 import sql
from psycopg2.extras import Json, RealDictCursor

DB_CONFIG = {
    "host": "localhost",
    "database": "postgis_36_sample",
    "user": "postgres",
    "password": "root",
    "port": "5432",
}

MAX_TABLES = 4
COUNT_KEYWORDS = [
    "how many",
    "count",
    "number of",
    "total",
    "record count",
    "rows count",
]

ROLE_CONFIG = {
    "viewer": {"max_limit": 50, "max_tables": 2},
    "analyst": {"max_limit": 200, "max_tables": 4},
    "admin": {"max_limit": 500, "max_tables": 4},
}

DATE_TYPES = {
    "date",
    "timestamp without time zone",
    "timestamp with time zone",
}

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "90"))


class PromptRequest(BaseModel):
    prompt: str = Field(min_length=3)
    limit: int = Field(default=50, ge=1, le=500)
    selected_tables: list[str] = Field(default_factory=list)
    role: str = Field(default="viewer")
    user_id: str = Field(default="anonymous")


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def get_model_reference() -> dict[str, Any]:
    return {
        "provider": "ollama",
        "model": OLLAMA_MODEL,
        "base_url": OLLAMA_BASE_URL,
        "pull_command": "ollama pull llama3.1:8b-instruct-q4_K_M",
    }


def ollama_status() -> tuple[bool, str | None]:
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        names = {m.get("name") for m in data.get("models", [])}
        if OLLAMA_MODEL not in names:
            return False, f"Model not found in Ollama: {OLLAMA_MODEL}"
        return True, None
    except Exception as exc:
        return False, str(exc)


def extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Ollama output did not contain valid JSON")
    return json.loads(text[start : end + 1])


def validate_sql(sql_query: str, allowed_tables: list[str]) -> str:
    cleaned = sql_query.strip().strip("`")
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1]

    lowered = cleaned.lower()
    if not lowered.startswith("select "):
        raise ValueError("Only SELECT queries are allowed")
    if any(k in lowered for k in ["insert ", "update ", "delete ", "drop ", "alter ", "truncate "]):
        raise ValueError("Unsafe SQL keyword detected")

    table_hits = re.findall(r"(?:from|join)\s+(?:public\.)?([a-zA-Z_][a-zA-Z0-9_]*)", lowered)
    if not table_hits:
        raise ValueError("Generated SQL did not reference a table")
    if any(t not in allowed_tables for t in table_hits):
        raise ValueError("Generated SQL referenced non-allowed table")

    return cleaned


def enforce_limit(sql_query: str, effective_limit: int) -> str:
    if re.search(r"\blimit\s+\d+\b", sql_query, flags=re.IGNORECASE):
        return re.sub(r"\blimit\s+\d+\b", f"LIMIT {effective_limit}", sql_query, flags=re.IGNORECASE)
    return f"{sql_query} LIMIT {effective_limit}"


def generate_sql_with_ollama(
    prompt: str, active_tables: list[str], role_name: str, effective_limit: int
) -> dict[str, Any]:
    schema = get_table_schema()
    schema_view = {t: list(schema.get(t, {}).keys()) for t in active_tables}
    role_cap = ROLE_CONFIG[role_name]["max_limit"]

    prompt_text = (
        "You are a PostgreSQL SQL planner. Return only strict JSON with keys: "
        "intent, tables, sql, confidence. "
        "Allowed intents: count_records, table_data_retrieval, state_filtered_data.\n"
        f"Allowed tables: {active_tables}\n"
        f"Schema: {json.dumps(schema_view)}\n"
        f"Role: {role_name}; role_cap={role_cap}; effective_limit={effective_limit}\n"
        "Generate one safe SELECT query only. If count asked, use COUNT and alias as total_records.\n"
        "Output JSON only."
        f"\nUser prompt: {prompt}"
    )

    last_error: str | None = None
    for _ in range(3):
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt_text,
                    "stream": False,
                    "options": {"temperature": 0},
                                    },
                #timeout=OLLAMA_TIMEOUT,
            )
            resp.raise_for_status()
            body = resp.json()
            parsed = extract_json_object(body.get("response", ""))
            intent = parsed.get("intent", "table_data_retrieval")
            tables = [t for t in parsed.get("tables", []) if t in active_tables] or active_tables[:1]
            sql_query = validate_sql(parsed.get("sql", ""), active_tables)
            if intent != "count_records":
                sql_query = enforce_limit(sql_query, effective_limit)
            return {
                "intent": intent,
                "tables": tables[:MAX_TABLES],
                "sql": sql_query,
                "confidence": parsed.get("confidence"),
            }
        except Exception as exc:
            last_error = str(exc)
            prompt_text += f"\nPrevious output invalid: {last_error}\nReturn valid JSON."

    raise ValueError(f"Ollama SQL generation failed: {last_error}")


def execute_generated_sql(sql_query: str) -> list[dict[str, Any]]:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql_query)
            return cur.fetchall()


@lru_cache(maxsize=1)
def get_table_schema() -> dict[str, dict[str, str]]:
    schema: dict[str, dict[str, str]] = {}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
                ORDER BY table_name, ordinal_position;
                """
            )
            for table_name, column_name, data_type in cur.fetchall():
                schema.setdefault(table_name, {})[column_name] = data_type
    return schema


def get_table_metadata() -> dict[str, list[str]]:
    schema = get_table_schema()
    return {table: list(columns.keys()) for table, columns in schema.items()}


def configured_allowed_tables(all_tables: list[str]) -> list[str]:
    env_value = os.getenv("APP_ALLOWED_TABLES", "").strip()
    if env_value:
        requested = [t.strip() for t in env_value.split(",") if t.strip()]
        valid = [t for t in requested if t in all_tables]
        if valid:
            return valid[:MAX_TABLES]
    return all_tables[:MAX_TABLES]


def normalize_role(role: str) -> str:
    role_name = role.lower().strip()
    if role_name not in ROLE_CONFIG:
        return "viewer"
    return role_name


def choose_active_tables(selected_tables: list[str] | None, role: str) -> list[str]:
    table_names = list(get_table_schema().keys())
    if not table_names:
        raise HTTPException(status_code=500, detail="No public tables found in database")

    default_allowed = configured_allowed_tables(table_names)
    role_max_tables = ROLE_CONFIG[role]["max_tables"]

    if not selected_tables:
        return default_allowed[:role_max_tables]

    valid_selected = [t for t in selected_tables if t in table_names]
    if not valid_selected:
        return default_allowed[:role_max_tables]

    return valid_selected[: min(MAX_TABLES, role_max_tables)]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def detect_count_intent(prompt: str) -> bool:
    lowered = normalize_text(prompt)
    return any(keyword in lowered for keyword in COUNT_KEYWORDS)


def extract_limit(prompt: str, fallback: int) -> int:
    match = re.search(r"\b(\d{1,3})\b", prompt)
    if not match:
        return fallback
    value = int(match.group(1))
    return max(1, min(500, value))


def score_table_match(prompt: str, table_name: str) -> float:
    prompt_norm = normalize_text(prompt)
    table_norm = table_name.lower()
    if table_norm in prompt_norm:
        return 1.0

    table_tokens = table_norm.split("_")
    token_hits = sum(1 for token in table_tokens if token in prompt_norm)
    token_score = token_hits / max(1, len(table_tokens))
    fuzzy_score = SequenceMatcher(None, prompt_norm, table_norm).ratio()
    return max(token_score, fuzzy_score)


def detect_target_tables(prompt: str, active_tables: list[str]) -> list[str]:
    scored: list[tuple[str, float]] = []
    for table in active_tables:
        score = score_table_match(prompt, table)
        if score >= 0.45:
            scored.append((table, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    if scored:
        return [t for t, _ in scored[:MAX_TABLES]]

    return active_tables


@lru_cache(maxsize=64)
def get_distinct_values(table_name: str, column_name: str) -> list[str]:
    schema = get_table_schema()
    if column_name not in schema.get(table_name, {}):
        return []

    query = sql.SQL(
        """
        SELECT DISTINCT {}
        FROM public.{}
        WHERE {} IS NOT NULL
        LIMIT 3000;
        """
    ).format(
        sql.Identifier(column_name),
        sql.Identifier(table_name),
        sql.Identifier(column_name),
    )

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            return [str(row[0]) for row in cur.fetchall() if row[0] is not None]


def extract_entity_filter(prompt: str, target_tables: list[str], column_name: str) -> tuple[str | None, str | None]:
    normalized = normalize_text(prompt)
    for table in target_tables:
        for value in get_distinct_values(table, column_name):
            if value.lower() in normalized:
                return table, value
    return None, None


def parse_date_value(value: str) -> str | None:
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return value
    except ValueError:
        return None


def extract_date_filter(prompt: str, target_tables: list[str]) -> dict[str, Any] | None:
    normalized = normalize_text(prompt)
    schema = get_table_schema()

    date_columns: list[tuple[str, str]] = []
    for table in target_tables:
        for col, data_type in schema.get(table, {}).items():
            if "date" in col.lower() or data_type in DATE_TYPES:
                date_columns.append((table, col))

    if not date_columns:
        return None

    chosen_table, chosen_col = date_columns[0]
    for table, col in date_columns:
        if col.lower() in normalized:
            chosen_table, chosen_col = table, col
            break

    patterns = [
        ("between", r"(?:between|from)\s+(\d{4}-\d{2}-\d{2})\s+(?:and|to)\s+(\d{4}-\d{2}-\d{2})"),
        ("gte", r"(?:after|since|from)\s+(\d{4}-\d{2}-\d{2})"),
        ("lte", r"(?:before|until|till)\s+(\d{4}-\d{2}-\d{2})"),
        ("eq", r"(?:on|date)\s+(\d{4}-\d{2}-\d{2})"),
    ]

    for op, pattern in patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue

        if op == "between":
            start = parse_date_value(match.group(1))
            end = parse_date_value(match.group(2))
            if start and end:
                return {
                    "table": chosen_table,
                    "column": chosen_col,
                    "op": op,
                    "start": start,
                    "end": end,
                }
            continue

        date_val = parse_date_value(match.group(1))
        if date_val:
            return {
                "table": chosen_table,
                "column": chosen_col,
                "op": op,
                "value": date_val,
            }

    return None


def extract_value_filter(prompt: str, target_tables: list[str]) -> dict[str, Any] | None:
    normalized = normalize_text(prompt)
    schema = get_table_schema()

    table_columns: dict[str, str] = {}
    for table in target_tables:
        for col in schema.get(table, {}).keys():
            table_columns[col.lower()] = col
            table_columns[col.lower().replace("_", " ")] = col

    if not table_columns:
        return None

    symbol_pattern = r"([a-z_][a-z0-9_\s]{1,60})\s*(>=|<=|=|>|<)\s*(-?\d+(?:\.\d+)?)"
    text_pattern = r"([a-z_][a-z0-9_\s]{1,60})\s*(greater than|more than|less than|at least|at most)\s*(-?\d+(?:\.\d+)?)"

    symbol_match = re.search(symbol_pattern, normalized)
    if symbol_match:
        raw_col = symbol_match.group(1).strip()
        op = symbol_match.group(2)
        val = float(symbol_match.group(3))
        col = table_columns.get(raw_col)
        if col:
            table_name = next((t for t in target_tables if col in schema.get(t, {})), None)
            if table_name:
                return {
                    "table": table_name,
                    "column": col,
                    "op": op,
                    "value": val,
                }

    text_match = re.search(text_pattern, normalized)
    if text_match:
        raw_col = text_match.group(1).strip()
        op_text = text_match.group(2)
        val = float(text_match.group(3))
        op_map = {
            "greater than": ">",
            "more than": ">",
            "less than": "<",
            "at least": ">=",
            "at most": "<=",
        }
        col = table_columns.get(raw_col)
        if col:
            table_name = next((t for t in target_tables if col in schema.get(t, {})), None)
            if table_name:
                return {
                    "table": table_name,
                    "column": col,
                    "op": op_map[op_text],
                    "value": val,
                }

    return None


def build_table_filters(
    table_name: str,
    state_filter: dict[str, Any] | None,
    district_filter: dict[str, Any] | None,
    date_filter: dict[str, Any] | None,
    value_filter: dict[str, Any] | None,
) -> dict[str, Any]:
    schema = get_table_schema().get(table_name, {})
    final_filters: dict[str, Any] = {}

    if state_filter and "state_name" in schema:
        final_filters["state"] = state_filter["value"]

    if district_filter and "district_name" in schema:
        final_filters["district"] = district_filter["value"]

    if date_filter and date_filter.get("table") == table_name:
        if date_filter.get("column") in schema:
            final_filters["date"] = date_filter

    if value_filter and value_filter.get("table") == table_name:
        if value_filter.get("column") in schema:
            final_filters["value"] = value_filter

    return final_filters


def build_where_clause(table_filters: dict[str, Any]) -> tuple[list[sql.SQL], list[Any]]:
    clauses: list[sql.SQL] = []
    params: list[Any] = []

    if "state" in table_filters:
        clauses.append(sql.SQL("state_name = %s"))
        params.append(table_filters["state"])

    if "district" in table_filters:
        clauses.append(sql.SQL("district_name = %s"))
        params.append(table_filters["district"])

    if "date" in table_filters:
        date_f = table_filters["date"]
        col = sql.Identifier(date_f["column"])
        if date_f["op"] == "between":
            clauses.append(sql.SQL("({})::date BETWEEN %s AND %s").format(col))
            params.extend([date_f["start"], date_f["end"]])
        elif date_f["op"] == "gte":
            clauses.append(sql.SQL("({})::date >= %s").format(col))
            params.append(date_f["value"])
        elif date_f["op"] == "lte":
            clauses.append(sql.SQL("({})::date <= %s").format(col))
            params.append(date_f["value"])
        elif date_f["op"] == "eq":
            clauses.append(sql.SQL("({})::date = %s").format(col))
            params.append(date_f["value"])

    if "value" in table_filters:
        value_f = table_filters["value"]
        op = value_f["op"] if value_f["op"] in {">", "<", ">=", "<=", "="} else "="
        clauses.append(sql.SQL("({})::double precision {} %s").format(sql.Identifier(value_f["column"]), sql.SQL(op)))
        params.append(value_f["value"])

    return clauses, params


def fetch_rows(table_name: str, limit: int, table_filters: dict[str, Any]):
    where_clauses, params = build_where_clause(table_filters)

    query = sql.SQL("SELECT * FROM public.{}").format(sql.Identifier(table_name))
    if where_clauses:
        query = query + sql.SQL(" WHERE ") + sql.SQL(" AND ").join(where_clauses)
    query = query + sql.SQL(" LIMIT %s;")
    params.append(limit)

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            return cur.fetchall()


def fetch_count(table_name: str, table_filters: dict[str, Any]) -> int:
    where_clauses, params = build_where_clause(table_filters)

    query = sql.SQL("SELECT COUNT(*) AS total_records FROM public.{}").format(sql.Identifier(table_name))
    if where_clauses:
        query = query + sql.SQL(" WHERE ") + sql.SQL(" AND ").join(where_clauses)
    query = query + sql.SQL(";")

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            result = cur.fetchone()
            return int(result["total_records"])


def ensure_audit_table():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS public.query_audit_log (
                    id BIGSERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    user_id TEXT NOT NULL,
                    role_name TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    tables_queried TEXT[] NOT NULL,
                    filters_json JSONB,
                    requested_limit INT NOT NULL,
                    effective_limit INT NOT NULL,
                    result_row_count INT NOT NULL,
                    status TEXT NOT NULL,
                    error_message TEXT
                );
                """
            )


@lru_cache(maxsize=1)
def init_audit_table_once() -> bool:
    ensure_audit_table()
    return True


def write_audit_log(
    user_id: str,
    role_name: str,
    prompt: str,
    intent: str,
    tables: list[str],
    filters_payload: dict[str, Any],
    requested_limit: int,
    effective_limit: int,
    result_row_count: int,
    status: str,
    error_message: str | None = None,
):
    try:
        init_audit_table_once()
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO public.query_audit_log (
                        user_id, role_name, prompt, intent, tables_queried, filters_json,
                        requested_limit, effective_limit, result_row_count, status, error_message
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    (
                        user_id,
                        role_name,
                        prompt,
                        intent,
                        tables,
                        Json(filters_payload),
                        requested_limit,
                        effective_limit,
                        result_row_count,
                        status,
                        error_message,
                    ),
                )
    except Exception:
        pass


@app.get("/table-metadata")
def table_metadata():
    schema = get_table_schema()
    table_names = list(schema.keys())
    allowed = configured_allowed_tables(table_names)
    ready, err = ollama_status()
    return {
        "all_tables": table_names,
        "allowed_tables": allowed,
        "max_tables": MAX_TABLES,
        "roles": ROLE_CONFIG,
        "model_reference": get_model_reference(),
        "llm_ready": ready,
        "llm_error": err,
        "columns": {table: list(schema[table].keys()) for table in allowed},
        "message": "Set APP_ALLOWED_TABLES=table1,table2,table3,table4 to lock exact tables.",
    }


@app.get("/model-reference")
def model_reference():
    ready, err = ollama_status()
    return {
        **get_model_reference(),
        "llm_ready": ready,
        "llm_error": err,
    }


@app.get("/audit-logs")
def get_audit_logs(role: str = "viewer", limit: int = 100):
    role_name = normalize_role(role)
    if role_name != "admin":
        raise HTTPException(status_code=403, detail="Only admin can read audit logs")

    safe_limit = max(1, min(500, limit))
    try:
        init_audit_table_once()
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, created_at, user_id, role_name, prompt, intent, tables_queried,
                           filters_json, requested_limit, effective_limit, result_row_count,
                           status, error_message
                    FROM public.query_audit_log
                    ORDER BY created_at DESC
                    LIMIT %s;
                    """,
                    (safe_limit,),
                )
                return cur.fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query-data")
def query_data(req: PromptRequest):
    role_name = normalize_role(req.role)
    intent = "table_data_retrieval"
    tables_for_log: list[str] = []
    parsed_filters: dict[str, Any] = {}
    effective_limit = req.limit

    try:
        active_tables = choose_active_tables(req.selected_tables, role_name)
        role_cap = ROLE_CONFIG[role_name]["max_limit"]
        extracted_limit = extract_limit(req.prompt, req.limit)
        effective_limit = max(1, min(extracted_limit, req.limit, role_cap))

        # LLM-first path via Ollama.
        try:
            llm_plan = generate_sql_with_ollama(req.prompt, active_tables, role_name, effective_limit)
            llm_rows = execute_generated_sql(llm_plan["sql"])
            intent = llm_plan["intent"]
            tables_for_log = llm_plan["tables"]
            parsed_filters = {"source": "ollama"}

            row_count = len(llm_rows)
            write_audit_log(
                user_id=req.user_id,
                role_name=role_name,
                prompt=req.prompt,
                intent=intent,
                tables=tables_for_log,
                filters_payload=parsed_filters,
                requested_limit=req.limit,
                effective_limit=effective_limit,
                result_row_count=row_count,
                status="success",
            )

            if intent == "count_records":
                first_table = tables_for_log[0] if tables_for_log else "count"
                count_payload: dict[str, Any]
                if llm_rows and "total_records" in llm_rows[0]:
                    count_payload = {"total_records": int(llm_rows[0]["total_records"])}
                else:
                    count_payload = {"rows": llm_rows}
                return {
                    "intent": intent,
                    "tables": [first_table],
                    "parsed": {
                        "role": role_name,
                        "requested_limit": req.limit,
                        "effective_limit": effective_limit,
                        "count": True,
                        "filters": parsed_filters,
                        "sql": llm_plan["sql"],
                        "llm_confidence": llm_plan.get("confidence"),
                        "model_reference": get_model_reference(),
                    },
                    "results": {first_table: count_payload},
                }

            first_table = tables_for_log[0] if tables_for_log else "query_result"
            return {
                "intent": intent,
                "tables": [first_table],
                "parsed": {
                    "role": role_name,
                    "requested_limit": req.limit,
                    "effective_limit": effective_limit,
                    "count": False,
                    "filters": parsed_filters,
                    "sql": llm_plan["sql"],
                    "llm_confidence": llm_plan.get("confidence"),
                    "model_reference": get_model_reference(),
                },
                "results": {first_table: llm_rows},
            }
        except Exception:
            # Fall back to heuristic logic if Ollama fails/unavailable.
            pass

        detected_tables = detect_target_tables(req.prompt, active_tables)
        count_intent = detect_count_intent(req.prompt)

        state_table, state_name = extract_entity_filter(req.prompt, detected_tables, "state_name")
        district_table, district_name = extract_entity_filter(req.prompt, detected_tables, "district_name")
        date_filter = extract_date_filter(req.prompt, detected_tables)
        value_filter = extract_value_filter(req.prompt, detected_tables)

        state_payload = {"table": state_table, "value": state_name} if state_name else None
        district_payload = {"table": district_table, "value": district_name} if district_name else None

        parsed_filters = {
            "state": state_payload,
            "district": district_payload,
            "date": date_filter,
            "value": value_filter,
            "source": "heuristic_fallback",
        }

        if count_intent:
            intent = "count_records"
            counts: dict[str, dict[str, int]] = {}
            for table in detected_tables:
                table_filters = build_table_filters(table, state_payload, district_payload, date_filter, value_filter)
                total = fetch_count(table, table_filters)
                counts[table] = {"total_records": total}

            tables_for_log = detected_tables
            result_row_count = sum(v["total_records"] for v in counts.values())
            write_audit_log(
                user_id=req.user_id,
                role_name=role_name,
                prompt=req.prompt,
                intent=intent,
                tables=tables_for_log,
                filters_payload=parsed_filters,
                requested_limit=req.limit,
                effective_limit=effective_limit,
                result_row_count=result_row_count,
                status="success",
            )

            return {
                "intent": intent,
                "tables": detected_tables,
                "parsed": {
                    "role": role_name,
                    "requested_limit": req.limit,
                    "effective_limit": effective_limit,
                    "count": True,
                    "filters": parsed_filters,
                },
                "results": counts,
            }

        if state_name and state_table:
            intent = "state_filtered_data"
            table_filters = build_table_filters(state_table, state_payload, district_payload, date_filter, value_filter)
            rows = fetch_rows(state_table, effective_limit, table_filters)
            tables_for_log = [state_table]
            result_row_count = len(rows)

            write_audit_log(
                user_id=req.user_id,
                role_name=role_name,
                prompt=req.prompt,
                intent=intent,
                tables=tables_for_log,
                filters_payload=parsed_filters,
                requested_limit=req.limit,
                effective_limit=effective_limit,
                result_row_count=result_row_count,
                status="success",
            )

            return {
                "intent": intent,
                "tables": [state_table],
                "parsed": {
                    "role": role_name,
                    "requested_limit": req.limit,
                    "effective_limit": effective_limit,
                    "count": False,
                    "filters": parsed_filters,
                },
                "results": {state_table: rows},
            }

        results: dict[str, list[dict[str, Any]]] = {}
        total_rows = 0
        for table in detected_tables:
            table_filters = build_table_filters(table, state_payload, district_payload, date_filter, value_filter)
            rows = fetch_rows(table, effective_limit, table_filters)
            results[table] = rows
            total_rows += len(rows)

        tables_for_log = detected_tables
        write_audit_log(
            user_id=req.user_id,
            role_name=role_name,
            prompt=req.prompt,
            intent=intent,
            tables=tables_for_log,
            filters_payload=parsed_filters,
            requested_limit=req.limit,
            effective_limit=effective_limit,
            result_row_count=total_rows,
            status="success",
        )

        return {
            "intent": intent,
            "tables": detected_tables,
            "parsed": {
                "role": role_name,
                "requested_limit": req.limit,
                "effective_limit": effective_limit,
                "count": False,
                "filters": parsed_filters,
            },
            "results": results,
        }

    except HTTPException as exc:
        write_audit_log(
            user_id=req.user_id,
            role_name=role_name,
            prompt=req.prompt,
            intent=intent,
            tables=tables_for_log,
            filters_payload=parsed_filters,
            requested_limit=req.limit,
            effective_limit=effective_limit,
            result_row_count=0,
            status="failed",
            error_message=str(exc.detail),
        )
        raise
    except Exception as e:
        write_audit_log(
            user_id=req.user_id,
            role_name=role_name,
            prompt=req.prompt,
            intent=intent,
            tables=tables_for_log,
            filters_payload=parsed_filters,
            requested_limit=req.limit,
            effective_limit=effective_limit,
            result_row_count=0,
            status="failed",
            error_message=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/all-data")
def fetch_all_data():
    try:
        tables = choose_active_tables([], "viewer")
        if not tables:
            raise HTTPException(status_code=500, detail="No tables configured")
        return fetch_rows(table_name=tables[0], limit=50, table_filters={})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chhattisgarh-data")
def get_chhattisgarh_data():
    try:
        tables = choose_active_tables([], "viewer")
        schema = get_table_schema()
        table = next((t for t in tables if "state_name" in schema.get(t, {})), tables[0])
        return fetch_rows(table_name=table, limit=10, table_filters={"state": "CHHATTISGARH"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
