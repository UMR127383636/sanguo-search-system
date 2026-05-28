import re
import pandas as pd
from pathlib import Path

from semantic_search import SemanticSearcher
from postprocess import load_alias_dict, normalize_query, rerank_results


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "sanguo_politics_clean.xlsx"
ALIAS_PATH = BASE_DIR / "data" / "alias_dict.csv"
MODEL_PATH = BASE_DIR / "models" / "bge-small-zh-v1.5"
INDEX_DIR = BASE_DIR / ".semantic_index"


searcher = SemanticSearcher(
    excel_path=DATA_PATH,
    model_name=str(MODEL_PATH),
    sheet_name="clean_data",
    index_dir=INDEX_DIR,
    fields=[
        "summary",
        "event",
        "keywords",
        "strategy_type",
        "strategy_method",
        "main_person",
        "related_persons",
        "text"
    ],
    query_prefix="auto"
)

searcher.build_or_load_index(rebuild=False)


if ALIAS_PATH.exists():
    alias_dict = load_alias_dict(ALIAS_PATH)
else:
    alias_dict = {}


def convert_results_to_dataframe(results):
    if isinstance(results, pd.DataFrame):
        df_results = results.copy()

        if "score" in df_results.columns and "semantic_score" not in df_results.columns:
            df_results["semantic_score"] = df_results["score"]

        return df_results

    records = []

    for item in results:
        if hasattr(item, "to_flat_dict"):
            row = item.to_flat_dict()
        elif isinstance(item, dict):
            row = item
        else:
            row = dict(item)

        if "semantic_score" not in row:
            row["semantic_score"] = row.get("score", 0)

        if "score" in row:
            del row["score"]

        records.append(row)

    return pd.DataFrame(records)


def normalize_selected_person(selected_person):
    if selected_person is None:
        return "不限"

    selected_person = str(selected_person).strip()

    if selected_person == "" or selected_person == "不限":
        return "不限"

    return normalize_query(selected_person, alias_dict)


def get_full_corpus_dataframe():
    if hasattr(searcher, "df") and searcher.df is not None:
        return searcher.df.copy()

    return pd.read_excel(DATA_PATH, sheet_name="clean_data")


def normalize_boolean_query(query):
    query = str(query)

    query = query.replace("与", " AND ")
    query = query.replace("且", " AND ")
    query = query.replace("或", " OR ")
    query = query.replace("非", " NOT ")

    query = re.sub(r"\band\b", " AND ", query, flags=re.IGNORECASE)
    query = re.sub(r"\bor\b", " OR ", query, flags=re.IGNORECASE)
    query = re.sub(r"\bnot\b", " NOT ", query, flags=re.IGNORECASE)

    query = re.sub(r"\s+", " ", query).strip()

    return query


def split_terms(text):
    text = re.sub(r"[，。！？、；：,.!?;:（）()《》“”\"']", " ", str(text))

    terms = []

    for part in text.split():
        part = part.strip()

        if part != "":
            terms.append(part)

    return terms


def group_match(text, group_expr):
    group_expr = str(group_expr).strip()

    if group_expr == "":
        return True

    if " NOT " in group_expr:
        include_part, *not_parts = group_expr.split(" NOT ")
    else:
        include_part = group_expr
        not_parts = []

    include_terms = []

    if " AND " in include_part:
        for part in include_part.split(" AND "):
            include_terms.extend(split_terms(part))
    else:
        include_terms.extend(split_terms(include_part))

    exclude_terms = []

    for part in not_parts:
        if " AND " in part:
            for sub in part.split(" AND "):
                exclude_terms.extend(split_terms(sub))
        else:
            exclude_terms.extend(split_terms(part))

    for term in exclude_terms:
        if term and term in text:
            return False

    if len(include_terms) == 0:
        return True

    for term in include_terms:
        if term not in text:
            return False

    return True


def boolean_match(text, query):
    query = normalize_boolean_query(query)

    if query == "":
        return False

    if " OR " in query:
        groups = query.split(" OR ")

        for group in groups:
            if group_match(text, group):
                return True

        return False

    return group_match(text, query)


def boolean_term_score(text, query):
    query = normalize_boolean_query(query)

    terms = split_terms(
        query.replace(" AND ", " ")
             .replace(" OR ", " ")
             .replace(" NOT ", " ")
    )

    terms = list(set([term for term in terms if term]))

    if len(terms) == 0:
        return 0

    matched = 0

    for term in terms:
        if term in text:
            matched += 1

    return round(matched / len(terms), 4)


def build_search_text(row):
    fields = [
        "chapter",
        "main_person",
        "related_persons",
        "strategy_type",
        "strategy_method",
        "event",
        "text",
        "summary",
        "keywords"
    ]

    parts = []

    for field in fields:
        parts.append(str(row.get(field, "")))

    return " ".join(parts)


def simple_boolean_search(query):
    df = get_full_corpus_dataframe()

    if df.empty:
        return pd.DataFrame()

    matched_rows = []

    for _, row in df.iterrows():
        search_text = build_search_text(row)

        if boolean_match(search_text, query):
            new_row = row.to_dict()
            new_row["semantic_score"] = boolean_term_score(search_text, query)
            matched_rows.append(new_row)

    if len(matched_rows) == 0:
        return pd.DataFrame()

    df_results = pd.DataFrame(matched_rows)

    df_results["semantic_score"] = pd.to_numeric(
        df_results["semantic_score"],
        errors="coerce"
    ).fillna(0)

    return df_results


def semantic_search_raw(query, candidate_k):
    try:
        return searcher.search(
            query=query,
            topk=candidate_k
        )
    except TypeError:
        try:
            return searcher.search(
                query=query,
                top_k=candidate_k
            )
        except TypeError:
            return searcher.search(query, candidate_k)


def search_system(
        query,
        selected_person="不限",
        selected_type="不限",
        selected_scope="全部字段",
        search_mode="语义检索",
        top_k=5
):
    if query is None or str(query).strip() == "":
        return pd.DataFrame()

    query = str(query).strip()
    selected_scope = str(selected_scope).strip()
    search_mode = str(search_mode).strip()

    try:
        top_k = int(top_k)
    except (ValueError, TypeError):
        top_k = 5

    if top_k <= 0:
        top_k = 5

    normalized_query = normalize_query(query, alias_dict)
    selected_person = normalize_selected_person(selected_person)

    if search_mode == "布尔检索":
        df_candidates = simple_boolean_search(
            query=normalized_query
        )
    else:
        total_docs = len(searcher.df) if hasattr(searcher, "df") and searcher.df is not None else 500
        candidate_k = min(max(20, top_k * 4), total_docs)

        raw_results = semantic_search_raw(
            query=normalized_query,
            candidate_k=candidate_k
        )

        df_candidates = convert_results_to_dataframe(raw_results)

    if df_candidates.empty:
        return pd.DataFrame()

    if "semantic_score" not in df_candidates.columns:
        df_candidates["semantic_score"] = 0

    df_candidates["semantic_score"] = pd.to_numeric(
        df_candidates["semantic_score"],
        errors="coerce"
    ).fillna(0)

    df_final = rerank_results(
        df_candidates,
        query=normalized_query,
        selected_person=selected_person,
        selected_type=selected_type,
        selected_scope=selected_scope,
        top_k=top_k,
        alias_dict=alias_dict
    )

    display_cols = [
        "rank",
        "chapter",
        "main_person",
        "related_persons",
        "strategy_type",
        "strategy_method",
        "event",
        "text",
        "summary",
        "semantic_score",
        "person_score",
        "theme_score",
        "field_score",
        "final_score"
    ]

    existing_cols = [col for col in display_cols if col in df_final.columns]

    return df_final[existing_cols].copy()


if __name__ == "__main__":
    result = search_system(
        query="曹操 AND 天子",
        selected_person="曹操",
        selected_type="不限",
        selected_scope="全部字段",
        search_mode="布尔检索",
        top_k=5
    )

    print(result.to_string(index=False))