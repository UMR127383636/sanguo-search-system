import re
import pandas as pd


WEIGHTS = {
    "semantic": 0.45,
    "person": 0.20,
    "theme": 0.15,
    "field": 0.20
}


def load_alias_dict(path="alias_dict.csv"):
    try:
        df_alias = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df_alias = pd.read_csv(path, encoding="gbk")

    alias_dict = {}

    if "alias" not in df_alias.columns or "standard_name" not in df_alias.columns:
        return alias_dict

    for _, row in df_alias.iterrows():
        alias = str(row.get("alias", "")).strip()
        standard_name = str(row.get("standard_name", "")).strip()

        if alias != "" and standard_name != "":
            alias_dict[alias] = standard_name

    return alias_dict


def normalize_query(query, alias_dict):
    if query is None:
        return ""

    query = str(query)

    if alias_dict is None:
        return query

    for alias, standard_name in sorted(alias_dict.items(), key=lambda x: len(x[0]), reverse=True):
        query = query.replace(alias, standard_name)

    return query


def compute_label_scores(
        row,
        selected_person="不限",
        selected_type="不限"
):
    selected_person = str(selected_person).strip()
    selected_type = str(selected_type).strip()

    main_person = str(row.get("main_person", "")).strip()
    related_persons = str(row.get("related_persons", "")).strip()
    strategy_type = str(row.get("strategy_type", "")).strip()

    if selected_person == "不限":
        person_score = 0.5
    else:
        if selected_person == main_person:
            person_score = 1.0
        elif selected_person in related_persons:
            person_score = 0.6
        else:
            person_score = 0.0

    if selected_type == "不限":
        theme_score = 0.5
    else:
        if selected_type == strategy_type:
            theme_score = 1.0
        else:
            theme_score = 0.0

    return person_score, theme_score


def extract_query_terms(query):
    if query is None:
        return []

    query = str(query)

    common_terms = [
        "曹操", "刘备", "诸葛亮", "孙权", "司马懿", "袁绍", "周瑜", "贾诩", "荀彧", "关羽",
        "政治合法性", "政治优势", "合法性", "天子", "朝廷", "许都", "魏王",
        "权力控制", "权力", "控制", "掌权",
        "联盟外交", "联盟", "外交", "孙刘联盟", "联吴", "抗曹",
        "人才任用", "人才", "任用", "招揽", "三顾茅庐", "求贤",
        "汉献帝", "奉天子", "鲁肃", "徐庶"
    ]

    terms = []

    for term in common_terms:
        if term in query:
            terms.append(term)

    cleaned_query = re.sub(r"[，。！？、；：,.!?;:（）()《》“”\"']", " ", query)

    cleaned_query = re.sub(
        r"\bAND\b|\bOR\b|\bNOT\b|与|或|非",
        " ",
        cleaned_query,
        flags=re.IGNORECASE
    )

    stop_words = [
        "如何", "怎么", "为什么", "哪些", "什么", "有关", "相关",
        "获得", "进行", "体现", "通过", "以及", "一个", "一些",
        "的", "了", "在", "中", "有", "吗", "呢"
    ]

    for stop_word in stop_words:
        cleaned_query = cleaned_query.replace(stop_word, " ")

    for part in cleaned_query.split():
        part = part.strip()

        if len(part) >= 2:
            terms.append(part)

    return list(set(terms))


def compute_field_score(row, query="", selected_scope="全部字段"):
    if selected_scope == "全部字段":
        return 0.5

    scope_map = {
        "原文片段": "text",
        "事件": "event",
        "解释摘要": "summary",
        "关键词": "keywords",
        "具体策略": "strategy_method"
    }

    field_name = scope_map.get(selected_scope)

    if field_name is None:
        return 0.5

    field_text = str(row.get(field_name, ""))
    query_terms = extract_query_terms(query)

    if len(query_terms) == 0:
        return 0.5

    matched_count = 0

    for term in query_terms:
        if term in field_text:
            matched_count += 1

    field_score = matched_count / len(query_terms)

    return round(field_score, 3)


def calculate_final_score(
        semantic_score,
        person_score,
        theme_score,
        field_score=0.5
):
    semantic_score = float(semantic_score)

    final_score = (
        WEIGHTS["semantic"] * semantic_score
        + WEIGHTS["person"] * person_score
        + WEIGHTS["theme"] * theme_score
        + WEIGHTS["field"] * field_score
    )

    return round(final_score, 3)


def deduplicate_results(df):
    df = df.sort_values(
        by="final_score",
        ascending=False
    )

    if "text" in df.columns:
        df = df.drop_duplicates(
            subset=["text"],
            keep="first"
        )

    if "event" in df.columns:
        df = df.drop_duplicates(
            subset=["event"],
            keep="first"
        )

    return df


def rerank_results(
        df_results,
        query="",
        selected_person="不限",
        selected_type="不限",
        selected_scope="全部字段",
        top_k=5,
        alias_dict=None
):
    if df_results is None or len(df_results) == 0:
        return pd.DataFrame()

    df_results = df_results.copy()

    if alias_dict is not None and selected_person != "不限":
        selected_person = normalize_query(selected_person, alias_dict)

    person_scores = []
    theme_scores = []
    field_scores = []
    final_scores = []

    for _, row in df_results.iterrows():
        person_score, theme_score = compute_label_scores(
            row,
            selected_person,
            selected_type
        )

        field_score = compute_field_score(
            row,
            query=query,
            selected_scope=selected_scope
        )

        final_score = calculate_final_score(
            row.get("semantic_score", 0),
            person_score,
            theme_score,
            field_score
        )

        person_scores.append(person_score)
        theme_scores.append(theme_score)
        field_scores.append(field_score)
        final_scores.append(final_score)

    df_results["person_score"] = person_scores
    df_results["theme_score"] = theme_scores
    df_results["field_score"] = field_scores
    df_results["final_score"] = final_scores

    df_results = deduplicate_results(df_results)

    df_results = df_results.sort_values(
        by="final_score",
        ascending=False
    )

    df_results = df_results.head(top_k)

    if "rank" in df_results.columns:
        df_results = df_results.drop(columns=["rank"])

    df_results.insert(
        0,
        "rank",
        range(1, len(df_results) + 1)
    )

    return df_results