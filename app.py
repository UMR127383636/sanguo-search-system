from flask import Flask, request, jsonify, render_template
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main_search import search_system

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


def safe_text(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass

    return str(value)


def safe_number(value, digits=3):
    if value is None:
        return 0

    try:
        if pd.isna(value):
            return 0
    except TypeError:
        pass

    try:
        return round(float(value), digits)
    except (ValueError, TypeError):
        return 0


@app.route("/api/search", methods=["POST"])
def api_search():
    try:
        req_data = request.get_json() or {}

        query = str(req_data.get("query", "")).strip()
        selected_person = req_data.get("character", "不限")
        selected_type = req_data.get("strategyType", "不限")
        selected_scope = req_data.get("searchScope", "全部字段")
        search_mode = req_data.get("searchMode", "语义检索")
        top_k = req_data.get("topK", 5)

        try:
            top_k = int(top_k)
        except (ValueError, TypeError):
            top_k = 5

        if top_k <= 0:
            top_k = 5

        if query == "":
            return jsonify({
                "status": "success",
                "count": 0,
                "data": []
            })

        print("query:", query)
        print("selected_person:", selected_person)
        print("selected_type:", selected_type)
        print("selected_scope:", selected_scope)
        print("search_mode:", search_mode)
        print("top_k:", top_k)

        result_df = search_system(
            query=query,
            selected_person=selected_person,
            selected_type=selected_type,
            selected_scope=selected_scope,
            search_mode=search_mode,
            top_k=top_k
        )

        formatted_results = []

        if result_df is not None and not result_df.empty:
            for _, row in result_df.iterrows():
                rank_value = safe_number(row.get("rank", 0), 0)

                formatted_results.append({
                    "rank": int(rank_value) if rank_value != 0 else "",
                    "chapter": safe_text(row.get("chapter", "")),
                    "main_person": safe_text(row.get("main_person", "")),
                    "related_persons": safe_text(row.get("related_persons", "")),
                    "strategy_type": safe_text(row.get("strategy_type", "")),
                    "strategy_method": safe_text(row.get("strategy_method", "")),
                    "event": safe_text(row.get("event", "")),
                    "text": safe_text(row.get("text", "")),
                    "summary": safe_text(row.get("summary", "")),
                    "semantic_score": safe_number(row.get("semantic_score", 0), 4),
                    "person_score": safe_number(row.get("person_score", 0), 2),
                    "theme_score": safe_number(row.get("theme_score", 0), 2),
                    "field_score": safe_number(row.get("field_score", 0), 2),
                    "final_score": safe_number(row.get("final_score", 0), 3)
                })

        return jsonify({
            "status": "success",
            "count": len(formatted_results),
            "data": formatted_results
        })

    except Exception as e:
        import traceback
        traceback.print_exc()

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)