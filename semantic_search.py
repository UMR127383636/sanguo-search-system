
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_MODEL = "./models/bge-small-zh-v1.5"

MODEL_ALIASES = {
    "bge-small-zh": "BAAI/bge-small-zh-v1.5",
    "bge-small-zh-v1.5": "BAAI/bge-small-zh-v1.5",
    "text2vec-base-chinese": "shibing624/text2vec-base-chinese",
}

DEFAULT_FIELDS = [
    "summary",
    "event",
    "keywords",
    "strategy_type",
    "strategy_method",
    "main_person",
    "related_persons",
    "text",
]

FIELD_LABELS = {
    "id": "编号",
    "chapter": "回目",
    "text": "原文",
    "main_person": "主要人物",
    "related_persons": "相关人物",
    "strategy_type": "策略类型",
    "strategy_method": "策略方法",
    "event": "事件",
    "keywords": "关键词",
    "summary": "摘要",
}


def resolve_model_name(model_name: str) -> str:
    return MODEL_ALIASES.get(model_name.strip(), model_name.strip())


def clean_text(value: Any) -> str:
    """把单元格内容转成适合拼接和检索的干净文本。"""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    return " ".join(text.split())


def parse_fields(fields_text: str) -> List[str]:
    fields = [x.strip() for x in fields_text.split(",") if x.strip()]
    if not fields:
        raise ValueError("fields 不能为空。")
    return fields


def read_excel_data(excel_path: Path, sheet_name: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    if not excel_path.exists():
        raise FileNotFoundError(f"找不到 Excel 文件：{excel_path}")

    xls = pd.ExcelFile(excel_path)
    if not xls.sheet_names:
        raise ValueError(f"Excel 没有可读取的工作表：{excel_path}")

    actual_sheet = sheet_name or xls.sheet_names[0]
    if actual_sheet not in xls.sheet_names:
        available = "、".join(xls.sheet_names)
        raise ValueError(f"找不到工作表 {actual_sheet!r}。可用工作表：{available}")

    df = pd.read_excel(excel_path, sheet_name=actual_sheet)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all").fillna("")
    return df, actual_sheet


def validate_fields(df: pd.DataFrame, fields: Sequence[str]) -> None:
    missing = [f for f in fields if f not in df.columns]
    if missing:
        available = "、".join(map(str, df.columns))
        raise ValueError(
            "检索字段在 Excel 中不存在："
            + "、".join(missing)
            + f"\n当前可用字段：{available}"
        )


def build_document_text(row: pd.Series, fields: Sequence[str]) -> str:

    parts: List[str] = []
    for field in fields:
        value = clean_text(row.get(field, ""))
        if not value:
            continue
        label = FIELD_LABELS.get(field, field)
        parts.append(f"{label}：{value}")
    return "\n".join(parts)


def make_cache_signature(
    excel_path: Path,
    sheet_name: str,
    model_name: str,
    fields: Sequence[str],
    query_prefix: str,
) -> str:
    stat = excel_path.stat()
    payload = {
        "excel": str(excel_path.resolve()),
        "mtime": round(stat.st_mtime, 3),
        "size": stat.st_size,
        "sheet": sheet_name,
        "model": model_name,
        "fields": list(fields),
        "query_prefix": query_prefix,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def ensure_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "缺少依赖 sentence-transformers。\n"
            "请先安装：pip install sentence-transformers\n"
            "如果还没有 Excel 读取依赖，也安装：pip install pandas openpyxl numpy"
        ) from exc
    return SentenceTransformer


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return embeddings / norms


def clip(text: str, max_len: int = 120) -> str:
    text = clean_text(text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def contains_text(value: Any, needle: Optional[str]) -> bool:
    if not needle:
        return True
    return needle.strip() in clean_text(value)


@dataclass
class SearchResult:
    rank: int
    score: float
    row_index: int
    data: Dict[str, Any]

    def to_flat_dict(self) -> Dict[str, Any]:
        out = {"rank": self.rank, "score": round(float(self.score), 6), "row_index": self.row_index}
        out.update(self.data)
        return out


class SemanticSearcher:
    def __init__(
        self,
        excel_path: str | Path,
        model_name: str = DEFAULT_MODEL,
        sheet_name: Optional[str] = None,
        index_dir: str | Path = ".semantic_index",
        fields: Optional[Sequence[str]] = None,
        device: Optional[str] = None,
        query_prefix: str = "auto",
        batch_size: int = 32,
    ) -> None:
        self.excel_path = Path(excel_path)
        self.model_name = resolve_model_name(model_name)
        self.sheet_name = sheet_name
        self.index_dir = Path(index_dir)
        self.fields = list(fields or DEFAULT_FIELDS)
        self.device = device
        self.query_prefix = query_prefix
        self.batch_size = batch_size

        self.df: Optional[pd.DataFrame] = None
        self.actual_sheet_name: Optional[str] = None
        self.texts: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self._model = None

    def _load_model(self):
        if self._model is None:
            SentenceTransformer = ensure_sentence_transformer()
            if self.device:
                self._model = SentenceTransformer(self.model_name, device=self.device)
            else:
                self._model = SentenceTransformer(self.model_name)
        return self._model

    def _query_text(self, query: str) -> str:
        query = query.strip()
        if self.query_prefix in ("", "none", "None", None):
            return query

        if self.query_prefix == "auto":
            # BGE 系列常用中文 query instruction；其他模型默认不加。
            if "bge" in self.model_name.lower():
                return "为这个句子生成表示以用于检索相关文章：" + query
            return query

        return str(self.query_prefix) + query

    def _cache_paths(self) -> Tuple[Path, Path]:
        if not self.actual_sheet_name:
            raise RuntimeError("需要先读取 Excel，才能确定缓存路径。")

        signature = make_cache_signature(
            excel_path=self.excel_path,
            sheet_name=self.actual_sheet_name,
            model_name=self.model_name,
            fields=self.fields,
            query_prefix=self.query_prefix,
        )
        stem = f"{self.excel_path.stem}_{signature}"
        return self.index_dir / f"{stem}.embeddings.npy", self.index_dir / f"{stem}.meta.json"

    def load_corpus(self) -> None:
        df, actual_sheet = read_excel_data(self.excel_path, self.sheet_name)
        validate_fields(df, self.fields)

        self.df = df
        self.actual_sheet_name = actual_sheet
        self.texts = [build_document_text(row, self.fields) for _, row in df.iterrows()]

        empty_count = sum(1 for t in self.texts if not t.strip())
        if empty_count == len(self.texts):
            raise ValueError("所有行拼接后的检索文本都是空的，请检查 fields 参数。")

    def build_or_load_index(self, rebuild: bool = False) -> None:
        self.load_corpus()
        emb_path, meta_path = self._cache_paths()
        self.index_dir.mkdir(parents=True, exist_ok=True)

        if emb_path.exists() and meta_path.exists() and not rebuild:
            self.embeddings = np.load(emb_path)
            if self.embeddings.shape[0] != len(self.texts):
                print("缓存行数和 Excel 行数不一致，将重新构建向量。", file=sys.stderr)
            else:
                return

        model = self._load_model()
        embeddings = model.encode(
            self.texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        self.embeddings = normalize_embeddings(embeddings)
        np.save(emb_path, self.embeddings)

        meta = {
            "excel_path": str(self.excel_path),
            "sheet_name": self.actual_sheet_name,
            "model_name": self.model_name,
            "fields": self.fields,
            "rows": len(self.texts),
            "embedding_shape": list(self.embeddings.shape),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def _filter_mask(
        self,
        person: Optional[str] = None,
        strategy_type: Optional[str] = None,
        keyword: Optional[str] = None,
    ) -> np.ndarray:
        if self.df is None:
            raise RuntimeError("请先调用 build_or_load_index()。")

        mask = np.ones(len(self.df), dtype=bool)

        if person:
            p = person.strip()
            person_mask = np.zeros(len(self.df), dtype=bool)
            for col in ("main_person", "related_persons"):
                if col in self.df.columns:
                    person_mask |= self.df[col].astype(str).str.contains(p, regex=False, na=False).to_numpy()
            mask &= person_mask

        if strategy_type and "strategy_type" in self.df.columns:
            st = strategy_type.strip()
            mask &= self.df["strategy_type"].astype(str).str.contains(st, regex=False, na=False).to_numpy()

        if keyword:
            kw = keyword.strip()
            keyword_mask = np.zeros(len(self.df), dtype=bool)
            for col in ("keywords", "summary", "event", "text"):
                if col in self.df.columns:
                    keyword_mask |= self.df[col].astype(str).str.contains(kw, regex=False, na=False).to_numpy()
            mask &= keyword_mask

        return mask

    def search(
        self,
        query: str,
        topk: int = 5,
        person: Optional[str] = None,
        strategy_type: Optional[str] = None,
        keyword: Optional[str] = None,
    ) -> List[SearchResult]:
        if self.df is None or self.embeddings is None:
            self.build_or_load_index(rebuild=False)

        assert self.df is not None
        assert self.embeddings is not None

        query_text = self._query_text(query)
        model = self._load_model()
        query_emb = model.encode(
            [query_text],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        query_emb = normalize_embeddings(query_emb)[0]

        scores = self.embeddings @ query_emb
        mask = self._filter_mask(person=person, strategy_type=strategy_type, keyword=keyword)

        candidate_indices = np.where(mask)[0]
        if len(candidate_indices) == 0:
            return []

        candidate_scores = scores[candidate_indices]
        topk = max(1, min(int(topk), len(candidate_indices)))
        local_order = np.argsort(-candidate_scores)[:topk]

        results: List[SearchResult] = []
        for rank, local_idx in enumerate(local_order, start=1):
            row_idx = int(candidate_indices[local_idx])
            row = self.df.iloc[row_idx].to_dict()
            results.append(
                SearchResult(
                    rank=rank,
                    score=float(scores[row_idx]),
                    row_index=row_idx,
                    data={k: clean_text(v) for k, v in row.items()},
                )
            )
        return results


def print_results(results: Sequence[SearchResult], show_text: bool = True) -> None:
    if not results:
        print("没有检索到结果。可以尝试放宽 person / strategy-type / keyword 过滤条件。")
        return

    for item in results:
        d = item.data
        print("=" * 88)
        print(f"#{item.rank}  score={item.score:.4f}  row={item.row_index}")
        print(f"id: {d.get('id', '')} | chapter: {d.get('chapter', '')}")
        print(f"人物: {d.get('main_person', '')} | 相关人物: {d.get('related_persons', '')}")
        print(f"策略: {d.get('strategy_type', '')} | 方法: {d.get('strategy_method', '')}")
        print(f"事件: {d.get('event', '')}")
        print(f"关键词: {d.get('keywords', '')}")
        print(f"摘要: {d.get('summary', '')}")
        if show_text:
            print(f"原文: {clip(d.get('text', ''), 180)}")


def save_results(results: Sequence[SearchResult], output_path: str | Path) -> None:
    path = Path(output_path)
    records = [r.to_flat_dict() for r in results]
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        pd.DataFrame(records).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存：{path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="三国政治策略语料的语义向量构建与相似度检索。"
    )
    parser.add_argument("--excel", default="sanguo_politics.xlsx", help="Excel 文件路径。")
    parser.add_argument("--sheet", default=None, help="工作表名称；不填则默认第一个 sheet。")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"语义模型，默认：{DEFAULT_MODEL}")
    parser.add_argument("--index-dir", default=".semantic_index", help="向量缓存目录。")
    parser.add_argument(
        "--fields",
        default=",".join(DEFAULT_FIELDS),
        help="参与向量化的字段，逗号分隔。默认会综合摘要、事件、关键词、策略、人物和原文。",
    )
    parser.add_argument("--query", default=None, help="检索问题 / 检索语句。")
    parser.add_argument("--topk", type=int, default=5, help="返回结果数量。")
    parser.add_argument("--person", default=None, help="人物过滤，会匹配 main_person 和 related_persons。")
    parser.add_argument("--strategy-type", default=None, help="策略类型过滤，如：联盟外交、政治斗争。")
    parser.add_argument("--keyword", default=None, help="关键词过滤，会匹配 keywords / summary / event / text。")
    parser.add_argument("--output", default=None, help="保存检索结果，支持 .csv 或 .json。")
    parser.add_argument("--rebuild", action="store_true", help="忽略缓存，重新生成向量。")
    parser.add_argument("--build-only", action="store_true", help="只构建向量，不执行检索。")
    parser.add_argument("--device", default=None, help="运行设备，如 cuda、cpu；不填由 sentence-transformers 自动判断。")
    parser.add_argument("--batch-size", type=int, default=32, help="向量化 batch size。")
    parser.add_argument(
        "--query-prefix",
        default="auto",
        help=(
            "查询前缀。默认 auto：BGE 模型自动加中文检索指令；"
            "设为 none 则不加；也可以传入自定义前缀。"
        ),
    )
    parser.add_argument("--no-text", action="store_true", help="打印结果时不显示原文片段。")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    fields = parse_fields(args.fields)
    searcher = SemanticSearcher(
        excel_path=args.excel,
        model_name=args.model,
        sheet_name=args.sheet,
        index_dir=args.index_dir,
        fields=fields,
        device=args.device,
        query_prefix=args.query_prefix,
        batch_size=args.batch_size,
    )

    searcher.build_or_load_index(rebuild=args.rebuild)
    print(
        f"语义索引已就绪：rows={len(searcher.texts)} | "
        f"sheet={searcher.actual_sheet_name} | model={searcher.model_name}"
    )

    if args.build_only:
        return

    if not args.query:
        print("\n未提供 --query，因此只完成了向量索引构建。")
        print('示例：python semantic_search.py --excel sanguo_politics.xlsx --query "曹操如何招揽人才"')
        return

    results = searcher.search(
        query=args.query,
        topk=args.topk,
        person=args.person,
        strategy_type=args.strategy_type,
        keyword=args.keyword,
    )
    print_results(results, show_text=not args.no_text)

    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
