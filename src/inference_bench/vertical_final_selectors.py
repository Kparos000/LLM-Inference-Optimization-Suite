"""Canonical vertical-specific final top-k selectors.

The selectors operate after dense/BM25/hybrid candidate generation. They use
prompt-visible query terms and context metadata only; they do not inspect gold
records, source hints, or answer-side identifiers.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from inference_bench.context_schema import ContextRecord

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

RankedCandidate = tuple[str, float, dict[str, float]]
RESEARCH_SECTION_FAMILY_MAP = {
    "abstract": "overview",
    "introduction": "overview",
    "background": "overview",
    "overview": "overview",
    "related_work": "overview",
    "related": "overview",
    "relatedwork": "overview",
    "relatedworks": "overview",
    "method": "method",
    "methods": "method",
    "methodology": "method",
    "approach": "method",
    "model": "method",
    "algorithm": "method",
    "training": "method",
    "data": "method",
    "dataset": "method",
    "datasets": "method",
    "experiment": "results",
    "experiments": "results",
    "experimental": "results",
    "evaluation": "results",
    "results": "results",
    "result": "results",
    "analysis": "results",
    "ablation": "results",
    "experimentalsetup": "results",
    "mainresults": "results",
    "limitations": "limitations",
    "limitation": "limitations",
    "discussion": "limitations",
    "conclusion": "limitations",
    "conclusions": "limitations",
}
RESEARCH_TARGET_TERMS = {
    "overview": {
        "abstract",
        "introduction",
        "overview",
        "claim",
        "contribution",
        "boundary",
        "cited",
        "section",
        "evidence",
    },
    "method": {
        "method",
        "methods",
        "methodology",
        "approach",
        "algorithm",
        "model",
        "training",
        "data",
        "dataset",
        "pipeline",
        "framework",
        "architecture",
    },
    "results": {
        "result",
        "results",
        "experiment",
        "experiments",
        "evaluation",
        "benchmark",
        "ablation",
        "performance",
        "outperform",
        "accuracy",
    },
    "limitations": {
        "limitation",
        "limitations",
        "discussion",
        "conclusion",
        "risk",
        "caveat",
        "failure",
    },
}


def tokens(text: str) -> set[str]:
    """Return lowercase alphanumeric tokens."""

    return {match.group(0).lower() for match in TOKEN_RE.finditer(text)}


def normalize(value: str) -> str:
    """Return a compact normalized value."""

    return " ".join(TOKEN_RE.findall(value.lower()))


def record_text(record: ContextRecord) -> str:
    """Return selector-visible record text."""

    metadata = record.metadata
    parts = [
        record.title,
        record.text,
        " ".join(str(value) for value in metadata.values() if not isinstance(value, (dict, list))),
    ]
    for value in metadata.values():
        if isinstance(value, list):
            parts.append(" ".join(str(item) for item in value))
    return " ".join(parts)


def lexical_overlap(query_tokens: set[str], candidate_tokens: set[str]) -> float:
    """Return query coverage by candidate tokens."""

    if not query_tokens:
        return 0.0
    return len(query_tokens & candidate_tokens) / len(query_tokens)


def metadata_tokens(record: ContextRecord, fields: Iterable[str]) -> set[str]:
    """Return tokens for selected metadata fields."""

    values: list[str] = []
    for field in fields:
        value = record.metadata.get(field)
        if isinstance(value, list):
            values.extend(str(item) for item in value)
        elif value is not None:
            values.append(str(value))
    return tokens(" ".join(values))


def research_paper_key(record: ContextRecord) -> str:
    """Return the stable Research AI paper grouping key."""

    meta = record.metadata
    return normalize(
        " ".join(
            str(value)
            for value in (
                meta.get("paper_id"),
                meta.get("paper_title"),
                record.parent_id,
                record.title,
            )
            if value
        )
    )


def research_section_family(record: ContextRecord) -> str:
    """Return the coarse Research AI section family for selector balancing."""

    meta = record.metadata
    values: list[str] = []
    for field in ("section_type", "evidence_type", "section_title", "document_type"):
        value = meta.get(field)
        if value is not None:
            values.extend(tokens(str(value).replace("-", "_")))
    for value in values:
        normalized_value = value.lower()
        if normalized_value in RESEARCH_SECTION_FAMILY_MAP:
            return RESEARCH_SECTION_FAMILY_MAP[normalized_value]
    if record.source_type == "paper_abstract":
        return "overview"
    return "other"


def infer_research_target_families(query_tokens: set[str]) -> list[str]:
    """Infer target Research AI section families from prompt-visible query terms."""

    families: list[str] = []
    for family in ("method", "results", "limitations", "overview"):
        if query_tokens & RESEARCH_TARGET_TERMS[family]:
            families.append(family)
    if "abstract" in query_tokens and "introduction" in query_tokens:
        families = ["overview", *[family for family in families if family != "overview"]]
    if not families:
        families = ["overview", "method", "results"]
    return list(dict.fromkeys(families))


def item_record_tokens(record: ContextRecord) -> set[str]:
    """Return selector tokens for a candidate."""

    return tokens(record_text(record))


def retail_selector_score(
    *,
    item: RankedCandidate,
    record: ContextRecord,
    query_tokens: set[str],
    query_text: str,
) -> float:
    """Score Retail final selection with product/category/intent focus."""

    candidate_tokens = item_record_tokens(record)
    meta = record.metadata
    product_tokens = metadata_tokens(record, ("product_title", "title"))
    category_tokens = metadata_tokens(record, ("category",))
    evidence_kind = normalize(str(meta.get("evidence_type") or meta.get("record_type") or ""))
    query_norm = normalize(query_text)
    score = item[1]
    score += 2.0 * lexical_overlap(query_tokens, product_tokens)
    score += 0.8 * lexical_overlap(query_tokens, category_tokens)
    score += 0.7 * lexical_overlap(query_tokens, candidate_tokens)
    if "review" in query_norm and "review" in evidence_kind:
        score += 1.2
    if "summary" in query_norm and ("summary" in evidence_kind or "review" in evidence_kind):
        score += 1.0
    if "policy" in query_norm and "policy" in evidence_kind:
        score += 1.0
    if evidence_kind in {"review", "summary"}:
        score += 2.4
    elif evidence_kind in {"policy", "multicategory", "metadata"}:
        score += 0.5
    else:
        score -= 2.2
    if "seed_expand" in record.context_id:
        score -= 1.6
    if ("return" in query_tokens or "refund" in query_tokens) and (
        {"return", "refund"} & candidate_tokens
    ):
        score += 0.7
    return score


def finance_selector_score(
    *,
    item: RankedCandidate,
    record: ContextRecord,
    query_tokens: set[str],
    query_text: str,
) -> float:
    """Score Finance final selection with company/metric/period/form focus."""

    candidate_tokens = item_record_tokens(record)
    meta = record.metadata
    score = item[1]
    ticker = str(meta.get("ticker") or "").lower()
    if ticker and ticker in query_tokens:
        score += 2.2
    company_tokens = metadata_tokens(record, ("company_name", "company", "registrant_name"))
    score += 1.2 * lexical_overlap(query_tokens, company_tokens)
    form_tokens = metadata_tokens(record, ("form", "filing_form", "document_type"))
    score += 1.0 * lexical_overlap(query_tokens, form_tokens)
    concept_tokens = metadata_tokens(record, ("concept", "concepts", "metric", "metric_family"))
    score += 1.6 * lexical_overlap(query_tokens, concept_tokens | candidate_tokens)
    period_tokens = metadata_tokens(
        record,
        ("period", "fiscal_year", "fiscal_period", "report_date", "filing_date"),
    )
    score += 1.0 * lexical_overlap(query_tokens, period_tokens)
    section_tokens = metadata_tokens(record, ("section_type", "section", "statement_type"))
    score += 0.8 * lexical_overlap(query_tokens, section_tokens)
    if "10 k" in normalize(query_text) and "10" in form_tokens and "k" in form_tokens:
        score += 0.7
    if "10 q" in normalize(query_text) and "10" in form_tokens and "q" in form_tokens:
        score += 0.7
    tags = meta.get("tags", [])
    tag_text = " ".join(str(tag) for tag in tags if tag) if isinstance(tags, list) else str(tags)
    document_kind = normalize(
        " ".join(
            str(value)
            for value in (
                meta.get("document_type"),
                meta.get("source_type"),
                record.source_type,
                record.title,
                tag_text,
            )
            if value
        )
    )
    if "8 k" in normalize(query_text) and "filing event" in document_kind:
        score += 3.2
    if (
        "8 k" in normalize(query_text)
        and "results of operations" in document_kind
        and not (query_tokens & {"revenue", "sales", "income", "margin", "cash", "flow"})
    ):
        score -= 1.4
    return score


def research_selector_score(
    *,
    item: RankedCandidate,
    record: ContextRecord,
    query_tokens: set[str],
) -> float:
    """Score Research AI final selection with paper and section-family focus."""

    candidate_tokens = item_record_tokens(record)
    meta = record.metadata
    score = item[1]
    section_tokens = metadata_tokens(record, ("section_type", "section_title", "section"))
    title_tokens = metadata_tokens(record, ("paper_title", "title"))
    family = research_section_family(record)
    target_families = set(infer_research_target_families(query_tokens))
    score += 1.3 * lexical_overlap(query_tokens, section_tokens)
    score += 1.2 * lexical_overlap(query_tokens, title_tokens)
    score += 0.5 * lexical_overlap(query_tokens, candidate_tokens)
    if family in target_families:
        score += 1.15
    if family == "overview" and {"abstract", "introduction"} & query_tokens:
        score += 0.65
    if family == "other":
        score -= 0.4
    if meta.get("paper_title") and title_tokens & query_tokens:
        score += 0.55
    return score


def airline_selector_score(
    *,
    item: RankedCandidate,
    record: ContextRecord,
    query_tokens: set[str],
) -> float:
    """Score Airline final selection for policy issue and MRR ordering."""

    candidate_tokens = item_record_tokens(record)
    meta_tokens = metadata_tokens(record, ("policy_tags", "category", "support_type"))
    score = item[1]
    score += 1.3 * lexical_overlap(query_tokens, meta_tokens)
    score += 0.7 * lexical_overlap(query_tokens, candidate_tokens)
    if query_tokens & {"refund", "cancel", "cancellation"} and candidate_tokens & {
        "refund",
        "cancel",
        "cancellation",
    }:
        score += 0.9
    if query_tokens & {"baggage", "bag"} and candidate_tokens & {"baggage", "bag"}:
        score += 0.9
    if query_tokens & {"fraud", "chargeback"} and candidate_tokens & {"fraud", "chargeback"}:
        score += 0.9
    return score


def healthcare_selector_score(
    *,
    item: RankedCandidate,
    record: ContextRecord,
    query_tokens: set[str],
) -> float:
    """Score Healthcare Admin conservatively to avoid harming the current path."""

    candidate_tokens = item_record_tokens(record)
    meta_tokens = metadata_tokens(
        record,
        (
            "support_type",
            "department",
            "safety_boundary",
            "category",
            "procedure_type",
        ),
    )
    score = item[1]
    score += 0.8 * lexical_overlap(query_tokens, meta_tokens)
    score += 0.35 * lexical_overlap(query_tokens, candidate_tokens)
    if query_tokens & {"privacy", "authorization", "consent"} and candidate_tokens & {
        "privacy",
        "authorization",
        "consent",
    }:
        score += 0.5
    return score


def ranked_with_scores(
    *,
    ranked: list[RankedCandidate],
    records_by_id: dict[str, ContextRecord],
    query_tokens: set[str],
    query_text: str,
) -> list[RankedCandidate]:
    """Return candidates with canonical selector scores."""

    scored: list[RankedCandidate] = []
    for item in ranked:
        context_id, _score, breakdown = item
        record = records_by_id[context_id]
        if record.vertical == "retail":
            score = retail_selector_score(
                item=item,
                record=record,
                query_tokens=query_tokens,
                query_text=query_text,
            )
        elif record.vertical == "finance":
            score = finance_selector_score(
                item=item,
                record=record,
                query_tokens=query_tokens,
                query_text=query_text,
            )
        elif record.vertical == "research_ai":
            score = research_selector_score(
                item=item,
                record=record,
                query_tokens=query_tokens,
            )
        elif record.vertical == "airline":
            score = airline_selector_score(item=item, record=record, query_tokens=query_tokens)
        elif record.vertical == "healthcare_admin":
            score = healthcare_selector_score(
                item=item,
                record=record,
                query_tokens=query_tokens,
            )
        else:
            score = item[1]
        new_breakdown = dict(breakdown)
        new_breakdown["canonical_selector_score"] = float(score - item[1])
        scored.append((context_id, score, new_breakdown))
    return sorted(scored, key=lambda candidate: (-candidate[1], candidate[0]))


def select_research_ai_section_candidates(
    *,
    rescored: list[RankedCandidate],
    records_by_id: dict[str, ContextRecord],
    query_tokens: set[str],
    final_top_k: int,
) -> list[RankedCandidate]:
    """Select Research AI top-k with target paper and section-family coverage."""

    research_ranked = [
        item
        for item in rescored
        if item[1] > 0 and records_by_id[item[0]].vertical == "research_ai"
    ]
    if not research_ranked:
        return rescored

    target_families = infer_research_target_families(query_tokens)
    paper_scores: dict[str, float] = {}
    for index, item in enumerate(research_ranked[:50]):
        record = records_by_id[item[0]]
        paper_key = research_paper_key(record)
        if not paper_key:
            continue
        title_overlap = lexical_overlap(
            query_tokens,
            metadata_tokens(record, ("paper_title", "title")),
        )
        section_bonus = 0.35 if research_section_family(record) in target_families else 0.0
        paper_scores[paper_key] = paper_scores.get(paper_key, 0.0) + (
            (item[1] / (index + 1)) + (2.4 * title_overlap) + section_bonus
        )
    anchor_papers = [
        paper
        for paper, _score in sorted(
            paper_scores.items(),
            key=lambda pair: (-pair[1], pair[0]),
        )[:2]
    ]

    selected: list[RankedCandidate] = []
    selected_ids: set[str] = set()
    selected_texts: set[str] = set()
    selected_family_counts: dict[str, int] = {}

    def add_candidate(item: RankedCandidate) -> None:
        if len(selected) >= final_top_k:
            return
        context_id = item[0]
        if context_id in selected_ids:
            return
        record = records_by_id[context_id]
        normalized_text = normalize(record.text)
        if normalized_text in selected_texts:
            return
        selected.append(item)
        selected_ids.add(context_id)
        selected_texts.add(normalized_text)
        family = research_section_family(record)
        selected_family_counts[family] = selected_family_counts.get(family, 0) + 1

    def candidates_for(
        *,
        paper_key: str | None = None,
        family: str | None = None,
    ) -> list[RankedCandidate]:
        candidates: list[RankedCandidate] = []
        for item in research_ranked:
            record = records_by_id[item[0]]
            if paper_key is not None and research_paper_key(record) != paper_key:
                continue
            if family is not None and research_section_family(record) != family:
                continue
            candidates.append(item)
        return candidates

    for paper_key in anchor_papers[:1]:
        for family in target_families:
            for item in candidates_for(paper_key=paper_key, family=family)[:2]:
                add_candidate(item)
                if len(selected) >= final_top_k:
                    break
            if len(selected) >= final_top_k:
                break

        if "overview" in target_families:
            for family in ("overview", "method", "results"):
                if selected_family_counts.get(family, 0) > 0:
                    continue
                for item in candidates_for(paper_key=paper_key, family=family)[:1]:
                    add_candidate(item)
                    break

    for paper_key in anchor_papers:
        for item in candidates_for(paper_key=paper_key):
            add_candidate(item)
            if len(selected) >= final_top_k:
                break
        if len(selected) >= final_top_k:
            break

    if len(selected) < final_top_k:
        for family in target_families:
            for item in candidates_for(family=family):
                add_candidate(item)
                if len(selected) >= final_top_k:
                    break
            if len(selected) >= final_top_k:
                break

    for item in research_ranked:
        add_candidate(item)
        if len(selected) >= final_top_k:
            break

    if len(selected) < final_top_k:
        for item in rescored:
            add_candidate(item)
            if len(selected) >= final_top_k:
                break

    selected_id_set = {item[0] for item in selected}
    remainder = [item for item in rescored if item[0] not in selected_id_set]
    return [*selected, *remainder]


def select_canonical_final_candidates(
    *,
    ranked: list[RankedCandidate],
    records_by_id: dict[str, ContextRecord],
    query_tokens: set[str],
    query_text: str,
    final_top_k: int,
) -> list[RankedCandidate]:
    """Return candidates with canonical vertical final selection applied."""

    if not ranked:
        return ranked
    rescored = ranked_with_scores(
        ranked=ranked,
        records_by_id=records_by_id,
        query_tokens=query_tokens,
        query_text=query_text,
    )
    if any(records_by_id[item[0]].vertical == "research_ai" for item in rescored):
        rescored = select_research_ai_section_candidates(
            rescored=rescored,
            records_by_id=records_by_id,
            query_tokens=query_tokens,
            final_top_k=final_top_k,
        )
    selected: list[RankedCandidate] = []
    selected_ids: set[str] = set()
    seen_texts: set[str] = set()

    for item in rescored:
        context_id = item[0]
        record = records_by_id[context_id]
        if context_id in selected_ids:
            continue
        normalized_text = normalize(record.text)
        if record.vertical in {"retail", "research_ai"} and normalized_text in seen_texts:
            continue
        selected.append(item)
        selected_ids.add(context_id)
        seen_texts.add(normalized_text)
        if len(selected) >= final_top_k:
            break

    if len(selected) < final_top_k:
        for item in rescored:
            if item[0] in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(item[0])
            if len(selected) >= final_top_k:
                break

    selected_id_set = {item[0] for item in selected}
    remainder = [item for item in rescored if item[0] not in selected_id_set]
    return [*selected, *remainder]
