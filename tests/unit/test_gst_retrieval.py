import math

from app.datasets.gst.document_mapper import map_provision_to_points, source_label
from eval.gst_retrieval import (
    corpus_keys,
    coverage,
    gold_sets,
    retrieval_metrics,
    retrieved_ids,
    score_row,
    source_matches,
    split_report,
)
from eval.reporting import aggregate

CORPUS_IDS = {"cgst_act_2017.section.7", "cgst_act_2017.section.8", "cgst_act_2017.section.2"}
LABELS = {
    "The Central Goods and Services Tax Act, 2017, Section 7": "cgst_act_2017.section.7",
    "The Central Goods and Services Tax Act, 2017, Section 8": "cgst_act_2017.section.8",
    "The Central Goods and Services Tax Act, 2017, Section 2": "cgst_act_2017.section.2",
}


def _golden(**over):
    g = {
        "id": "q1",
        "gold_primary_sources": ["cgst_act_2017.section.8"],
        "gold_supporting_sources": ["cgst_act_2017.section.2"],
        "missing_gold_sources": [],
        "primary_required_sources": ["cgst_act_2017.section.8", "cgst_act_2017.section.7"],
        "supporting_sources": ["cgst_act_2017.section.2"],
        "external_sources_missing_from_corpus": ["Circular No. 11/11/2017-GST"],
    }
    g.update(over)
    return g


def test_source_label_matches_ingested_payload():
    record = {
        "id": "cgst_act_2017.section.7", "instrument_title": "The Central Goods and Services Tax Act, 2017",
        "provision_type": "section", "provision_number": "7", "raw_text": "supply includes...",
    }
    assert map_provision_to_points(record)[0]["source"] == source_label(record)
    assert source_label(record) == "The Central Goods and Services Tax Act, 2017, Section 7"


def test_gold_sets_reads_each_annotation_version():
    g = _golden()
    assert gold_sets(g, "original") == (["cgst_act_2017.section.8"], ["cgst_act_2017.section.2"], [])
    assert gold_sets(g, "corrected")[0] == ["cgst_act_2017.section.8", "cgst_act_2017.section.7"]


def test_coverage_buckets():
    assert coverage(_golden(), CORPUS_IDS, "original") == "full"
    assert coverage(_golden(), CORPUS_IDS, "corrected") == "partial"
    assert coverage(_golden(primary_required_sources=[]), CORPUS_IDS, "corrected") == "none"
    assert coverage(_golden(gold_primary_sources=["igst_act_2017.section.10"]), CORPUS_IDS, "original") == "none"


def test_retrieved_ids_keeps_rank_order_dedupes_and_drops_non_corpus():
    sources = [
        "The Central Goods and Services Tax Act, 2017, Section 2",
        "https://example.com/web-fallback",
        "The Central Goods and Services Tax Act, 2017, Section 8",
        "The Central Goods and Services Tax Act, 2017, Section 2",
    ]
    assert retrieved_ids(sources, LABELS) == ["cgst_act_2017.section.2", "cgst_act_2017.section.8"]


def test_retrieval_metrics():
    primary = ["cgst_act_2017.section.8", "cgst_act_2017.section.7"]
    m = retrieval_metrics(["cgst_act_2017.section.2", "cgst_act_2017.section.8"], primary, ["cgst_act_2017.section.2"])
    assert m == {
        "required_recall": 0.5, "all_required_hit": 0.0, "any_required_hit": 1.0,
        "required_mrr": 0.5, "supporting_recall": 1.0,
    }
    assert retrieval_metrics([], primary, [])["required_mrr"] == 0.0
    assert retrieval_metrics(["x"], [], [])["required_recall"] is None


def test_score_row_prefers_ranked_sources_and_split_report():
    row = {
        "id": "q1",
        "actual_sources": ["The Central Goods and Services Tax Act, 2017, Section 8"],
        "retrieved_sources": [
            "The Central Goods and Services Tax Act, 2017, Section 7",
            "The Central Goods and Services Tax Act, 2017, Section 8",
        ],
        "ragas_metrics": {"faithfulness": 0.5, "context_precision": 1.0, "context_recall": float("nan"),
                          "answer_relevancy": 0.8},
    }
    score_row(row, _golden(), CORPUS_IDS, LABELS)
    assert row["rank_order_known"] is True
    assert row["retrieved_ids"] == ["cgst_act_2017.section.7", "cgst_act_2017.section.8"]
    assert row["coverage"] == {"original": "full", "corrected": "partial"}
    assert row["retrieval"]["original"]["required_mrr"] == 0.5
    assert row["retrieval"]["corrected"]["all_required_hit"] == 1.0

    report = split_report([row], "corrected")
    assert report["all"]["n"] == 1 and report["partial"]["n"] == 1 and report["full"]["n"] == 0
    assert report["partial"]["required_recall"] == 1.0
    assert report["full"]["required_recall"] is None
    assert report["partial"]["context_recall"] is None  # only value was NaN


def test_aggregate_ignores_nan():
    rows = [{"ragas_metrics": {"context_recall": float("nan")}}, {"ragas_metrics": {"context_recall": 0.4}}]
    out = aggregate(rows)
    assert out["context_recall"] == 0.4 and not math.isnan(out["context_recall"])


def test_instrument_level_gold_source_matches_any_of_its_provisions():
    assert source_matches("notification_12_2017_ct_rate", "notification_12_2017_ct_rate.entry.12")
    assert not source_matches("notification_12_2017_ct_rate", "notification_12_2017_ct_rate_x.entry.1")
    assert not source_matches("cgst_act_2017.section.1", "cgst_act_2017.section.10")

    corpus = [{"id": "notification_12_2017_ct_rate.entry.12", "instrument_id": "notification_12_2017_ct_rate"},
              {"id": "cgst_act_2017.section.8", "instrument_id": "cgst_act_2017"}]
    keys = corpus_keys(corpus)
    g = _golden(primary_required_sources=["cgst_act_2017.section.8", "notification_12_2017_ct_rate"],
                external_sources_missing_from_corpus=[])
    assert coverage(g, keys, "corrected") == "full"
    m = retrieval_metrics(["cgst_act_2017.section.2", "notification_12_2017_ct_rate.entry.12"],
                          ["notification_12_2017_ct_rate"], [])
    assert m["required_recall"] == 1.0 and m["required_mrr"] == 0.5
