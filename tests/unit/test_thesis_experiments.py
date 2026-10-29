"""Unit tests for the thesis experiment components: BM25, the provision graph and its post-processors,
the citation parser and the paired statistics. No external services."""

import math

from app.services.gst_graph import ProvisionGraph, cap_doc_types, expand_with_graph
from app.services.sparse_vector_service import BM25Index
from eval.citation_check import extract_citations, score_answer
from eval.stats import paired


def _record(pid, title, number, ptype="section", refs=(), instrument_kind=None):
    instrument = pid.rsplit(".", 2)[0]
    rec = {
        "id": pid, "instrument_id": instrument, "instrument_title": title, "provision_type": ptype,
        "provision_number": number, "retrieval_text_enriched": f"{title} {ptype} {number} text",
        "explicit_references": {"sections": [{"number": r.rsplit(".", 1)[-1], "target_id": r} for r in refs]},
    }
    if instrument_kind:
        rec["instrument_kind"] = instrument_kind
    return rec


CGST = "The Central Goods and Services Tax Act, 2017"
RATE = "Notification No. 12/2017-Central Tax (Rate)"
RECORDS = [
    _record("cgst_act_2017.section.9", CGST, "9"),
    _record("cgst_act_2017.section.11", CGST, "11"),
    _record("cgst_act_2017.section.16", CGST, "16", refs=["cgst_act_2017.section.9"]),
    _record("notification_12_2017_ct_rate.paragraph.1", RATE, "1", "paragraph",
            refs=["cgst_act_2017.section.11"], instrument_kind="notification"),
    _record("notification_12_2017_ct_rate.entry.1", RATE, "1", "entry", instrument_kind="notification"),
    _record("notification_12_2017_ct_rate.entry.2", RATE, "2", "entry", instrument_kind="notification"),
]


def _chunk(graph, pid, score=1.0):
    return graph.chunk_for(pid, score)


def test_bm25_prefers_rare_terms_and_keeps_numbers():
    docs = [{"text": "levy and collection section 9 tax", "source": "a"},
            {"text": "input tax credit eligibility conditions tax", "source": "b"},
            {"text": "tax tax tax tax returns", "source": "c"}]
    index = BM25Index()
    index.fit(docs)
    assert [c.source for c in index.search("section 9 levy")][0] == "a"
    assert [c.source for c in index.search("credit")] == ["b"]
    assert index.search("unknownword") == []


def test_graph_links_references_and_instrument_entries():
    g = ProvisionGraph(RECORDS)
    assert "cgst_act_2017.section.9" in g.neighbours["cgst_act_2017.section.16"]
    assert "cgst_act_2017.section.16" in g.neighbours["cgst_act_2017.section.9"]  # both directions
    # entries inherit the enabling section from the instrument's preamble paragraph
    assert "cgst_act_2017.section.11" in g.neighbours["notification_12_2017_ct_rate.entry.1"]


def test_expand_with_graph_adds_neighbours_after_direct_hits():
    g = ProvisionGraph(RECORDS)
    chunks = [_chunk(g, "notification_12_2017_ct_rate.entry.1"), _chunk(g, "notification_12_2017_ct_rate.entry.2")]
    out = [g.id_of(c) for c in expand_with_graph(chunks, g)]
    assert out[:2] == ["notification_12_2017_ct_rate.entry.1", "notification_12_2017_ct_rate.entry.2"]
    assert "cgst_act_2017.section.11" in out


def test_cap_doc_types_limits_subordinate_items_and_fills_up():
    g = ProvisionGraph(RECORDS)
    chunks = [_chunk(g, p) for p in ("notification_12_2017_ct_rate.entry.1", "notification_12_2017_ct_rate.entry.2",
                                     "cgst_act_2017.section.9", "cgst_act_2017.section.16")]
    assert [g.id_of(c) for c in cap_doc_types(chunks, g, top_k=3, cap=1)] == [
        "notification_12_2017_ct_rate.entry.1", "cgst_act_2017.section.9", "cgst_act_2017.section.16"]
    # not enough non-subordinate items: fill with the skipped ones
    assert len(cap_doc_types(chunks[:2], g, top_k=2, cap=0)) == 2


def test_extract_citations_resolves_acts_rules_notifications_circulars():
    keys = {"cgst_act_2017.section.16", "igst_act_2017.section.12", "notification_12_2017_ct_rate",
            "circular_92_11_2019_gst", "cgst_rules_2017.rule.36", "cgst_act_2017.schedule.2"}
    text = ("As per Section 16(2)(b) of the CGST Act read with rule 36(4), and section 12 of the IGST Act; "
            "see Schedule II, Notification No. 12/2017-Central Tax (Rate) and Circular No. 92/11/2019-GST.")
    assert extract_citations(text, keys) == [
        "cgst_act_2017.section.16", "igst_act_2017.section.12", "cgst_rules_2017.rule.36",
        "cgst_act_2017.schedule.2", "notification_12_2017_ct_rate", "circular_92_11_2019_gst"]


def test_score_answer_grounding_and_gold_recall():
    keys = {"cgst_act_2017.section.16", "cgst_act_2017.section.17", "notification_12_2017_ct_rate"}
    s = score_answer("Section 16 and section 17 apply; also Notification 12/2017-CT (Rate) and section 999.",
                     retrieved=["cgst_act_2017.section.16", "notification_12_2017_ct_rate.entry.3"],
                     gold_required=["cgst_act_2017.section.16", "cgst_act_2017.section.18"], keys=keys)
    assert s["n_citations"] == 4
    assert math.isclose(s["grounded_precision"], 0.5)
    assert s["outside_context"] == 2 and s["unknown_law"] == 1
    assert math.isclose(s["gold_citation_recall"], 0.5)


def test_paired_stats_direction_and_ci():
    def rows(vals):
        return [{"id": str(i), "coverage": {"corrected": "full"},
                 "retrieval": {"corrected": {"required_recall": v}}} for i, v in enumerate(vals)]
    r = paired(rows([0.0] * 30), rows([1.0] * 20 + [0.0] * 10), "required_recall")
    assert r["n"] == 30 and r["b_better"] == 20 and r["a_better"] == 0
    assert r["ci_low"] > 0 and r["p_wilcoxon"] < 0.05
