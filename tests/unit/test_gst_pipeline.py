from scripts.gst_pipeline.enrich import extract_explicit_references, hierarchy_prefix
from scripts.gst_pipeline.parse_bare_law import parse_bare_law
from scripts.gst_pipeline.parse_notifications import build_records, split_long_segment

BOOK = """
## THE CENTRAL GOODS AND SERVICES TAX ACT, 2017

## CHAPTER I : PRELIMINARY

## Section 7. Scope of supply.-

(1) For the purposes of this Act, the expression "supply" includes all forms of supply.

## SCHEDULE I

[ See section 7 ]

## ACTIVITIES TO BE TREATED AS SUPPLY EVEN IF MADE WITHOUT CONSIDERATION

(1) Permanent transfer or disposal of business assets.

## POINTS TO NOTE

## 1. Section 10 of the CGST Act, 2017 - Composition levy

Editorial note that must not end up in Schedule I.

## THE CENTRAL GOODS AND SERVICES TAX RULES, 2017

## CHAPTER I : PRELIMINARY

## Rule 88B. Manner of calculating interest.-

Interest text.

## 27 [Rule 88 C. Manner of dealing with difference in liability.-

Rule 88C text.

## THE INTEGRATED GOODS AND SERVICES TAX ACT, 2017

## CHAPTER IV : DETERMINATION OF NATURE OF SUPPLY

## Section 7. Inter-State supply.-

(1) Subject to the provisions of section 10, supply of goods ...
"""


def test_parse_bare_law_schedules_instruments_and_spaced_rule_numbers():
    records, _ = parse_bare_law(BOOK)
    by_id = {r["id"]: r for r in records}
    assert set(by_id) == {
        "cgst_act_2017.section.7",
        "cgst_act_2017.schedule.1",
        "cgst_rules_2017.rule.88b",
        "cgst_rules_2017.rule.88c",
        "igst_act_2017.section.7",
    }
    schedule = by_id["cgst_act_2017.schedule.1"]
    assert schedule["provision_number"] == "I"
    assert (
        schedule["provision_title"]
        == "Activities to be treated as supply even if made without consideration"
    )
    assert "Editorial note" not in schedule["raw_text"]
    assert by_id["cgst_rules_2017.rule.88b"]["raw_text"] == "Interest text."
    assert (
        by_id["igst_act_2017.section.7"]["instrument_title"]
        == "The Integrated Goods and Services Tax Act, 2017"
    )


def test_references_resolve_to_the_right_instrument():
    igst = {"id": "igst_act_2017.section.20", "instrument_id": "igst_act_2017"}
    refs = extract_explicit_references(
        "subject to section 10 of this Act and section 54 of the Central Goods and Services Tax Act",
        igst,
    )
    assert [r["target_id"] for r in refs["sections"]] == [
        "cgst_act_2017.section.54",
        "igst_act_2017.section.10",
    ]
    cgst = {"id": "cgst_rules_2017.rule.59", "instrument_id": "cgst_rules_2017"}
    refs = extract_explicit_references(
        "under section 14 of the Integrated Goods and Services Tax Act and rule 60", cgst
    )
    assert [r["target_id"] for r in refs["sections"]] == ["igst_act_2017.section.14"]
    assert [r["target_id"] for r in refs["rules"]] == ["cgst_rules_2017.rule.60"]


def test_hierarchy_prefix_with_and_without_chapter():
    act = {
        "instrument_title": "The Central Goods and Services Tax Act, 2017",
        "chapter_number": "I",
        "chapter_title": "PRELIMINARY",
        "provision_type": "section",
        "provision_number": "1",
        "provision_title": "Short title",
    }
    assert hierarchy_prefix(act) == (
        "The Central Goods and Services Tax Act, 2017. Chapter I: PRELIMINARY. "
        "Section 1: Short title."
    )
    entry = {
        "instrument_title": "Notification No. 12/2017-Central Tax (Rate)",
        "instrument_subject": "Exemptions on supply of services",
        "provision_type": "entry",
        "provision_number": "12",
        "provision_title": "Renting of residential dwelling",
    }
    assert hierarchy_prefix(entry) == (
        "Notification No. 12/2017-Central Tax (Rate). Exemptions on supply of "
        "services. Entry 12: Renting of residential dwelling."
    )


NOTIFICATION_MD = """## [TO BE PUBLISHED IN THE GAZETTE OF INDIA, EXTRAORDINARY, PART II, SECTION 3, SUB-SECTION (i)]

Government of India Ministry of Finance (Department of Revenue)

## Notification No. 13/2017- Central Tax (Rate)

New Delhi, the 28 th June, 2017

G.S.R.(E).- In exercise of the powers conferred by section 9 the Central Government notifies the Table below:-

| Sl. No.   | Category of Supply of Services   | Supplier of service   | Recipient of Service   |
|-----------|----------------------------------|-----------------------|------------------------|
| (1)       | (2)                              | (3)                   | (4)                    |
| 1         | Services by a goods transport    | GTA                   | Any factory            |

|   | agency in respect of transport of goods by road. |   | located in the taxable territory. |
|---|---|---|---|
| [2A | Services supplied by an arbitral tribunal. | An arbitral tribunal. | Any business entity. |

2. This notification shall come into force on the 1st day of July, 2017, and applies to every supply of the services listed above by any person to any recipient in the taxable territory.

(Ruchi Bisht)

Under Secretary to the Government of India
"""


def test_notification_rows_merge_across_page_breaks_and_boilerplate_is_dropped():
    src = {
        "instrument_id": "notification_13_2017_ct_rate",
        "kind": "notification",
        "number": "13/2017-Central Tax (Rate)",
        "date": "2017-06-28",
        "subject": "Reverse charge services",
        "url": "https://example/13.pdf",
    }
    records = {r["id"]: r for r in build_records(src, NOTIFICATION_MD)}
    assert set(records) == {
        "notification_13_2017_ct_rate.paragraph.1",
        "notification_13_2017_ct_rate.paragraph.2",
        "notification_13_2017_ct_rate.entry.1",
        "notification_13_2017_ct_rate.entry.2a",
    }
    entry = records["notification_13_2017_ct_rate.entry.1"]
    assert (
        "Category of Supply of Services: Services by a goods transport agency in respect of transport"
        in entry["raw_text"]
    )
    assert (
        "Recipient of Service: Any factory located in the taxable territory." in entry["raw_text"]
    )
    assert entry["instrument_title"] == "Notification No. 13/2017-Central Tax (Rate)"
    preamble = records["notification_13_2017_ct_rate.paragraph.1"]["raw_text"]
    assert preamble.startswith("G.S.R.") and "Government of India" not in preamble
    assert "Under Secretary" not in records["notification_13_2017_ct_rate.paragraph.2"]["raw_text"]


def test_long_definition_paragraph_splits_at_clause_sequence():
    lines = [f"({c}) '{c}-term' means " + "word " * 150 for c in "abcdef"]
    lines.insert(2, "(ii) a sub-item that is not a clause boundary " + "word " * 10)
    parts = split_long_segment({"number": "2", "lines": lines})
    assert [p["number"] for p in parts] == ["2(a-c)", "2(d-f)"]
    assert any(line.startswith("(ii)") for line in parts[0]["lines"])
