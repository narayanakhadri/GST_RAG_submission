"""Stage 1b (download): fetch the notifications and circulars the golden set cites, as issued.

Source: gstcouncil.gov.in. For the main services rate notifications (12, 13, 17/2017-CT(R) and
9, 10/2017-IT(R)) we use the Council's consolidated reference copies "as amended up to 01.04.2019"
(gstcouncil.gov.in/gst-rates-services) — closest to the law position the golden answers assume. Those
copies carry a disclaimer that only the Gazette text has legal force, and amendments after
01.04.2019 are not folded in. Everything else is the English PDF *as originally issued* from the
Council's "CGST Tax Notification" / "CGST Circulars" listings. CBIC's own consolidated texts
(taxinformation.cbic.gov.in) are not downloadable without a browser session.

Each PDF is saved to seed/docs/gst/sources/pdf/ and converted with Docling to Markdown in
seed/docs/gst/sources/md/ (tables kept as Markdown tables), which parse_notifications.py reads.
A manifest with URL, SHA-256 and metadata is written next to them.

    uv run python -m scripts.gst_pipeline.fetch_sources            # download + convert missing ones
    uv run python -m scripts.gst_pipeline.fetch_sources --reconvert
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

SOURCES_DIR = Path("seed/docs/gst/sources")
GC = "https://gstcouncil.gov.in/sites/default/files"

# instrument_id, kind, number, issue date, subject (from the GST Council listing), PDF URL; law_position
# defaults to "as issued". table_text="pdf": split the table from the PDF text layer instead of Docling's
# table model, which scrambles cells in these long consolidated copies (see parse_notifications.py);
# table_text="faq_pdf": a two-column question | answer FAQ, columns cut from the PDF layout text.
SOURCES: list[dict[str, str]] = [
    {
        "instrument_id": "notification_12_2017_ct_rate",
        "kind": "notification",
        "number": "12/2017-Central Tax (Rate)",
        "date": "2017-06-28",
        "table_text": "pdf",
        "law_position": "as amended up to 01.04.2019 (GST Council consolidated reference copy)",
        "subject": "Exemptions on supply of services under the CGST Act",
        "url": f"{GC}/2024-02/12-exemption_cgst-01.04.2019-revised.pdf",
    },
    {
        "instrument_id": "notification_13_2017_ct_rate",
        "kind": "notification",
        "number": "13/2017-Central Tax (Rate)",
        "date": "2017-06-28",
        "table_text": "pdf",
        "law_position": "as amended up to 01.04.2019 (GST Council consolidated reference copy)",
        "subject": "Categories of services on which tax is payable under reverse charge mechanism under the CGST Act",
        "url": f"{GC}/2024-02/13-reverse_charge-cgst-01.04.2019-revised.pdf",
    },
    {
        "instrument_id": "notification_17_2017_ct_rate",
        "kind": "notification",
        "number": "17/2017-Central Tax (Rate)",
        "date": "2017-06-28",
        "law_position": "as amended (GST Council consolidated reference copy)",
        "subject": "Categories of services the tax on intra-State supplies of which shall be paid by the electronic commerce operator",
        "url": f"{GC}/2024-02/17-eco_notification-cgst.pdf",
    },
    {
        "instrument_id": "notification_9_2017_it_rate",
        "kind": "notification",
        "number": "9/2017-Integrated Tax (Rate)",
        "date": "2017-06-28",
        "table_text": "pdf",
        "law_position": "as amended up to 01.04.2019 (GST Council consolidated reference copy)",
        "subject": "Exemptions on inter-State supply of services under the IGST Act",
        "url": f"{GC}/2024-02/9-mega-igst-01.04.2019.pdf",
    },
    {
        "instrument_id": "notification_10_2017_it_rate",
        "kind": "notification",
        "number": "10/2017-Integrated Tax (Rate)",
        "date": "2017-06-28",
        "table_text": "pdf",
        "law_position": "as amended up to 01.04.2019 (GST Council consolidated reference copy)",
        "subject": "Categories of services on which integrated tax is payable under reverse charge mechanism",
        "url": f"{GC}/2024-02/10-reverse_charge-igst-01.04.2019.pdf",
    },
    {
        "instrument_id": "notification_2_2019_ct_rate",
        "kind": "notification",
        "number": "2/2019-Central Tax (Rate)",
        "date": "2019-03-07",
        "subject": "Composition scheme for suppliers of services (6% rate) with annual turnover in the preceding year up to Rs 50 lakhs",
        "url": f"{GC}/2024-05/download_2024-05-17t161230.215.pdf",
    },
    {
        "instrument_id": "notification_14_2019_ct",
        "kind": "notification",
        "number": "14/2019-Central Tax",
        "date": "2019-03-07",
        "subject": "Threshold of aggregate turnover for availing the composition scheme under section 10 of the CGST Act (supersedes 8/2017-CT)",
        "url": f"{GC}/2024-05/notfctn-14-central-tax-english-2019.pdf",
    },
    {
        "instrument_id": "notification_10_2019_ct",
        "kind": "notification",
        "number": "10/2019-Central Tax",
        "date": "2019-03-07",
        "subject": "Exemption from registration for persons exclusively supplying goods with aggregate turnover up to Rs 40 lakhs",
        "url": f"{GC}/2024-05/notfctn-10-central-tax-english-2019.pdf",
    },
    {
        "instrument_id": "notification_66_2017_ct",
        "kind": "notification",
        "number": "66/2017-Central Tax",
        "date": "2017-11-15",
        "subject": "Exemption from payment of tax on advances received for supply of goods",
        "url": f"{GC}/2024-05/notfctn-66-central-tax-english.pdf",
    },
    {
        "instrument_id": "notification_16_2020_ct",
        "kind": "notification",
        "number": "16/2020-Central Tax",
        "date": "2020-03-23",
        "subject": "Central Goods and Services Tax (Third Amendment) Rules, 2020",
        "url": f"{GC}/2024-05/notfctn-16-central-tax-english-2020.pdf",
    },
    {
        "instrument_id": "notification_50_2018_ct",
        "kind": "notification",
        "number": "50/2018-Central Tax",
        "date": "2018-09-13",
        "subject": "Brings section 51 of the CGST Act (tax deduction at source) into force from 01.10.2018",
        "url": f"{GC}/2024-05/notfctn-50-central-tax-english.pdf",
    },
    {
        "instrument_id": "circular_92_11_2019_gst",
        "kind": "circular",
        "number": "92/11/2019-GST",
        "date": "2019-03-07",
        "subject": "Treatment of sales promotion schemes under GST",
        "url": f"{GC}/2024-06/circular-cgst-92.pdf",
    },
    {
        "instrument_id": "circular_32_06_2018_gst",
        "kind": "circular",
        "number": "32/06/2018-GST",
        "date": "2018-02-12",
        "subject": "Clarifications regarding GST in respect of certain services as decided in the 25th GST Council meeting",
        "url": f"{GC}/2024-06/circularno-32-cgst.pdf",
    },
    {
        "instrument_id": "circular_103_22_2019_gst",
        "kind": "circular",
        "number": "103/22/2019-GST",
        "date": "2019-06-28",
        "subject": "Determination of place of supply in certain cases",
        "url": f"{GC}/2024-06/circular-cgst-103_0.pdf",
    },
    {
        "instrument_id": "circular_11_11_2017_gst",
        "kind": "circular",
        "number": "11/11/2017-GST",
        "date": "2017-10-20",
        "subject": "Taxability of printing contracts",
        "url": f"{GC}/2024-06/circularno-11-gst.pdf",
    },
    {
        "instrument_id": "circular_116_35_2019_gst",
        "kind": "circular",
        "number": "116/35/2019-GST",
        "date": "2019-10-11",
        "subject": "GST on display of the donor's name in the premises of charitable organisations receiving donations",
        "url": f"{GC}/2024-06/circular-cgst-116.pdf",
    },
    {
        "instrument_id": "circular_119_38_2019_gst",
        "kind": "circular",
        "number": "119/38/2019-GST",
        "date": "2019-10-11",
        "subject": "Taxability of supply of securities under the Securities Lending Scheme, 1997",
        "url": f"{GC}/2024-06/circular-cgst-119.pdf",
    },
    {
        "instrument_id": "circular_140_10_2020_gst",
        "kind": "circular",
        "number": "140/10/2020-GST",
        "date": "2020-06-10",
        "subject": "Levy of GST on Director's remuneration",
        "url": f"{GC}/2024-06/circular_refund_140_10_2020.pdf",
    },
    {
        "instrument_id": "circular_178_10_2022_gst",
        "kind": "circular",
        "number": "178/10/2022-GST",
        "date": "2022-08-03",
        "subject": "GST on liquidated damages, compensation and penalty arising out of breach of contract",
        "url": f"{GC}/2024-06/cir-178-08-2022-cgst.pdf",
    },
    {
        "instrument_id": "circular_44_18_2018_cgst",
        "kind": "circular",
        "number": "44/18/2018-CGST",
        "date": "2018-05-02",
        "subject": "Taxability of tenancy rights under GST",
        "url": f"{GC}/2024-06/circular_no.44.pdf",
    },
    {
        "instrument_id": "circular_47_21_2018_gst",
        "kind": "circular",
        "number": "47/21/2018-GST",
        "date": "2018-06-08",
        "subject": "Clarifications of certain issues under GST",
        "url": f"{GC}/2024-06/circular_no.47.pdf",
    },
    {
        "instrument_id": "circular_57_31_2018_gst",
        "kind": "circular",
        "number": "57/31/2018-GST",
        "date": "2018-09-04",
        "subject": "Scope of principal-agent relationship in the context of Schedule I of the CGST Act",
        "url": f"{GC}/2024-06/circular_no.57.pdf",
    },
    {
        "instrument_id": "circular_76_50_2018_gst",
        "kind": "circular",
        "number": "76/50/2018-GST",
        "date": "2018-12-31",
        "subject": "Clarification on certain issues (sale by government departments to unregistered persons, penalty under section 73(11), rate on debit/credit notes, etc.)",
        "url": f"{GC}/2024-06/circular-no-76.pdf",
    },
    {
        "instrument_id": "circular_82_01_2019_gst",
        "kind": "circular",
        "number": "82/01/2019-GST",
        "date": "2019-01-01",
        "subject": "Applicability of GST on programmes conducted by the Indian Institutes of Management (IIMs)",
        "url": f"{GC}/2024-06/circular-cgst-82-new.pdf",
    },
    # Added 2026-10-19 from seed/docs/gst/sources/to_ingest.json (cited by the golden set / SME pre-check).
    {
        "instrument_id": "notification_11_2017_ct_rate",
        "kind": "notification",
        "number": "11/2017-Central Tax (Rate)",
        "date": "2017-06-28",
        "table_text": "pdf",
        "law_position": "as amended up to 01.04.2019 (GST Council consolidated reference copy)",
        "subject": "Rates of central tax on supply of services (incl. GTA forward-charge option)",
        "url": f"{GC}/2024-02/11-rate_notification-cgst-01.04.2019.pdf",
    },
    {
        "instrument_id": "notification_41_2017_it_rate",
        "kind": "notification",
        "number": "41/2017-Integrated Tax (Rate)",
        "date": "2017-10-23",
        "law_position": "as issued (copy hosted by CHEMEXCIL; not on gstcouncil.gov.in)",
        "subject": "Integrated tax at 0.1% on inter-State supply of goods to a registered recipient for export (merchant exporters)",
        "url": "https://chemexcil.in/uploads/files/Notification_No_41.pdf",
    },
    {
        "instrument_id": "notification_52_2018_ct",
        "kind": "notification",
        "number": "52/2018-Central Tax",
        "date": "2018-09-20",
        "subject": "Rate of tax collection at source (TCS) by electronic commerce operators under section 52",
        "url": f"{GC}/2024-05/notfctn-52-central-tax-english-new.pdf",
    },
    # CBIC FAQs the golden answers cite (TCS multiple-ECO model, one e-way bill per invoice). The e-way bill
    # portal blocks automated downloads; the West Bengal Commercial Tax department hosts the official FAQ sets.
    {
        "instrument_id": "faq_tcs_cbic",
        "kind": "faq",
        "title": "CBIC FAQs on Tax Collection at Source (TCS) under GST",
        "number": "FAQs on TCS (updated 30.11.2018)",
        "date": "2018-11-30",
        "subject": "Tax collection at source by electronic commerce operators under section 52 (rate, registration, multiple-ECO model)",
        "url": f"{GC}/2024-02/faqs_tcs.pdf",
        "table_text": "faq_pdf",
    },
    {
        "instrument_id": "faq_eway_bill_part1",
        "kind": "faq",
        "title": "FAQs on e-way bill (Part 1)",
        "number": "e-way bill FAQs, Part 1",
        "date": "2018-04-01",
        "law_position": "as issued (copy hosted by the West Bengal Commercial Tax department)",
        "subject": "E-way bill under section 68 and rule 138: who generates it, when it is required",
        "url": "https://comtax.wb.gov.in/GST/GST_e-Waybill/FAQ%20-%201.pdf",
    },
    {
        "instrument_id": "faq_eway_bill_part2",
        "kind": "faq",
        "title": "FAQs on e-way bill (Part 2)",
        "number": "e-way bill FAQs, Part 2",
        "date": "2018-04-01",
        "law_position": "as issued (copy hosted by the West Bengal Commercial Tax department)",
        "subject": "E-way bill: multiple invoices, consolidated e-way bill, transport modes",
        "url": "https://comtax.wb.gov.in/GST/GST_e-Waybill/FAQs%20-2.pdf",
    },
]


def download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    if not data.startswith(b"%PDF-"):
        raise ValueError(f"{url} did not return a PDF")
    return data


PAGES_PER_BATCH = 10  # Docling runs out of memory converting ~45-page PDFs in one go


def convert_to_markdown(converter, pdf_path: Path) -> str:
    """Docling Markdown for a PDF, converted in page batches; batches are joined with blank lines."""
    import pypdf

    pages = len(pypdf.PdfReader(str(pdf_path)).pages)
    parts = []
    for start in range(1, pages + 1, PAGES_PER_BATCH):
        end = min(start + PAGES_PER_BATCH - 1, pages)
        parts.append(
            converter.convert(str(pdf_path), page_range=(start, end)).document.export_to_markdown()
        )
    return "\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", help="Comma-separated instrument_ids")
    ap.add_argument(
        "--reconvert", action="store_true", help="Re-run Docling even if the Markdown exists"
    )
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None
    pdf_dir, md_dir = SOURCES_DIR / "pdf", SOURCES_DIR / "md"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    manifest_path = SOURCES_DIR / "manifest.json"
    manifest = (
        {m["instrument_id"]: m for m in json.loads(manifest_path.read_text(encoding="utf-8"))}
        if manifest_path.exists()
        else {}
    )

    for src in SOURCES:
        iid = src["instrument_id"]
        if only and iid not in only:
            continue
        pdf_path, md_path = pdf_dir / f"{iid}.pdf", md_dir / f"{iid}.md"
        if not pdf_path.exists():
            pdf_path.write_bytes(download(src["url"]))
            time.sleep(0.3)
        if args.reconvert or not md_path.exists():
            t0 = time.time()
            md_path.write_text(convert_to_markdown(converter, pdf_path), encoding="utf-8")
            print(f"{iid}: converted in {time.time() - t0:.0f}s")
        data = pdf_path.read_bytes()
        manifest[iid] = {
            "law_position": "as issued",
            **src,
            "pdf": pdf_path.as_posix(),
            "markdown": md_path.as_posix(),
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }

    ordered = [manifest[s["instrument_id"]] for s in SOURCES if s["instrument_id"] in manifest]
    manifest_path.write_text(json.dumps(ordered, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(ordered)} sources -> {manifest_path}")


if __name__ == "__main__":
    main()
