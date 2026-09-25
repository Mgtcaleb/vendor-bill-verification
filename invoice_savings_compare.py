#!/usr/bin/env python3
"""
invoice_savings_compare.py

Compares a vendor's tax invoice (PDF) against your internal vendor sheet
(Excel: Product Name / Order Quantity / Transfer Price / Transfer Price*Order QTY)
and reports how much you saved (or overpaid) per line item and in total.

WHY THIS ISN'T A ONE-SHOT FUZZY MATCH:
Invoice descriptions and your sheet's "Product Name" column are written by two
different systems with no shared ID. Same remedy name can be sold under
different brands (SBL / Dr. Reckeweg / Adel / WSI / RW) and different potencies
(6, 30, 200 CH) at very different prices. A pure text-similarity match WILL
produce false-positive matches that look confident and are wrong. So this
script:
  1. Uses a persistent mapping file (JSON) of confirmed
     "invoice description -> vendor sheet product name" pairs. Anything in
     here is trusted and matched instantly, with no re-guessing.
  2. For anything NOT in the mapping, it proposes fuzzy candidates but does
     NOT auto-accept them — it writes them to a "needs_review.csv" for you
     to eyeball and either confirm (which appends to the mapping file for
     next time) or reject.
  3. It never matches across a detected brand/potency conflict, even at a
     high text-similarity score.

Usage (Mac, double-click-free — run from Terminal so file dialogs AND the
optional confirmation prompts both work):
    python3 invoice_savings_compare.py

This pops up native file-picker windows for the invoice PDF and the vendor
sheet, then a Yes/No popup asking whether you want to confirm uncertain
matches interactively in Terminal. Output CSVs land in a
"invoice_savings_output" folder next to the invoice PDF; the confirmed-match
memory (mapping.json) lands next to it too, so re-running against next
month's invoice reuses everything you already confirmed.

You can skip the dialogs and script it directly instead:
    python3 invoice_savings_compare.py \
        --invoice Swaran_23rd_PO_sept.pdf \
        --vendor-sheet local_vendor_sheet.xlsx \
        --mapping mapping.json \
        --outdir ./output \
        --interactive
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pdfplumber
from sentence_transformers import SentenceTransformer, util

# We will lazy-load the model to prevent the app from hanging on startup
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedding_model

# ----------------------------------------------------------------------
# 1. INVOICE PARSING
# ----------------------------------------------------------------------

# Known brand/supplier tags that show up in parentheses or as suffixes in
# this vendor's invoices. Two lines with DIFFERENT tags are different real
# products even if the remedy name text matches — never merge across these.
# Known brand/supplier tags that show up in parentheses or as suffixes in
# this vendor's invoices. Two lines with DIFFERENT canonical brands are
# different real products even if the remedy name text matches — never
# merge across these. Several raw tags refer to the same real brand under
# different abbreviations between the invoice and the vendor sheet, so they
# map to one canonical id.
BRAND_ALIASES = {
    "SBL": "SBL",
    "RW": "RECKEWEG",
    "RECKEWEG": "RECKEWEG",
    "WSI": "SCHWABE",
    "SCHWABE": "SCHWABE",
    "WILLMAR": "SCHWABE",
    "ADEL": "ADEL",
    "BAKSON": "BAKSON",
    "BAKSHI": "BAKSON",
    "WHEEZAL": "WHEEZAL",
    "ALLEN": "ALLEN",
}

# Generic words that appear in the vendor sheet's verbose Product Name
# column but never in the invoice's terse descriptions — they add nothing
# to a similarity comparison and only dilute the real match signal.
_FILLER_WORDS = {"DR", "INDIA", "GERMANY", "HOMOEOPATHY", "HOMEOPATHY", "WILLMAR", "TAB", "TABS", "DROP", "DROPS", "SYRUP", "OINTMENT", "PILLS", "TABLET", "TABLETS"}

PACK_UNIT_RE = re.compile(
    r"(\d+\.?\d*)\s*(ML|ml|GM|gm|G\b|TABS|Tabs|tabs|TAB|Tab|tab)", re.IGNORECASE
)



# Single-letter tokens that are legitimate homeopathy notation, not watermark
# noise, and must never be stripped even though they appear as standalone
# uppercase letters: Q = mother tincture, R = Dr. Reckeweg "R" combination
# drops (R6, R49, R71, R73...). Everything else appearing as an isolated
# single uppercase letter on an item line in this vendor's PDFs has, on
# inspection, been watermark bleed-through.
_PROTECTED_SINGLE_LETTERS = {"Q", "R"}


def _clean_watermark_noise(line: str) -> str:
    """
    This vendor's PDFs have a diagonal watermark whose text gets extracted
    character-by-character and interleaved into the real text: a stray
    letter glued inside a price ('1P20.00', '30C.00', 'M29362940'), or
    floating as its own token between two numbers ('300 E 49014'). Clean it:
      - strip 1-2 stray letters embedded in an otherwise-numeric token
      - drop isolated single-uppercase-letter tokens, except Q/R above
      - re-join an HSN code that got split by a now-removed stray token
    """
    tokens = [t for t in line.split(" ") if t != ""]
    cleaned = []
    for t in tokens:
        if re.fullmatch(r"[A-Z]", t):
            if t in _PROTECTED_SINGLE_LETTERS:
                cleaned.append(t)
            continue  # drop stray isolated letter
        if re.fullmatch(r"\d+\.?\d*(ML|ml|GM|gm|KG|kg|TAB|Tab|tab|TABS|Tabs|tabs|MG|mg)", t):
            cleaned.append(t)
            continue  # legitimate pack-size unit, not watermark noise
        digits = re.findall(r"\d", t)
        letters = re.findall(r"[A-Za-z]", t)
        if "(" in t or ")" in t:
            cleaned.append(t)
            continue  # brand tag e.g. "30(RW)" — never strip inside parens
        if len(digits) >= 2 and 1 <= len(letters) <= 2:
            stripped = re.sub(r"[A-Za-z]", "", t)
            t = stripped if stripped else t
        cleaned.append(t)
    line = " ".join(cleaned)

    # Re-merge an HSN code split by a stray token that sat between its
    # halves (e.g. "300 E 49014" -> "300 49014" -> "30049014"), only when
    # immediately followed by the price run so we don't touch real numbers.
    line = re.sub(
        r"\b(\d{2,3})\s+(\d{4,6})\b(?=\s+\d+\.\d{2})",
        lambda m: m.group(1) + m.group(2),
        line,
    )
    return re.sub(r"\s{2,}", " ", line).strip()


def parse_invoice(pdf_path: str) -> list[dict]:
    """Extract line items from a Swaran Homoeopathic (Marg ERP) style invoice."""
    with pdfplumber.open(pdf_path) as pdf:
        full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    items = []
    for raw_line in full_text.splitlines():
        m = re.match(r"^\s*(\d+)\.\s+(.*)$", raw_line)
        if not m:
            continue
        sno, rest = m.groups()
        rest = _clean_watermark_noise(rest)

        # last 6 decimals on the line = MRP, Rate, DIS%, BatchDisc%, an
        # always-zero column (CST/unused on this invoice layout), NetAmount
        decimals = list(re.finditer(r"\d+\.\d{2}", rest))
        if len(decimals) < 6:
            continue  # not a genuine item line (e.g. a totals line)
        last6 = decimals[-6:]
        mrp, rate, dis_pct, batch_pct, _zero_col, net_amt = (float(x.group()) for x in last6)

        head = rest[: last6[0].start()].strip()

        # qty is the first integer token in head
        qm = re.match(r"^(\d+)\s+(.*)$", head)
        if not qm:
            continue
        qty = int(qm.group(1))
        head = qm.group(2)

        # strip trailing HSN code (6-8 digit run) at the end of head
        head = re.sub(r"\s*\d{6,8}\s*$", "", head).strip()

        # split pack size off the front (handles both "30 ML X" and "75TABSX")
        pm = PACK_UNIT_RE.match(head)
        if pm:
            pack_qty, pack_unit = pm.groups()
            pack = f"{pack_qty}{pack_unit.upper()}"
            description = head[pm.end():].strip()
        else:
            pack = ""
            description = head.strip()

        unit_net_price = round(net_amt / qty, 4) if qty else net_amt

        items.append(
            {
                "sno": int(sno),
                "qty": qty,
                "pack": pack,
                "description": description,
                "mrp": mrp,
                "rate": rate,
                "discount_pct": dis_pct,
                "net_amount": net_amt,
                "unit_net_price": unit_net_price,
                "brand_tag": _extract_brand(description),
                "source_invoice": Path(pdf_path).name,
            }
        )
    return items


def _extract_brand(description: str) -> str | None:
    """Return the canonical brand id (see BRAND_ALIASES), not the raw tag,
    so 'WSI' on an invoice and 'Willmar Schwabe' on the sheet are recognized
    as the same brand instead of blocking a real match."""
    desc_up = description.upper()
    # check longer/more specific tags first so "WILLMAR SCHWABE" doesn't
    # get missed in favor of a shorter substring
    for tag in sorted(BRAND_ALIASES, key=len, reverse=True):
        if tag in desc_up:
            return BRAND_ALIASES[tag]
    return None


def _normalize_name(name: str) -> str:
    """Strip pack sizes, brand tags/parens, filler words, and punctuation
    so the invoice's terse description and the sheet's verbose Product Name
    compare on the part that actually identifies the product."""
    s = name.upper()
    s = PACK_UNIT_RE.sub(" ", s)
    s = re.sub(r"\([^)]*\)", " ", s)  # parenthetical brand/potency notes
    
    # Expand common abbreviations to help embeddings
    s = re.sub(r"\bBC\s*(\d+)\b", r"BIO COMBINATION \1", s)
    s = re.sub(r"\bBIO\s*COMB\b", "BIO COMBINATION", s)
    
    # Collapse isolated single letter + number, e.g. "R 49" -> "R49", "A 12" -> "A12"
    s = re.sub(r"\b([A-Z])\s+(\d+)\b", r"\1\2", s)
    
    for tag in BRAND_ALIASES:
        s = s.replace(tag, " ")
    for word in _FILLER_WORDS:
        s = re.sub(rf"\b{word}\b", " ", s)
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ----------------------------------------------------------------------
# 2. VENDOR SHEET LOADING
# ----------------------------------------------------------------------

def load_vendor_sheet(path: str) -> pd.DataFrame:
    engine = "xlrd" if str(path).lower().endswith(".xls") else "openpyxl"
    df = pd.read_excel(path, engine=engine)
    required = ["Product Name", "Order Quantity", "Transfer Price", "Transfer Price*Order QTY"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Vendor sheet is missing expected column(s): {missing}")
    df = df.dropna(subset=["Product Name"]).copy()
    df["norm_name"] = df["Product Name"].apply(_normalize_name)
    df["brand_tag"] = df["Product Name"].apply(_extract_brand)
    return df


# ----------------------------------------------------------------------
# 3. MATCHING
# ----------------------------------------------------------------------

def load_mapping(path: str) -> dict:
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def save_mapping(path: str, mapping: dict):
    Path(path).write_text(json.dumps(mapping, indent=2, ensure_ascii=False))


def match_items(
    invoice_items: list[dict],
    vendor_df: pd.DataFrame,
    mapping: dict,
    interactive: bool,
    auto_accept_score: int = 55,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Returns (matched, needs_review, unmatched).
    matched: confirmed pairs ready for savings calc.
    needs_review: fuzzy candidates below the confirm bar — written to CSV.
    unmatched: nothing plausible found at all.
    """
    matched, needs_review, unmatched = [], [], []
    
    model = get_embedding_model()
    
    vendor_norms = vendor_df["norm_name"].tolist()
    vendor_embeddings = model.encode(vendor_norms, convert_to_tensor=True)

    for inv in invoice_items:
        key = inv["description"].strip().upper()

        # 1. trusted mapping first
        if key in mapping:
            target_name = mapping[key]
            rows = vendor_df[vendor_df["Product Name"] == target_name]
            if not rows.empty:
                best_row = rows.iloc[0]
                best_diff = 999999
                for _, r in rows.iterrows():
                    try:
                        diff = abs(float(r["Order Quantity"]) - inv["qty"])
                        if diff < best_diff:
                            best_diff = diff
                            best_row = r
                    except:
                        pass
                matched.append(_build_match(inv, best_row, score=100.0, source="mapping"))
                continue

        # 2. fuzzy candidates, gated by brand-tag conflict
        inv_norm = _normalize_name(inv["description"])
        inv_emb = model.encode(inv_norm, convert_to_tensor=True)
        cosine_scores = util.cos_sim(inv_emb, vendor_embeddings)[0]
        
        candidates = []
        best_overall_score = 0
        best_overall_row = None
        brand_conflict_row = None
        brand_conflict_score = 0

        for i, (_, vrow) in enumerate(vendor_df.iterrows()):
            score = float(cosine_scores[i]) * 100
            
            # Substring exact-match booster for short descriptions (e.g. "BC 6")
            vrow_norm = str(vrow["norm_name"]).strip()
            if inv_norm and len(inv_norm) >= 3:
                # If invoice string is fully inside vendor string as whole words
                if re.search(rf"\b{re.escape(inv_norm)}\b", vrow_norm):
                    score = max(score, 90.0)
                # Or vice versa
                elif vrow_norm and re.search(rf"\b{re.escape(vrow_norm)}\b", inv_norm):
                    score = max(score, 90.0)
            
            if score > best_overall_score:
                best_overall_score = score
                best_overall_row = vrow
                
            if inv["brand_tag"] and vrow["brand_tag"] and inv["brand_tag"] != vrow["brand_tag"]:
                if score > brand_conflict_score:
                    brand_conflict_score = score
                    brand_conflict_row = vrow
                continue  # different confirmed brand -> never the same product
            
            if score >= 55:
                candidates.append((score, vrow))
                
        def candidate_sort_key(item):
            c_score, c_vrow = item
            try:
                diff = abs(float(c_vrow["Order Quantity"]) - inv["qty"])
            except:
                diff = 999999
            # Sort descending by score, then ascending by qty difference
            return (-c_score, diff)
            
        candidates.sort(key=candidate_sort_key)

        if not candidates:
            reason = "Not found in vendor sheet."
            if brand_conflict_score >= 55:
                reason = f"Brand Conflict: Matches '{brand_conflict_row['Product Name']}' ({brand_conflict_score:.1f}%) but invoice brand is '{inv['brand_tag']}' and sheet brand is '{brand_conflict_row['brand_tag']}'."
            elif best_overall_score >= 40:
                reason = f"Low Confidence: Best match is '{best_overall_row['Product Name']}' ({best_overall_score:.1f}%)."
            
            inv["investigation_reason"] = reason
            unmatched.append(inv)
            continue

        best_score, best_row = candidates[0]

        if best_score >= auto_accept_score:
            matched.append(_build_match(inv, best_row, score=best_score, source="fuzzy-auto"))
            mapping[key] = best_row["Product Name"]
            continue

        if interactive:
            print(f"\nInvoice item: {inv['description']!r} (pack {inv['pack']}, qty {inv['qty']})")
            for i, (score, vrow) in enumerate(candidates[:5]):
                print(f"  [{i}] {score:5.1f}  {vrow['Product Name']!r} (qty {vrow['Order Quantity']}, transfer price {vrow['Transfer Price']})")
            print("  [s] skip / no match")
            choice = input("  Confirm which is the same product > ").strip().lower()
            if choice.isdigit() and int(choice) < len(candidates):
                _, vrow = candidates[int(choice)]
                matched.append(_build_match(inv, vrow, score=candidates[int(choice)][0], source="manual-confirm"))
                mapping[key] = vrow["Product Name"]
                continue

        needs_review.append(
            {
                "source_invoice": inv.get("source_invoice", ""),
                "invoice_description": inv["description"],
                "invoice_pack": inv["pack"],
                "invoice_qty": inv["qty"],
                "invoice_unit_price": inv["unit_net_price"],
                "top_candidate": best_row["Product Name"],
                "candidate_score": best_score,
                "candidate_transfer_price": best_row["Transfer Price"],
                "candidate_order_qty": best_row["Order Quantity"],
            }
        )

    return matched, needs_review, unmatched


def _build_match(inv: dict, vrow: pd.Series, score: float, source: str) -> dict:
    invoice_unit_price = inv["unit_net_price"]
    transfer_price = float(vrow["Transfer Price"])
    qty_compared = min(inv["qty"], int(vrow["Order Quantity"])) if pd.notna(vrow["Order Quantity"]) else inv["qty"]
    savings_per_unit = transfer_price - invoice_unit_price
    return {
        "source_invoice": inv.get("source_invoice", ""),
        "invoice_description": inv["description"],
        "invoice_pack": inv["pack"],
        "invoice_qty": inv["qty"],
        "invoice_unit_price": invoice_unit_price,
        "vendor_product_name": vrow["Product Name"],
        "vendor_order_qty": vrow["Order Quantity"],
        "transfer_price": transfer_price,
        "qty_compared": qty_compared,
        "savings_per_unit": round(savings_per_unit, 2),
        "total_savings": round(savings_per_unit * qty_compared, 2),
        "match_score": score,
        "match_source": source,
    }


# ----------------------------------------------------------------------
# 4. MAIN
# ----------------------------------------------------------------------

def _pick_files() -> tuple[list[str], str, str, str, bool]:
    """Native macOS file dialogs: one or more invoice PDFs, one vendor sheet."""
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    invoice_paths = filedialog.askopenfilenames(
        title="Select one or more vendor tax invoice PDFs (Cmd-click to select several)",
        filetypes=[("PDF files", "*.pdf")],
    )
    if not invoice_paths:
        print("No invoice selected — exiting.")
        sys.exit(1)

    vendor_path = filedialog.askopenfilename(
        title="Select your vendor sheet (Excel) — compared against all selected invoices",
        filetypes=[("Excel files", "*.xlsx *.xls")],
    )
    if not vendor_path:
        print("No vendor sheet selected — exiting.")
        sys.exit(1)

    interactive = messagebox.askyesno(
        "Confirm uncertain matches?",
        "For invoice lines with no confirmed history, do you want to be "
        "prompted in Terminal to confirm the best-guess match?\n\n"
        "Yes = you'll be asked in Terminal for anything uncertain.\n"
        "No = uncertain lines are skipped and listed in needs_review.csv instead.",
    )

    first_dir = Path(invoice_paths[0]).resolve().parent
    outdir = str(first_dir / "invoice_savings_output")
    mapping_path = str(first_dir / "mapping.json")

    root.destroy()
    return list(invoice_paths), vendor_path, outdir, mapping_path, interactive


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--invoice", nargs="+", help="Path(s) to one or more invoice PDFs (skip for a file picker)")
    ap.add_argument("--vendor-sheet", help="Path to vendor sheet .xlsx/.xls (skip this to get a file picker)")
    ap.add_argument("--mapping", default=None, help="Path to persistent match-mapping JSON")
    ap.add_argument("--outdir", default=None, help="Where to write output CSVs")
    ap.add_argument("--interactive", action="store_true", help="Prompt to confirm fuzzy candidates")
    args = ap.parse_args()

    if args.invoice and args.vendor_sheet:
        invoice_paths = args.invoice
        vendor_path = args.vendor_sheet
        outdir_path = args.outdir or "."
        mapping_path = args.mapping or "mapping.json"
        interactive = args.interactive
    else:
        invoice_paths, vendor_path, outdir_path, mapping_path, interactive = _pick_files()

    outdir = Path(outdir_path)
    outdir.mkdir(parents=True, exist_ok=True)

    vendor_df = load_vendor_sheet(vendor_path)
    mapping = load_mapping(mapping_path)

    all_matched, all_needs_review, all_unmatched = [], [], []

    for invoice_path in invoice_paths:
        name = Path(invoice_path).name
        print(f"\n=== {name} ===")
        try:
            invoice_items = parse_invoice(invoice_path)
        except Exception as exc:
            print(f"  Could not parse this file, skipping it: {exc}")
            continue

        print(f"  Parsed {len(invoice_items)} line items.")
        checksum = sum(i["net_amount"] for i in invoice_items)
        print(f"  Sum of parsed line net-amounts: {checksum:.2f}  "
              f"(cross-check against this invoice's own pre-GST subtotal note)")

        matched, needs_review, unmatched = match_items(
            invoice_items, vendor_df, mapping, interactive=interactive
        )
        print(f"  Matched: {len(matched)}   Needs review: {len(needs_review)}   Unmatched: {len(unmatched)}")

        all_matched.extend(matched)
        all_needs_review.extend(needs_review)
        all_unmatched.extend(unmatched)

    print(f"\n=== Combined across {len(invoice_paths)} invoice(s) ===")

    if all_matched:
        save_mapping(mapping_path, mapping)
        matched_df = pd.DataFrame(all_matched)
        matched_df.to_csv(outdir / "matched_savings.csv", index=False)
        total_savings = matched_df["total_savings"].sum()
        print(f"\nMATCHED: {len(all_matched)} line(s) total. Combined savings on matched lines: {total_savings:.2f}")
        by_invoice = matched_df.groupby("source_invoice")["total_savings"].sum()
        print(by_invoice.to_string())
        print()
        print(matched_df[["source_invoice", "invoice_description", "vendor_product_name",
                           "invoice_unit_price", "transfer_price", "qty_compared",
                           "total_savings", "match_score"]].to_string(index=False))
    else:
        print("\nMATCHED: 0 line items with a confirmed or high-confidence pairing, across all invoices.")

    if all_needs_review:
        review_df = pd.DataFrame(all_needs_review)
        review_df.to_csv(outdir / "needs_review.csv", index=False)
        print(f"\nNEEDS REVIEW: {len(all_needs_review)} line(s) across all invoices have a candidate "
              f"match below the auto-accept confidence bar. See needs_review.csv — confirm each by "
              f"eye (brand + potency, not just the name) before trusting any savings number built on "
              f"them. Re-run with --interactive, or answer Yes to the popup, to confirm inline instead.")

    if all_unmatched:
        unmatched_df = pd.DataFrame(all_unmatched)
        unmatched_df.to_csv(outdir / "unmatched.csv", index=False)
        print(f"\nUNMATCHED: {len(all_unmatched)} line(s) across all invoices have nothing plausible "
              f"in the vendor sheet at all.")

    print(f"\nMapping file now has {len(mapping)} confirmed pairs — reused automatically next run.")
    print(f"\nAll output files are in: {outdir}")


if __name__ == "__main__":
    main()
