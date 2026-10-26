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

import os
import argparse
import json
import re
import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")

# Configure Hugging Face Token for authenticated Hub requests if provided in environment/secrets
HF_TOKEN = os.environ.get("HF_TOKEN")

import pandas as pd
import pdfplumber

# sentence_transformers (and its transitive dep transformers) are imported
# lazily inside get_embedding_model() so that transformers/__init__.py —
# which calls importlib.util.find_spec("torchvision") at module load time —
# is never executed during app startup on Streamlit Cloud.
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        token = os.environ.get("HF_TOKEN", HF_TOKEN)
        if token:
            try:
                from huggingface_hub import login
                login(token=token, add_to_git_credential=False)
            except Exception:
                pass
        from sentence_transformers import SentenceTransformer  # lazy import
        if token:
            try:
                _embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", token=token)
            except Exception:
                _embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        else:
            _embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedding_model


_ocr_model = None

def get_ocr_model():
    """Lazy-load PaddleOCR engine to process invoice images on demand."""
    global _ocr_model
    if _ocr_model is None:
        try:
            from paddleocr import PaddleOCR
            _ocr_model = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        except ImportError as e:
            raise ImportError(
                "PaddleOCR / paddlepaddle is not installed or not available on Python 3.14. "
                "To process invoice images on Streamlit Cloud, set your app's Python version to 3.11 or 3.12 under App Settings > Advanced Settings."
            ) from e
    return _ocr_model

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


def extract_text_from_image(image_path: str) -> str:
    """Extract and reconstruct horizontal lines from an invoice image using PaddleOCR."""
    ocr = get_ocr_model()
    result = ocr.ocr(str(image_path), cls=True)
    if not result or not result[0]:
        return ""
    
    ocr_lines = result[0]
    items = []
    for line_info in ocr_lines:
        if not line_info or len(line_info) < 2:
            continue
        box, (text, conf) = line_info[0], line_info[1]
        text = str(text).strip()
        if not text:
            continue
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        y_center = sum(ys) / len(ys)
        x_min = min(xs)
        height = max(ys) - min(ys)
        items.append({"text": text, "y": y_center, "x": x_min, "h": height})
    
    if not items:
        return ""
    
    # Sort detected boxes top to bottom
    items.sort(key=lambda it: it["y"])
    
    lines = []
    current_line = []
    current_y = None
    
    for it in items:
        thresh = max(12.0, it["h"] * 0.45)
        if current_y is None or abs(it["y"] - current_y) <= thresh:
            current_line.append(it)
            current_y = sum(x["y"] for x in current_line) / len(current_line)
        else:
            current_line.sort(key=lambda x: x["x"])
            lines.append(" ".join(x["text"] for x in current_line))
            current_line = [it]
            current_y = it["y"]
            
    if current_line:
        current_line.sort(key=lambda x: x["x"])
        lines.append(" ".join(x["text"] for x in current_line))
        
    return "\n".join(lines)


def parse_invoice_tables(pdf_path: str) -> list[dict]:
    """Fallback: Extract line items from PDF tables (e.g. Purchase Orders)."""
    items = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables() or []
                for table in tables:
                    if not table or len(table) < 2:
                        continue
                    
                    desc_col, qty_col, rate_col, amount_col = -1, -1, -1, -1
                    header_row_idx = -1
                    
                    for r_idx, row in enumerate(table[:5]):
                        row_strs = [str(c or "").lower().strip() for c in row]
                        for c_idx, cell in enumerate(row_strs):
                            if any(k in cell for k in ["item", "product", "description", "particular", "name"]):
                                desc_col = c_idx
                            elif any(k in cell for k in ["qty", "quantity", "order qty"]):
                                qty_col = c_idx
                            elif any(k in cell for k in ["rate", "price", "mrp", "unit price", "cost"]):
                                rate_col = c_idx
                            elif any(k in cell for k in ["amount", "total", "net"]):
                                amount_col = c_idx
                        if desc_col != -1 and (qty_col != -1 or rate_col != -1):
                            header_row_idx = r_idx
                            break
                    
                    if header_row_idx == -1 or desc_col == -1:
                        continue
                        
                    for row in table[header_row_idx + 1:]:
                        if not row or desc_col >= len(row):
                            continue
                        desc = str(row[desc_col] or "").strip()
                        if not desc or desc.lower() in ["total", "sub total", "grand total", "subtotal"]:
                            continue
                            
                        qty = 1
                        if qty_col != -1 and qty_col < len(row):
                            q_text = re.sub(r"[^\d\.]", "", str(row[qty_col] or ""))
                            try:
                                qty = int(float(q_text)) if q_text else 1
                            except ValueError:
                                qty = 1
                                
                        rate = 0.0
                        if rate_col != -1 and rate_col < len(row):
                            r_text = re.sub(r"[^\d\.]", "", str(row[rate_col] or ""))
                            try:
                                rate = float(r_text) if r_text else 0.0
                            except ValueError:
                                rate = 0.0
                                
                        amount = rate * qty
                        if amount_col != -1 and amount_col < len(row):
                            a_text = re.sub(r"[^\d\.]", "", str(row[amount_col] or ""))
                            try:
                                amount = float(a_text) if a_text else amount
                            except ValueError:
                                pass
                                
                        if amount <= 0 and rate <= 0:
                            continue
                            
                        unit_net_price = round(amount / qty, 4) if qty else rate
                        items.append({
                            "sno": len(items) + 1,
                            "qty": qty,
                            "pack": "",
                            "description": desc,
                            "mrp": rate,
                            "rate": rate,
                            "discount_pct": 0.0,
                            "net_amount": amount,
                            "unit_net_price": unit_net_price,
                            "brand_tag": _extract_brand(desc),
                            "source_invoice": Path(pdf_path).name,
                        })
    except Exception:
        pass
    return items


def parse_invoice_text(full_text: str, source_name: str = "") -> list[dict]:
    """Extract line items from formatted text (from PDF or PaddleOCR)."""
    items = []
    lines = full_text.splitlines()
    for idx, raw_line in enumerate(lines):
        raw_line_clean = raw_line.strip()
        if not raw_line_clean:
            continue

        m = re.match(r"^\s*(\d+)[\.\)]?\s+(.*)$", raw_line_clean)
        if m:
            sno, rest = m.groups()
        else:
            sno = str(idx + 1)
            rest = raw_line_clean

        rest = _clean_watermark_noise(rest)
        rest_normalized = re.sub(r"(\d+),(\d{2})\b", r"\1.\2", rest)

        # decimals on the line: MRP, Rate, DIS%, CST/Batch%, NetAmount
        decimals = list(re.finditer(r"\d+\.\d{2}", rest_normalized))
        if len(decimals) < 2:
            continue  # Needs at least 2 price/amount decimals

        if len(decimals) >= 6:
            last_decimals = decimals[-6:]
            mrp, rate, dis_pct, batch_pct, _zero_col, net_amt = (float(x.group()) for x in last_decimals)
        elif len(decimals) == 5:
            last_decimals = decimals[-5:]
            mrp, rate, dis_pct, _, net_amt = (float(x.group()) for x in last_decimals)
        elif len(decimals) == 4:
            last_decimals = decimals[-4:]
            mrp, rate, dis_pct, net_amt = (float(x.group()) for x in last_decimals)
        elif len(decimals) == 3:
            last_decimals = decimals[-3:]
            mrp, rate, net_amt = (float(x.group()) for x in last_decimals)
            dis_pct = 0.0
        else:
            last_decimals = decimals[-2:]
            rate, net_amt = (float(x.group()) for x in last_decimals)
            mrp = rate
            dis_pct = 0.0

        head = rest_normalized[: last_decimals[0].start()].strip()
        if not head:
            continue

        # Extract qty: either at start ("10 ECHINACEA"), at end ("ECHINACEA 10"), or inferred from net/rate
        qty = 1
        qm_start = re.match(r"^(\d+)\s+(.*)$", head)
        qm_end = re.search(r"^(.*?)\s+(\d+)\s*$", head)

        if qm_start and int(qm_start.group(1)) > 0:
            qty = int(qm_start.group(1))
            head = qm_start.group(2)
        elif qm_end and int(qm_end.group(2)) > 0:
            qty = int(qm_end.group(2))
            head = qm_end.group(1)
        elif rate > 0 and net_amt > 0:
            calc_q = round(net_amt / rate)
            if 0 < calc_q < 10000:
                qty = int(calc_q)

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

        if len(description) < 2 or description.lower() in ["total", "sub total", "subtotal", "amount", "taxable"]:
            continue

        unit_net_price = round(net_amt / qty, 4) if qty else net_amt

        items.append(
            {
                "sno": int(sno) if sno.isdigit() else idx + 1,
                "qty": qty,
                "pack": pack,
                "description": description,
                "mrp": mrp,
                "rate": rate,
                "discount_pct": dis_pct,
                "net_amount": net_amt,
                "unit_net_price": unit_net_price,
                "brand_tag": _extract_brand(description),
                "source_invoice": source_name,
            }
        )
    return items


def parse_invoice(file_path: str) -> list[dict]:
    """Extract line items from a Swaran Homoeopathic (Marg ERP) style invoice (PDF or Image)."""
    path = Path(file_path)
    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

    full_text = ""
    if path.suffix.lower() in image_extensions:
        full_text = extract_text_from_image(str(path))
    elif path.suffix.lower() == ".pdf":
        try:
            with pdfplumber.open(str(path)) as pdf:
                full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception:
            full_text = ""

        # Fallback to PaddleOCR if the PDF contains scanned pages with no embedded text
        if not full_text.strip():
            try:
                full_text = extract_text_from_image(str(path))
            except Exception:
                pass
    else:
        full_text = extract_text_from_image(str(path))

    # 1. Parse text lines
    items = parse_invoice_text(full_text, source_name=path.name)
    
    # 2. If 0 items from text and it's a PDF, fallback to tabular parsing
    if not items and path.suffix.lower() == ".pdf":
        items = parse_invoice_tables(str(path))

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
    
    # Strip whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]

    # Map common column aliases flexibly
    col_map = {}
    for c in df.columns:
        c_norm = re.sub(r"[^a-zA-Z0-9]", "", c.lower())
        if c_norm in {"productname", "product", "itemname", "item", "description", "items"}:
            col_map[c] = "Product Name"
        elif c_norm in {"orderquantity", "orderqty", "quantity", "qty", "units", "orderedqty"}:
            col_map[c] = "Order Quantity"
        elif c_norm in {"transferprice", "price", "unitprice", "rate", "tp", "ourprice"}:
            col_map[c] = "Transfer Price"
        elif c_norm in {"transferpriceorderqty", "totalprice", "totalamount", "amount", "total"}:
            col_map[c] = "Transfer Price*Order QTY"

    df = df.rename(columns=col_map)

    # If Total Price column is missing but we have Qty and Price, compute it automatically
    if "Transfer Price*Order QTY" not in df.columns and "Transfer Price" in df.columns and "Order Quantity" in df.columns:
        df["Transfer Price*Order QTY"] = pd.to_numeric(df["Transfer Price"], errors="coerce") * pd.to_numeric(df["Order Quantity"], errors="coerce")

    required = ["Product Name", "Order Quantity", "Transfer Price", "Transfer Price*Order QTY"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Vendor sheet is missing expected column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )
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
    
    from sentence_transformers import util  # lazy import — avoids transformers load at startup
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



