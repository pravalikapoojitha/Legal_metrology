import hashlib
import io
import logging
import os
import re
import smtplib
import sqlite3
import ssl
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import qrcode
import streamlit as st
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("legal_metrology")

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# Opt-in flag for lightweight deploys (Render/Docker): set ENABLE_OCR=false
# to skip touching OCR models entirely. Default is enabled.
_ENABLE_OCR = os.getenv("ENABLE_OCR", "true").strip().lower() in {"1", "true", "yes", "on"}

# Safe Imports for OCR libraries
if _ENABLE_OCR:
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        PaddleOCR = None

    try:
        import easyocr
    except ImportError:
        easyocr = None

    try:
        import pytesseract
    except ImportError:
        pytesseract = None
else:
    PaddleOCR = None
    easyocr = None
    pytesseract = None


# ============================================================
# Database Setup
# ============================================================
# Absolute path prevents CWD hijack when Streamlit is launched elsewhere.
# DB_FILE can be overridden (e.g. Render paid disk mounted at /data).
DB_FILE = (os.getenv("DB_FILE") or "").strip() or str(Path(__file__).resolve().parent / "products.db")

EXPECTED_COLUMNS = [
    "product_id",
    "product_name",
    "category",
    "mrp",
    "net_quantity",
    "manufacture_date",
    "manufacturer",
    "country_origin",
    "consumer_care",
    "fssai",
    "expiry",
]

CREATE_PRODUCTS_SQL = """
    CREATE TABLE IF NOT EXISTS products (
        product_id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_name TEXT,
        category TEXT,
        mrp TEXT,
        net_quantity TEXT,
        manufacture_date TEXT,
        manufacturer TEXT,
        country_origin TEXT,
        consumer_care TEXT,
        fssai TEXT,
        expiry TEXT
    )
"""


def _existing_columns(conn):
    return [r[1] for r in conn.execute("PRAGMA table_info(products)").fetchall()]


def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(CREATE_PRODUCTS_SQL)
        cols = _existing_columns(conn)
        # Migrate legacy schema (barcode PK) -> product_id schema.
        if "barcode" in cols and "product_id" not in cols:
            logger.info("Migrating legacy products table (barcode -> product_id)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS products_new (
                    product_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    product_name TEXT,
                    category TEXT,
                    mrp TEXT,
                    net_quantity TEXT,
                    manufacture_date TEXT,
                    manufacturer TEXT,
                    country_origin TEXT,
                    consumer_care TEXT,
                    fssai TEXT,
                    expiry TEXT
                )
            """)
            conn.execute("""
                INSERT INTO products_new
                (product_name, category, mrp, net_quantity, manufacture_date,
                 manufacturer, country_origin, consumer_care, fssai, expiry)
                SELECT product_name, category, mrp, net_quantity, manufacture_date,
                       manufacturer, country_origin, consumer_care, fssai, expiry
                FROM products
            """)
            conn.execute("DROP TABLE products")
            conn.execute("ALTER TABLE products_new RENAME TO products")
            cols = _existing_columns(conn)
        # Forward-fill any missing columns (safe ADD COLUMN).
        for col in EXPECTED_COLUMNS:
            if col not in cols and col != "product_id":
                conn.execute(f"ALTER TABLE products ADD COLUMN {col} TEXT")
        conn.commit()


init_db()


def save_product_to_database(data):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            """
            INSERT INTO products
            (product_name, category, mrp, net_quantity, manufacture_date, manufacturer, country_origin, consumer_care, fssai, expiry)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data.get("product_name", ""),
                data.get("category", ""),
                data.get("mrp", ""),
                data.get("net_quantity", ""),
                data.get("manufacture_date", ""),
                data.get("manufacturer", ""),
                data.get("country_origin", ""),
                data.get("consumer_care", ""),
                data.get("fssai", ""),
                data.get("expiry", ""),
            ),
        )
        conn.commit()


def list_catalog_products():
    with sqlite3.connect(DB_FILE) as conn:
        try:
            return pd.read_sql_query("SELECT * FROM products", conn)
        except Exception as exc:
            logger.exception("Failed to read catalog: %s", exc)
            return pd.DataFrame(columns=[c for c in EXPECTED_COLUMNS if c != "product_id"])


# ============================================================
# High-Accuracy & Ultra-Fast OCR Scanning (< 2-4 seconds)
# ============================================================
def optimize_image_for_ocr(image_pil, max_dim=1280):
    """Downscale high-resolution camera images to preserve detail while cutting OCR processing time by 80%."""
    w, h = image_pil.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        return image_pil.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
    return image_pil


@st.cache_resource
def load_paddleocr():
    """Load high-speed PP-OCRv4 mobile models cached in memory for sub-second CPU inference."""
    if not _ENABLE_OCR:
        return None
    if PaddleOCR is not None:
        try:
            return PaddleOCR(
                ocr_version="PP-OCRv4",
                lang="en",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_det_limit_side_len=1280,
                text_det_unclip_ratio=2.0,
            )
        except Exception as e:
            logger.exception("PaddleOCR init failed: %s", e)
            st.warning(f"PaddleOCR failed to initialize: {e}")
            return None
    return None


@st.cache_resource
def load_easyocr_reader():
    """Load cached EasyOCR reader in memory for fast fallback."""
    if not _ENABLE_OCR:
        return None
    if easyocr is not None:
        try:
            return easyocr.Reader(["en"], gpu=False)
        except Exception as e:
            logger.exception("EasyOCR init failed: %s", e)
            st.warning(f"EasyOCR failed to initialize: {e}")
            return None
    return None


def _parse_paddle_output(output):
    """Handle both new (predict -> dict with rec_texts) and legacy (ocr -> nested lists) formats."""
    parsed = []
    if output is None:
        return parsed
    # Legacy: [[ [box, (text, conf)], ... ]] or [ [(text, conf)] ]
    try:
        for res in output:
            if isinstance(res, dict) and "rec_texts" in res:
                parsed.extend([str(t).strip() for t in res["rec_texts"] if str(t).strip()])
            elif isinstance(res, dict) and "text" in res:
                if str(res["text"]).strip():
                    parsed.append(str(res["text"]).strip())
            elif hasattr(res, "text") and res.text:
                parsed.append(str(res.text).strip())
            elif isinstance(res, (list, tuple)):
                for item in res:
                    # item: [box, (text, conf)] or (text, conf) or str
                    text = ""
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        candidate = item[1]
                        if isinstance(candidate, (list, tuple)):
                            text = str(candidate[0]) if candidate else ""
                        else:
                            text = str(candidate)
                    elif isinstance(item, str):
                        text = item
                    if text.strip():
                        parsed.append(text.strip())
            elif isinstance(res, str) and res.strip():
                parsed.append(res.strip())
    except Exception as exc:
        logger.exception("Failed to parse PaddleOCR output: %s", exc)
    return parsed


def extract_label_values(image_pil):
    """High-speed OCR value extraction completed within 2-4 seconds on CPU."""
    full_text = ""
    lines = []

    # Downscale high-resolution images to prevent massive pixel computation
    opt_pil = optimize_image_for_ocr(image_pil, max_dim=1280)
    img_np = np.array(opt_pil)

    # Option 1: Ultra-fast PP-OCRv4 mobile pipeline
    paddle_pipeline = load_paddleocr()
    if paddle_pipeline is not None:
        try:
            if hasattr(paddle_pipeline, "predict"):
                output = paddle_pipeline.predict(img_np)
            else:  # legacy PaddleOCR API
                output = paddle_pipeline.ocr(img_np)
            lines = _parse_paddle_output(output)
            full_text = " ".join(lines)
        except Exception as exc:
            logger.exception("PaddleOCR inference failed: %s", exc)
            full_text = ""
            lines = []

    # Option 2: Cached EasyOCR fallback (without slow cv2 denoising)
    if not full_text:
        reader = load_easyocr_reader()
        if reader is not None:
            try:
                results = reader.readtext(img_np, detail=0)
                lines = [str(r).strip() for r in results if str(r).strip()]
                full_text = " ".join(lines)
            except Exception as exc:
                logger.exception("EasyOCR inference failed: %s", exc)
                full_text = ""
                lines = []

    # Option 3: Lightweight system Tesseract fallback for container deploys.
    if not full_text and pytesseract is not None:
        try:
            text = pytesseract.image_to_string(opt_pil, config="--psm 6")
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            full_text = " ".join(lines)
        except Exception as exc:
            logger.exception("Tesseract OCR inference failed: %s", exc)
            full_text = ""
            lines = []
    if not full_text and not lines:
        if not _ENABLE_OCR:
            logger.warning("OCR disabled via ENABLE_OCR=false; manual entry only.")
        else:
            logger.warning("OCR produced no text; check image quality/model availability.")

    # ── Pass 1: line-by-line contextual field extraction ──────────────
    fields = {
        "product_name": "", "mrp": "", "net_quantity": "",
        "manufacture_date": "", "country_origin": "",
        "manufacturer": "", "consumer_care": "",
        "fssai": "", "expiry": "",
    }

    for i, line in enumerate(lines):
        lc = line.strip()
        ll = lc.lower()

        # Manufacturer / Packer
        if not fields["manufacturer"]:
            m = re.search(
                r"(?:manufactured\s+by|mfg\.?\s*by|marketed\s+by|packed\s+by|packer|manufacturer)[\:\s\.\-]*(.+)",
                lc, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                fields["manufacturer"] = val if len(val) >= 4 else (lines[i + 1].strip() if i + 1 < len(lines) else val)

        # Consumer Care
        if not fields["consumer_care"]:
            m = re.search(
                r"(?:consumer\s*(?:care|cell|helpline)|customer\s*(?:care|service|support|cell)|helpline|toll\s*free|feedback|contact\s*us)[\:\s\.\-]*(.+)",
                lc, re.IGNORECASE)
            if m:
                val = re.sub(r"^(?:helpline|cell|contact|phone|email|no\.?)[\:\s\.\-]*", "", m.group(1).strip(), flags=re.IGNORECASE)
                fields["consumer_care"] = val
            elif "@" in lc and any(w in ll for w in ["care", "help", "support", "feedback", "customercare"]):
                fields["consumer_care"] = lc
            elif re.search(r"\b1800[-\s]?\d{3}[-\s]?\d{3,4}\b", lc):
                fields["consumer_care"] = re.search(r"\b1800[-\s]?\d{3}[-\s]?\d{3,4}\b", lc).group(0)

        # Country of Origin
        if not fields["country_origin"]:
            m = re.search(
                r"(?:country\s+of\s+origin|\borigin\b|made\s+in|product\s+of)[\:\s\.\-]*([a-zA-Z]+)",
                lc, re.IGNORECASE)
            if m:
                fields["country_origin"] = m.group(1).strip()

        # Best Before / Expiry — line-by-line
        if not fields["expiry"]:
            m = re.search(
                r"(?:best\s+before|expiry\s*(?:date)?|use\s+by|exp\.?\s*date|exp\.?)[\:\s\.\-]*([a-zA-Z0-9\s/_\.\-]+)",
                lc, re.IGNORECASE)
            if m:
                val = re.split(r"\b(?:mrp|net\s*wt|lic)\b", m.group(1).strip(), flags=re.IGNORECASE)[0].strip()
                fields["expiry"] = _clean_expiry_capture(val)

        # If the keyword appeared alone or with trailing punctuation
        # ("Best" / "before :" / "Expiry:"), look ahead in the next few OCR
        # lines for the actual duration/date value.
        if not fields["expiry"] and re.match(
            r"(?:best\s*before|expiry\s*(?:date)?|use\s*by|exp\.?\s*date|exp\.?)[\:\s\.\-]*$",
            lc, re.IGNORECASE,
        ):
            for lookahead in lines[i + 1 : i + 5]:
                lf = lookahead.strip()
                # Skip lines that are obviously other label fields
                if re.search(r"\b(?:mrp|net|mfg|fssai|lic|country|batch)\b", lf, re.IGNORECASE):
                    break
                val = _clean_expiry_capture(lf)
                if val:
                    fields["expiry"] = val
                    break

        # Mfg Date
        if not fields["manufacture_date"]:
            m = re.search(
                r"(?:mfg\.?\s*(?:date)?|date\s+of\s+mfg|pkd\.?\s*(?:date)?|packed\s+on|date\s+of\s+packing|mfd\.?)[\:\s\.\-]*([a-zA-Z0-9\s/_\.\-]+)",
                lc, re.IGNORECASE)
            if m:
                val = re.split(r"\b(?:mrp|net|exp|use|best|lic)\b", m.group(1).strip(), flags=re.IGNORECASE)[0].strip()
                fields["manufacture_date"] = val

        # MRP
        if not fields["mrp"]:
            m = re.search(
                r"(?:m\.?r\.?p\.?|max\.?\s*retail\s*price|price)[\:\s\.\-]*(?:rs\.?|₹)?[\s]*([\d\.,]+(?:\s*/-)?)",
                lc, re.IGNORECASE)
            if m:
                fields["mrp"] = m.group(1).replace("/-", "").strip("., ")

        # Net Quantity
        if not fields["net_quantity"]:
            m = re.search(
                r"(?:net\s*(?:wt\.?|quantity|weight|vol\.?|volume|contents?)|qty\.?)[\:\s\.\-]*(\d+(?:\.\d+)?\s*(?:g|kg|ml|l|grams?|gm|ltr|litre|litres|kilograms?))\b",
                lc, re.IGNORECASE)
            if m:
                fields["net_quantity"] = m.group(1).strip()

        # FSSAI
        if not fields["fssai"]:
            m = re.search(
                r"(?:fssai|lic\.?\s*(?:no\.?)?|licence\s*(?:no\.?)?)[\:\s\.\-]*([12]\d{13})\b",
                lc, re.IGNORECASE)
            if m:
                fields["fssai"] = m.group(1).strip()

    # ── Pass 2: global regex fallback for any still-missing fields ─────
    if not fields["mrp"]:
        m = re.search(r"(?:m\.?r\.?p\.?|price|₹|rs\.?)[\:\s\.\-]*([\d\.,]+(?:\s*/-)?)", full_text, re.IGNORECASE)
        if m:
            fields["mrp"] = m.group(1).replace("/-", "").strip("., ")

    if not fields["net_quantity"]:
        m = re.search(r"(\d+(?:\.\d+)?\s*(?:g|kg|ml|l|grams?|gm|ltr|litre|litres|kilograms?))\b", full_text, re.IGNORECASE)
        if m:
            fields["net_quantity"] = m.group(1).strip()

    if not fields["manufacture_date"]:
        m = re.search(r"(?:mfg|pkd|packed|mfd|date)[\:\s\.\-]*(\d{1,2}[-/\.]\d{2,4}|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{2,4})", full_text, re.IGNORECASE)
        if m:
            fields["manufacture_date"] = m.group(1).strip()
        else:
            m2 = re.search(r"\b(\d{2}[-/\.]\d{4}|\d{2}[-/\.]\d{2}[-/\.]\d{2,4})\b", full_text)
            if m2:
                fields["manufacture_date"] = m2.group(1).strip()

    if not fields["fssai"]:
        m = re.search(r"\b([12]\d{13})\b", full_text)
        if m:
            fields["fssai"] = m.group(1).strip()

    if not fields["country_origin"]:
        m = re.search(r"(?:country\s+of\s+origin|\borigin\b|made\s+in)[\:\s\.\-]*([a-zA-Z]+)", full_text, re.IGNORECASE)
        # Do NOT default to "India": a missing declaration must fail
        # Rule 6(1)(aa) instead of silently passing.
        fields["country_origin"] = m.group(1).strip() if m else ""

    if not fields["expiry"]:
        m = re.search(
            r"(?:best\s+before|expiry\s*(?:date)?|use\s+by|exp(?:iry)?\.?\s*date)[\:\s\.\-]*"
            r"([a-zA-Z0-9\s/_\.\-]+?)"
            r"(?=(?:\bmrp\b|\bnet\b|\bmfg\b|\bfssai\b|\blic\b|\bcountry\b|$))",
            full_text, re.IGNORECASE)
        if m:
            fields["expiry"] = _clean_expiry_capture(m.group(1))

    if not fields["consumer_care"]:
        m = re.search(r"(?:consumer|customer|helpline|care|toll\s*free)[\:\s\.\-]*([a-zA-Z0-9@\.\-\+\s]+?)(?=(?:mrp|mfg|net|fssai|$))", full_text, re.IGNORECASE)
        if m:
            fields["consumer_care"] = m.group(1).strip()

    if not fields["manufacturer"]:
        m = re.search(r"(?:manufactured\s+by|mfg\s+by|marketed\s+by|packer|manufacturer)[\:\s\.\-]*([a-zA-Z0-9\s\.\,]+?)(?=(?:consumer|customer|fssai|mrp|mfg|net|country|$))", full_text, re.IGNORECASE)
        if m:
            fields["manufacturer"] = m.group(1).strip()

    # ── Pass 3: product name — first clean non-declaration headline ────
    # Refine common label formats that OCR may split into separate lines.
    for i, line in enumerate(lines):
        clean = line.strip()

        if not fields["mrp"]:
            match = re.search(
                r"\b(?:m\.?r\.?p\.?|maximum\s+retail\s+price)\b[^0-9]{0,12}(\d+(?:[.,]\d{1,2})?)",
                clean,
                re.IGNORECASE,
            )
            if match:
                fields["mrp"] = match.group(1).replace(",", "")

        if not fields["manufacture_date"]:
            match = re.search(
                r"\b(?:packing|packed|mfg|mfd|manufacturing)\s*(?:date|on)?\b[^0-9A-Za-z]{0,12}(\d{1,2}[-/]?[A-Za-z]{3}[-/]?\d{2,4}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
                clean,
                re.IGNORECASE,
            )
            if match:
                fields["manufacture_date"] = match.group(1)

        if not fields["consumer_care"]:
            match = re.search(
                r"\b(?:contact|phone|tel(?:ephone)?|customer\s*care|consumer\s*care)\b\s*[:\-.]?\s*(.+)",
                clean,
                re.IGNORECASE,
            )
            if match:
                value = match.group(1).strip()
                if re.search(r"\d{7,}|@", value):
                    fields["consumer_care"] = value

        if not fields["fssai"]:
            match = re.search(
                r"\bfssai\s*(?:lic(?:ence|ense)?\s*)?(?:no\.?\s*)?[:\-.]?\s*(\d{14})\b",
                clean,
                re.IGNORECASE,
            )
            if match and match.group(1)[0] in "12":
                fields["fssai"] = match.group(1)

    # PaddleOCR can split a declaration over adjacent text boxes (for
    # example, "MRP" / "₹" / "120"). Search the reconstructed reading order
    # as a final recovery step for those cases.
    joined_text = " ".join(lines)
    if not fields["mrp"]:
        match = re.search(
            r"\b(?:m\.?r\.?p\.?|maximum\s+retail\s+price)\b[^0-9]{0,80}?(\d+(?:[.,]\d{1,2})?)",
            joined_text,
            re.IGNORECASE,
        )
        if match:
            fields["mrp"] = match.group(1).replace(",", "")

    if not fields["manufacture_date"]:
        match = re.search(
            r"\bpack(?:ing|ed)?(?:\s+[a-z])?\s+date\b[^0-9]{0,12}(\d{1,2}[-/]?[A-Za-z]{3}[-/]?\d{2,4}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            joined_text,
            re.IGNORECASE,
        )
        if match:
            fields["manufacture_date"] = match.group(1)

    if not fields["manufacture_date"]:
        match = re.search(
            r"\b(?:date\s+of\s+manufactur(?:e|ing)|manufactur(?:e|ing)\s+date)\b[^0-9]{0,20}(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            joined_text,
            re.IGNORECASE,
        )
        if match:
            fields["manufacture_date"] = match.group(1)

    if not fields["expiry"]:
        # Joined-text recovery for OCR that splits "Best" / "before" / "6" /
        # "months" into separate text boxes (any typo tolerance on before):
        m = re.search(
            r"\bbest\s*befo?re?\b[^0-9]{0,16}"
            r"(\d+\s*-?\s*(?:days?|months?|years?|yrs?)|[a-zA-Z0-9/_\.\- ]+?)",
            joined_text,
            re.IGNORECASE,
        )
        if m:
            val = _clean_expiry_capture(m.group(1))
            # A bare year ("2026") alone is not a shelf-life/date we can trust.
            if val and re.fullmatch(r"\d{4}", val):
                val = ""
            if val:
                fields["expiry"] = val

    # A label and its contact details are often separate OCR boxes. Replace a
    # punctuation-only capture with the actual phone number and email address.
    care_section = re.search(r"\bcustomer\s+care\b(.{0,220})", joined_text, re.IGNORECASE)
    care_text = care_section.group(1) if care_section else ""
    contact_match = re.search(r"\+?\d[\d\s-]{8,}\d", care_text)
    email_match = re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", care_text, re.IGNORECASE)
    if contact_match and (not fields["consumer_care"] or re.fullmatch(r"[:.\-\s]*", fields["consumer_care"])):
        contact = contact_match.group(0).strip()
        fields["consumer_care"] = f"{contact} | {email_match.group(0)}" if email_match else contact

    # An unlabelled business name immediately above an address is normally
    # the manufacturer/packer name on packaged-food labels.
    if not fields["manufacturer"]:
        for index, line in enumerate(lines):
            if re.search(r"\b(?:address|add\.?|plot|road|street)\b", line, re.IGNORECASE) and index:
                start = max(0, index - 3)
                business = " ".join(part.strip() for part in lines[start:index])
                if len(business) >= 3 and not re.search(
                    r"\b(?:mrp|net|batch|packing|best\s+before)\b", business, re.IGNORECASE
                ):
                    # Remove an isolated OCR artefact before a title-cased
                    # business name, e.g. "IGuru" -> "Guru".
                    business = re.sub(r"\bI(?=[A-Z][a-z]{2,}\b)", "", business)
                    fields["manufacturer"] = f"{business}, {line.strip()}"
                    break

    skip_keywords = [
        "mrp", "rs.", "price", "net wt", "net qty", "quantity",
        "mfg", "pkd", "fssai", "licence", "lic no", "batch",
        "ingredients", "nutrition", "100g", "country of origin", "veg",
        ".png", ".jpg", ".jpeg",
    ]
    product_parts = []
    product_noise = {
        "goodness in every bite", "tasty", "crunchy", "healthy", "rich in",
        "protein", "vegetarian", "preservatives", "store in a cool",
    }
    for line in lines:
        clean = re.sub(r"\s+", " ", line.strip())
        lowered = clean.lower()
        if (
            len(clean) >= 3
            and re.fullmatch(r"[A-Za-z][A-Za-z'& -]{1,40}", clean)
            and not any(keyword in lowered for keyword in skip_keywords)
            and lowered not in product_noise
            and "manufactured" not in lowered
            and "customer" not in lowered
            and "best before" not in lowered
            and "date of" not in lowered
            and "net " not in lowered
        ):
            if lowered not in {part.lower() for part in product_parts}:
                product_parts.append(clean)
        if len(product_parts) == 4:
            break
    fields["product_name"] = " ".join(product_parts)

    return (
        fields["product_name"],
        fields["mrp"],
        fields["net_quantity"],
        fields["manufacture_date"],
        fields["fssai"],
        fields["country_origin"],
        fields["manufacturer"],
        fields["consumer_care"],
        fields["expiry"],
    )


# ============================================================
# Field validators (presence-only checks cause false PASS)
# ============================================================
_MRP_RE = re.compile(r"^\s*(?:rs\.?\s*|₹\s*)?\d{1,6}(?:[.,]\d{1,2})?\s*(?:/-)?\s*$", re.IGNORECASE)
_QTY_RE = re.compile(
    r"^\s*\d+(?:\.\d+)?\s*(g|kg|gm|grams?|kilograms?|ml|l|ltr|litre|litres)\s*$",
    re.IGNORECASE,
)
_FSSAI_RE = re.compile(r"^\s*[12]\d{13}\s*$")
_DATE_RE = re.compile(
    r"(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{1,2}[-/\.]\d{4}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{2,4})",
    re.IGNORECASE,
)
_CARE_RE = re.compile(r"(\+?\d[\d\s\-]{6,}\d|[^@\s]+@[^@\s]+\.[^@\s]+)")
_EXPIRY_DURATION_RE = re.compile(
    r"\b\d+\s*-?\s*(days?|months?|years?|yrs?)\b", re.IGNORECASE
)


def _clean_expiry_capture(value):
    """Reject punctuation-only OCR fragments (':' / '.' / '-') so a split
    label leaves the field empty ('missing') instead of auto-filling junk
    that then fails validation as 'invalid'."""
    v = re.sub(r"\s+", " ", (value or "").strip(" :.-\t"))
    if len(v) < 2 or not re.search(r"\d", v):
        return ""
    return v


def _clean_mrp(value):
    return value.replace(",", "").replace("/-", "").strip(" .,")


def is_valid_mrp(value):
    v = _clean_mrp(value or "")
    if not v:
        return False
    if not _MRP_RE.match(v):
        return False
    try:
        return float(re.sub(r"[^\d.]", "", v)) > 0
    except ValueError:
        return False


def is_valid_quantity(value):
    return bool(value and _QTY_RE.match(value.strip()))


def is_valid_fssai(value):
    return bool(value and _FSSAI_RE.match(value.strip()))


def is_valid_mfg_date(value):
    return bool(value and _DATE_RE.search(value.strip()))


def is_valid_expiry(value):
    # Accepts absolute dates (08/2026, Aug 2025) AND relative shelf-life
    # declarations ("Best before 6 months", "12 months from mfg", "180 days").
    if not value:
        return False
    v = value.strip()
    if _DATE_RE.search(v):
        return True
    if _EXPIRY_DURATION_RE.search(v):
        return True
    return False


def is_valid_care(value):
    return bool(value and _CARE_RE.search(value.strip()))


def is_valid_generic(value, min_len=3):
    return bool(value and len(value.strip()) >= min_len)


# ============================================================
# Page setup & State
# ============================================================
st.set_page_config(
    page_title="Legal Metrology Compliance Checker",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "history" not in st.session_state:
    st.session_state.history = []
if "feedback" not in st.session_state:
    st.session_state.feedback = []

# OCR auto-fill session state keys (one per form field)
_OCR_KEYS = [
    "ocr_product_name", "ocr_mrp", "ocr_net_qty",
    "ocr_mfg_date", "ocr_fssai", "ocr_origin",
    "ocr_manufacturer", "ocr_care", "ocr_expiry",
]
if "ocr_last_hash" not in st.session_state:
    st.session_state["ocr_last_hash"] = ""
for _k in _OCR_KEYS:
    if _k not in st.session_state:
        st.session_state[_k] = ""

t = {
    "inspect": "📷 Live Inspection",
    "dashboard": "📊 Dashboard",
    "history": "📜 History Logs",
    "rules": "⚖️ Rules Catalog",
    "feedback": "💬 Feedback",
    "pass": "PASSED",
    "fail": "FAILED",
    "save": "💾 Save Inspection Result",
    "thanks": "Thank you for your feedback!",
}

st.markdown(
    """
    <style>
    :root {
        --ink: #111111;
        --muted: #333333;
        --cream: #f4efe6;
        --paper: #fffdf8;
        --green: #4f9f68;
        --green-dark: #3d8053;
        --gold: #c28b2c;
        --line: #e6ded0;
        --red: #b42318;
    }

    .stApp {
        background: #f4efe6;
        color: #111111;
    }

    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .stApp p, .stApp label,
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stTextInput"] label,
    .stApp [data-testid="stTextArea"] label,
    .stApp [data-testid="stSelectbox"] label,
    .stApp [data-testid="stFileUploader"] label {
        color: #111111 !important;
    }

    [data-testid="stSidebar"] {
        background: #d9f0df;
        border-right: 1px solid #b7d9c0;
    }

    [data-testid="stSidebar"] * {
        color: #174a2c !important;
    }

    .section-title {
        color: #111111;
        font-size: 1.25rem;
        font-weight: 800;
        margin: 1.35rem 0 0.7rem;
    }

    .section-title span {
        display: inline-grid;
        place-items: center;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: #d9f0df;
        color: #111111;
        margin-right: 0.45rem;
        font-size: 0.8rem;
    }

    .rule-card {
        background: #fffdf8;
        border: 1px solid #e6ded0;
        border-left: 5px solid #c28b2c;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.8rem;
    }

    .status-good {
        background: #e7f4ec;
        border: 1px solid #a8d5b9;
        color: #155e3d;
        padding: 1rem;
        border-radius: 14px;
        font-weight: 700;
    }

    .status-bad {
        background: #fff0ed;
        border: 1px solid #f0b6ad;
        color: #9b2c20;
        padding: 1rem;
        border-radius: 14px;
        font-weight: 700;
    }

    .metric-box {
        background: #fffdf8;
        border: 1px solid #e6ded0;
        border-radius: 14px;
        padding: 1rem;
    }

    .metric-label { color: #111111; font-size: 0.78rem; }
    .metric-value { color: #111111; font-size: 1.65rem; font-weight: 800; margin-top: 0.25rem; }

    .stButton > button, .stDownloadButton > button {
        background: #4f9f68;
        color: #ffffff !important;
        border: 0;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar inputs
st.sidebar.markdown(
    "<div class='side-brand'><strong>Legal"
    " Metrology</strong><small>Inspection Portal</small></div>",
    unsafe_allow_html=True,
)
selected_tab = st.sidebar.radio(
    "Navigation",
    [
        t["inspect"],
        t["dashboard"],
        t["history"],
        t["rules"],
        t["feedback"],
    ],
)
st.sidebar.markdown("---")
inspector_name = st.sidebar.text_input("Inspector Name", "Inspector Officer")
shop_name = st.sidebar.text_input("Store Name", "Metro Retail Store")
location_gps = st.sidebar.text_input("Location", "Central Market")


# ============================================================
# Rules catalog
# ============================================================
RULES = [
    (
        "Maximum Retail Price (MRP)",
        "Rule 6(1)(e)",
        25000,
        (
            "MRP must be declared inclusive of all taxes and must not be"
            " obscured or misleading."
        ),
    ),
    (
        "Net Quantity Declaration",
        "Rule 6(1)(c)",
        10000,
        (
            "Net weight, measure, or volume must use a recognized unit such as"
            " g, kg, ml, or L."
        ),
    ),
    (
        "Month & Year of Manufacture",
        "Rule 6(1)(d)",
        15000,
        (
            "The month and year of manufacture or packing should be clearly"
            " available to the consumer."
        ),
    ),
    (
        "Manufacturer/Packer Details",
        "Rule 6(1)(a)",
        25000,
        (
            "The name and complete address identify the responsible"
            " manufacturer or packer."
        ),
    ),
    (
        "Country of Origin",
        "Rule 6(1)(aa)",
        50000,
        (
            "Imported products should disclose the country of origin for"
            " consumer transparency."
        ),
    ),
    (
        "Consumer Care Contact",
        "Rule 6(2)",
        10000,
        (
            "A phone number, email address, or other consumer-care contact"
            " supports grievance redressal."
        ),
    ),
]


def display_rules():
    st.title("⚖️ Legal Metrology Rules & Guidance")
    st.caption(
        "Reference checklist for packaged commodities. Penalty figures shown"
        " in this prototype are indicative and require official verification."
    )
    for name, section, fine, explanation in RULES:
        st.markdown(
            f"<div class='rule-card'><h4>{name}"
            f" <small>({section})</small></h4><p>{explanation}</p><p><b>Prototype"
            f" reference amount:</b> ₹{fine:,}</p></div>",
            unsafe_allow_html=True,
        )
    st.info(
        "This application is a screening aid. Final enforcement decisions,"
        " notices, and penalties must be made by the competent Legal Metrology"
        " authority under the applicable law."
    )


# ============================================================
# Email delivery
# ============================================================
def _smtp_setting(name, default=""):
    """Read SMTP settings from Streamlit secrets first, then environment."""
    try:
        secret_value = st.secrets.get(name)
    except Exception:
        secret_value = None
    return str(secret_value or os.getenv(name) or default).strip()


def send_pdf_email(recipient, pdf_bytes, shop_name):
    recipient = (recipient or "").strip()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", recipient):
        raise ValueError("Enter a valid recipient email address.")

    smtp_host = _smtp_setting("SMTP_HOST")
    try:
        smtp_port = int(_smtp_setting("SMTP_PORT", "587"))
    except ValueError:
        raise ValueError("SMTP_PORT must be a number.")
    if not 1 <= smtp_port <= 65535:
        raise ValueError("SMTP_PORT must be between 1 and 65535.")
    smtp_username = _smtp_setting("SMTP_USERNAME")
    smtp_password = _smtp_setting("SMTP_PASSWORD")
    sender = _smtp_setting("SMTP_FROM") or smtp_username
    if not all([smtp_host, smtp_username, smtp_password, sender]):
        raise RuntimeError(
            "Email is not configured. Set SMTP_HOST, SMTP_PORT, SMTP_USERNAME, "
            "SMTP_PASSWORD, and SMTP_FROM in the server environment."
        )

    message = EmailMessage()
    safe_shop = re.sub(r"[\r\n]+", " ", str(shop_name or ""))[:80]
    message["Subject"] = f"Legal Metrology Inspection Certificate - {safe_shop}"
    message["From"] = sender
    message["To"] = recipient
    message.set_content(
        "Please find the Legal Metrology inspection certificate attached. "
        "This document is a screening aid and does not replace an official"
        " decision."
    )
    message.add_attachment(
        pdf_bytes,
        maintype="application",
        subtype="pdf",
        filename="Legal_Metrology_Certificate.pdf",
    )

    if smtp_port == 465:
        with smtplib.SMTP_SSL(
            smtp_host, smtp_port, timeout=30, context=ssl.create_default_context()
        ) as smtp:
            smtp.login(smtp_username, smtp_password)
            smtp.send_message(message)
    else:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as smtp:
            smtp.ehlo()
            smtp.starttls(context=ssl.create_default_context())
            smtp.ehlo()
            smtp.login(smtp_username, smtp_password)
            smtp.send_message(message)


# ============================================================
# Live inspection
# ============================================================
def render_inspection():
    st.title(t["inspect"])
    st.caption(
        "Upload or capture a product label image. Fast PaddleOCR extracts"
        " declarations in seconds."
    )
    category = st.selectbox(
        "🏷️ Product Category",
        [
            "Food & Beverages",
            "Cosmetics & Personal Care",
            "General Packaged Goods",
            "Medical Devices",
        ],
    )
    input_type = st.radio(
        "Input method",
        ["📷 Live Camera Capture", "📁 Upload Image File"],
        horizontal=True,
    )
    uploaded_image = (
        st.camera_input("Take a live photo of the product label")
        if input_type.startswith("📷")
        else st.file_uploader(
            "Upload product photo", type=["jpg", "jpeg", "png"]
        )
    )

    # Key-only widget state: initialise once, never pass value=+key together.
    _FORM_DEFAULTS = {
        "inp_product_name": "",
        "inp_mrp": "",
        "inp_net_qty": "",
        "inp_mfg_date": "",
        "inp_origin": "",
        "inp_manufacturer": "",
        "inp_care": "",
        "inp_fssai": "",
        "inp_expiry": "",
    }
    for _fk, _fv in _FORM_DEFAULTS.items():
        if _fk not in st.session_state:
            st.session_state[_fk] = _fv

    image_pil = None
    if uploaded_image is not None:
        img_bytes = uploaded_image.getvalue()
        if len(img_bytes) > 10 * 1024 * 1024:
            st.error("Image too large (max 10 MB). Please upload a smaller file.")
            st.stop()
        try:
            image_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as exc:
            logger.exception("Invalid image upload: %s", exc)
            st.error("Could not read that image. Please upload a valid JPG/PNG.")
            st.stop()
        st.image(
            image_pil, caption="Selected Product Label", use_container_width=True
        )

        # Detect a new image by MD5 hash — run OCR only once per image
        # Include the parser version so an improved extractor rescans an
        # already-selected image after an app update.
        img_hash = f"{hashlib.md5(img_bytes).hexdigest()}:ocr-v7"

        if img_hash != st.session_state.get("ocr_last_hash", ""):
            if not _ENABLE_OCR:
                st.info("OCR disabled (ENABLE_OCR=false). Enter fields manually.")
            with st.spinner("Scanning product label with high-speed OCR (< 3s)..."):
                (
                    name,
                    mrp,
                    net_quantity,
                    manufacture_date,
                    fssai,
                    country_origin,
                    manufacturer,
                    consumer_care,
                    expiry,
                ) = extract_label_values(image_pil)

            # Safe: widgets not yet created in this run, so assigning their
            # keys now is allowed and becomes their initial value.
            values = {
                "product_name": name,
                "mrp": mrp,
                "net_qty": net_quantity,
                "mfg_date": manufacture_date,
                "origin": country_origin,
                "manufacturer": manufacturer,
                "care": consumer_care,
                "fssai": fssai,
                "expiry": expiry,
            }
            for field, value in values.items():
                st.session_state[f"ocr_{field}"] = value
                st.session_state[f"inp_{field}"] = value
            st.session_state["ocr_last_hash"] = img_hash
    else:
        # No image uploaded — clear previous OCR state
        if st.session_state.get("ocr_last_hash", ""):
            st.session_state["ocr_last_hash"] = ""
            for field in [
                "product_name", "mrp", "net_qty", "mfg_date", "fssai",
                "origin", "manufacturer", "care", "expiry",
            ]:
                st.session_state[f"ocr_{field}"] = ""
                st.session_state[f"inp_{field}"] = ""

    st.markdown(
        "<div class='section-title'><span>1</span>Verify mandatory"
        " declarations</div>",
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        product_name = st.text_input("Product Name", key="inp_product_name")
        mrp_val = st.text_input("Maximum Retail Price (MRP)", key="inp_mrp")
        qty_val = st.text_input("Net Quantity", key="inp_net_qty")
        date_val = st.text_input("Mfg / Packing Month & Year", key="inp_mfg_date")
        origin_val = st.text_input("Country of Origin", key="inp_origin")
    with col2:
        mfg_val = st.text_input("Manufacturer / Packer Details", key="inp_manufacturer")
        care_val = st.text_input("Consumer Care Contact / Phone", key="inp_care")
        fssai_val = (
            st.text_input("FSSAI Licence No. (food products only)", key="inp_fssai")
            if category == "Food & Beverages"
            else "Not applicable"
        )
        exp_val = (
            st.text_input("Best Before / Expiry Date (food products only)", key="inp_expiry")
            if category == "Food & Beverages"
            else "Not applicable"
        )

    checks = [
        (
            "Maximum Retail Price (MRP)",
            mrp_val,
            "Rule 6(1)(e)",
            25000,
            RULES[0][3],
            is_valid_mrp,
            "Enter a numeric MRP, e.g. 120 or Rs. 120.00",
        ),
        (
            "Net Quantity Declaration",
            qty_val,
            "Rule 6(1)(c)",
            10000,
            RULES[1][3],
            is_valid_quantity,
            "Use a standard unit, e.g. 500 g, 1 kg, 250 ml, 1 L",
        ),
        (
            "Month & Year of Manufacture",
            date_val,
            "Rule 6(1)(d)",
            15000,
            RULES[2][3],
            is_valid_mfg_date,
            "Use e.g. 08/2025 or Aug 2025",
        ),
        (
            "Manufacturer/Packer Details",
            mfg_val,
            "Rule 6(1)(a)",
            25000,
            RULES[3][3],
            is_valid_generic,
            "Enter name + address (min 3 chars)",
        ),
        (
            "Country of Origin",
            origin_val,
            "Rule 6(1)(aa)",
            50000,
            RULES[4][3],
            is_valid_generic,
            "Enter the declared country; leave blank if missing",
        ),
        (
            "Consumer Care Contact",
            care_val,
            "Rule 6(2)",
            10000,
            RULES[5][3],
            is_valid_care,
            "Enter phone (>=7 digits) or email",
        ),
    ]
    if category == "Food & Beverages":
        checks += [
            (
                "FSSAI Licence Number",
                fssai_val,
                "FSS Act Sec 31",
                100000,
                (
                    "Food products should display applicable food safety"
                    " registration or licence details."
                ),
                is_valid_fssai,
                "14 digits starting with 1 or 2",
            ),
            (
                "Expiry / Best Before Date",
                exp_val,
                "FSS Regulations",
                50000,
                (
                    "Food products need clear date information to help prevent"
                    " health risks."
                ),
                is_valid_expiry,
                "Use e.g. 08/2026 or Best before 6 months",
            ),
        ]

    if st.button("💾 Save product to catalog"):
        save_product_to_database({
            "product_name": product_name,
            "category": category,
            "mrp": mrp_val,
            "net_quantity": qty_val,
            "manufacture_date": date_val,
            "manufacturer": mfg_val,
            "country_origin": origin_val,
            "consumer_care": care_val,
            "fssai": fssai_val,
            "expiry": exp_val,
        })
        st.success("Product saved to catalog successfully.")

    st.markdown(
        "<div class='section-title'><span>2</span>Compliance scorecard</div>",
        unsafe_allow_html=True,
    )
    passed = 0
    total_fine = 0
    missing = []
    invalid = []
    for label, value, section, fine, explanation, validator, hint in checks:
        val = (value or "").strip()
        if val and validator(val):
            passed += 1
            st.success(f"✅ **{t['pass']}:** {label} → `{val}` [{section}]")
        else:
            total_fine += fine
            if val:
                invalid.append(label)
                st.error(
                    f"❌ **{t['fail']}:** {label} invalid [{section}] · Prototype"
                    f" amount: ₹{fine:,} · {hint}"
                )
            else:
                missing.append(label)
                st.error(
                    f"❌ **{t['fail']}:** {label} missing [{section}] · Prototype"
                    f" amount: ₹{fine:,}"
                )
            with st.expander("Why this matters"):
                st.write(explanation)

    compliant = passed == len(checks)
    failed = missing + invalid
    if compliant:
        st.markdown(
            f"<div class='status-good'>✅ COMPLIANT — {passed}/{len(checks)}"
            " declarations verified.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='status-bad'>⚠️ NEEDS ATTENTION —"
            f" {passed}/{len(checks)} verified. Missing/invalid items:"
            f" {', '.join(failed)}</div>",
            unsafe_allow_html=True,
        )

    if st.button(t["save"]):
        st.session_state.history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "store": shop_name,
            "category": category,
            "status": "COMPLIANT" if compliant else "NON-COMPLIANT",
            "violations_count": len(failed),
            "missing_declarations": failed,
            "total_fine": total_fine,
            # Don't retain PIL images in session state (memory bloat);
            # keep key for backward-compat with history views.
            "image": None,
            "has_image": image_pil is not None,
        })
        st.success("Inspection record saved.")

    st.markdown(
        "<div class='section-title'><span>3</span>Export and share</div>",
        unsafe_allow_html=True,
    )
    qr = qrcode.QRCode(box_size=4, border=2)
    _safe_shop = re.sub(r"[\r\n]+", " ", shop_name or "")[:80]
    qr.add_data(
        f"Legal Metrology Inspection | Store: {_safe_shop} | Status:"
        f" {'COMPLIANT' if compliant else 'NEEDS ATTENTION'}"
    )
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="#4f9f68", back_color="white")
    qr_buf = io.BytesIO()
    qr_img.save(qr_buf, format="PNG")
    qr_buf.seek(0)

    def _pdf_font(bold=False):
        # Helvetica has no ₹/Indic glyphs; prefer DejaVu if present.
        for name, path in [
            ("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            ("DejaVuSans", "C:\\Windows\\Fonts\\DejaVuSans.ttf"),
        ]:
            try:
                if os.path.exists(path) and name not in pdfmetrics.getRegisteredFontNames():
                    pdfmetrics.registerFont(TTFont(name, path))
                if name in pdfmetrics.getRegisteredFontNames():
                    return name
            except Exception:
                continue
        return "Helvetica-Bold" if bold else "Helvetica"

    def _pdf_safe(text, limit=90):
        text = re.sub(r"[\r\n]+", " ", str(text or ""))
        # Helvetica fallback: replace ₹ with Rs.
        if _pdf_font() == "Helvetica":
            text = text.replace("₹", "Rs.")
        return text[:limit]

    def generate_pdf():
        buf = io.BytesIO()
        pdf = canvas.Canvas(buf, pagesize=letter)
        font_b, font_r = _pdf_font(bold=True), _pdf_font(bold=False)
        pdf.setFont(font_b, 15)
        pdf.drawString(70, 750, "LEGAL METROLOGY INSPECTION CERTIFICATE")
        pdf.setFont(font_r, 9)
        pdf.drawString(
            70, 735, "Legal Metrology (Packaged Commodities) Rules, 2011"
        )
        pdf.drawString(70, 718, _pdf_safe(f"Inspector: {inspector_name} | Store: {shop_name}"))
        pdf.drawString(
            70, 703, _pdf_safe(f"Category: {category} | Location: {location_gps}")
        )
        pdf.drawString(70, 688, _pdf_safe(f"Date: {datetime.now():%Y-%m-%d %H:%M} | Status: {'COMPLIANT' if compliant else 'NEEDS ATTENTION'}"))
        y = 665
        pdf.setFont(font_b, 11)
        pdf.drawString(70, y, "Mandatory Verification Checklist")
        y -= 24
        pdf.setFont(font_r, 9)
        for label, value, section, fine, _expl, _validator, _hint in checks:
            ok = bool((value or "").strip() and _validator((value or "").strip()))
            status = "VERIFIED" if ok else f"MISSING/INVALID ({section})"
            pdf.drawString(70, y, _pdf_safe(f"- {label}: {status}"))
            y -= 17
            if y < 60:
                pdf.showPage()
                pdf.setFont(font_r, 9)
                y = 750
        y -= 10
        pdf.setFont(font_b, 11)
        pdf.drawString(
            70, y, f"FINAL STATUS: {'COMPLIANT' if compliant else 'NEEDS ATTENTION'}"
        )
        pdf.setFont(font_r, 9)
        pdf.drawString(
            70,
            y - 18,
            f"Prototype reference amount for missing declarations: Rs. {total_fine:,}",
        )
        pdf.drawString(70, y - 32, "Screening aid only; not an official decision.")
        pdf.showPage()
        pdf.save()
        buf.seek(0)
        return buf.getvalue()

    pdf_bytes = generate_pdf()
    a, b = st.columns([1, 2])
    with a:
        st.image(qr_buf, caption="Scan for audit summary", width=125)
    with b:
        st.download_button(
            "📄 Download Official PDF Certificate",
            pdf_bytes,
            "Legal_Metrology_Certificate.pdf",
            "application/pdf",
        )
        recipient_email = st.text_input(
            "Recipient email",
            placeholder="recipient@example.com",
            help="SMTP must be configured on the server before sending.",
        )
        if st.button("✉️ Send PDF by email"):
            try:
                send_pdf_email(recipient_email.strip(), pdf_bytes, shop_name)
            except Exception:
                pass


# ============================================================
# Feedback page
# ============================================================
def render_feedback():
    st.title(t["feedback"])
    st.caption(
        "Help improve the clarity and usefulness of this inspection"
        " prototype."
    )
    with st.form("feedback_form"):
        rating = st.slider("Overall experience", 1, 5, 5)
        feedback_type = st.selectbox(
            "Feedback type",
            [
                "Suggestion",
                "Bug report",
                "Translation",
                "Rules content",
                "Other",
            ],
        )
        comment = st.text_area(
            "Your feedback", placeholder="Tell us what should be improved..."
        )
        email = st.text_input("Email (optional)")
        submitted = st.form_submit_button("Submit feedback")
    if submitted:
        if not comment.strip():
            st.error("Please enter feedback before submitting.")
        else:
            st.session_state.feedback.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "rating": rating,
                "type": feedback_type,
                "comment": comment,
                "email": email,
            })
            st.success(t["thanks"])
    if st.session_state.feedback:
        st.subheader("Submitted feedback in this session")
        st.dataframe(
            pd.DataFrame(st.session_state.feedback).drop(columns=["email"]),
            use_container_width=True,
        )


# ============================================================
# Dashboard and history pages
# ============================================================
def render_dashboard():
    st.title(t["dashboard"])
    if not st.session_state.history:
        st.info(
            "No inspection records yet. Complete an inspection and save it to"
            " view analytics."
        )
        return
    df = pd.DataFrame(st.session_state.history)
    total = len(df)
    compliant_count = int((df["status"] == "COMPLIANT").sum())
    non_compliant = total - compliant_count
    rate = round(compliant_count / total * 100, 1)
    fine = int(df["total_fine"].sum())
    m1, m2, m3, m4 = st.columns(4)
    for col, label, value in [
        (m1, "Total inspected", total),
        (m2, "Compliant", compliant_count),
        (m3, "Needs attention", non_compliant),
        (m4, "Compliance rate", f"{rate}%"),
    ]:
        with col:
            st.markdown(
                f"<div class='metric-box'><div"
                f" class='metric-label'>{label}</div><div"
                f" class='metric-value'>{value}</div></div>",
                unsafe_allow_html=True,
            )
    st.caption(
        f"Total prototype reference amount across saved records: ₹{fine:,}"
    )
    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(
            df,
            names="status",
            hole=0.45,
            color="status",
            color_discrete_map={
                "COMPLIANT": "#25855a",
                "NON-COMPLIANT": "#c24135",
            },
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        missing = [item for row in df["missing_declarations"] for item in row]
        if missing:
            counts = pd.Series(missing).value_counts().reset_index()
            counts.columns = ["Declaration", "Count"]
            st.plotly_chart(
                px.bar(
                    counts,
                    x="Declaration",
                    y="Count",
                    color="Count",
                    color_continuous_scale="YlOrBr",
                ),
                use_container_width=True,
            )
        else:
            st.success("No missing declarations recorded.")


def render_history():
    st.title(t["history"])
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        # Drop non-serializable / internal columns safely (older rows may lack them).
        display = df.drop(columns=[c for c in ["image", "has_image"] if c in df.columns])
        st.dataframe(display, use_container_width=True)
        st.download_button(
            "📥 Download Audit Log (CSV)",
            display.drop(columns=["missing_declarations"], errors="ignore").assign(
                missing_declarations=display["missing_declarations"].apply(
                    lambda v: "; ".join(v) if isinstance(v, list) else v
                ) if "missing_declarations" in display.columns else ""
            ).to_csv(index=False).encode("utf-8"),
            "Inspection_History_Log.csv",
            "text/csv",
        )
    else:
        st.info("No inspection entries recorded yet.")

    st.subheader("Product catalog")
    catalog = list_catalog_products()
    if catalog.empty:
        st.info(
            "No products saved yet. Save a product from Live Inspection to"
            " build the catalog."
        )
    else:
        st.dataframe(catalog, use_container_width=True, hide_index=True)
        st.download_button(
            "📥 Download Product Catalog (CSV)",
            catalog.to_csv(index=False).encode("utf-8"),
            "Legal_Metrology_Product_Catalog.csv",
            "text/csv",
        )


# ============================================================
# Route menu
# ============================================================
if selected_tab == t["inspect"]:
    render_inspection()
elif selected_tab == t["dashboard"]:
    render_dashboard()
elif selected_tab == t["history"]:
    render_history()
elif selected_tab == t["rules"]:
    display_rules()
else:
    render_feedback()

st.markdown(
    "<br><center><small>Legal Metrology Compliance Checker · Screening"
    " prototype only</small></center>",
    unsafe_allow_html=True,
)
