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

# Safe Import for Google Cloud Vision API
try:
    from google.cloud import vision
except ImportError:
    vision = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("legal_metrology")


# ============================================================
# Database Setup
# ============================================================
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
# Google Cloud Vision API OCR Integration
# ============================================================
@st.cache_resource
def get_vision_client():
    """Initialize and cache the Google Cloud Vision client."""
    if vision is None:
        st.error("Google Cloud Vision library is not installed. Please install `google-cloud-vision`.")
        return None
    try:
        # Check Streamlit secrets or environment for Google Cloud credentials if needed
        if "gcp_service_account" in st.secrets:
            return vision.ImageAnnotatorClient.from_service_account_info(
                dict(st.secrets["gcp_service_account"])
            )
        return vision.ImageAnnotatorClient()
    except Exception as e:
        logger.exception("Google Cloud Vision API client init failed: %s", e)
        st.error(f"Failed to initialize Google Vision Client: {e}")
        return None


def run_google_vision_ocr(image_bytes):
    """Executes Text Detection using Google Cloud Vision API."""
    client = get_vision_client()
    if client is None:
        return []

    try:
        image = vision.Image(content=image_bytes)
        response = client.text_detection(image=image)
        
        if response.error.message:
            logger.error("Vision API Error: %s", response.error.message)
            st.error(f"Vision API Error: {response.error.message}")
            return []

        annotations = response.text_annotations
        if annotations:
            # The first annotation contains the full text block separated by newlines
            raw_text = annotations[0].description
            return [line.strip() for line in raw_text.split("\n") if line.strip()]
    except Exception as e:
        logger.exception("Google Cloud Vision OCR failed: %s", e)
        st.error(f"Vision API processing error: {e}")
    
    return []


def extract_label_values(image_bytes):
    """Parses extracted Vision API text into field values using pattern matching."""
    lines = run_google_vision_ocr(image_bytes)
    full_text = " ".join(lines)

    fields = {
        "product_name": "", "mrp": "", "net_quantity": "",
        "manufacture_date": "", "country_origin": "",
        "manufacturer": "", "consumer_care": "",
        "fssai": "", "expiry": "",
    }

    # Pass 1: Line-by-line contextual extraction
    for i, line in enumerate(lines):
        lc = line.strip()
        ll = lc.lower()

        # Manufacturer
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

        # Expiry
        if not fields["expiry"]:
            m = re.search(
                r"(?:best\s+before|expiry\s*(?:date)?|use\s+by|exp\.?\s*date|exp\.?)[\:\s\.\-]*([a-zA-Z0-9\s/_\.\-]+)",
                lc, re.IGNORECASE)
            if m:
                val = re.split(r"\b(?:mrp|net\s*wt|lic)\b", m.group(1).strip(), flags=re.IGNORECASE)[0].strip()
                fields["expiry"] = _clean_expiry_capture(val)

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

    # Pass 2: Global text fallback
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

    if not fields["fssai"]:
        m = re.search(r"\b([12]\d{13})\b", full_text)
        if m:
            fields["fssai"] = m.group(1).strip()

    # Pass 3: Extract product name from headline text
    skip_keywords = [
        "mrp", "rs.", "price", "net wt", "net qty", "quantity",
        "mfg", "pkd", "fssai", "licence", "lic no", "batch",
        "ingredients", "nutrition", "100g", "country of origin", "veg",
    ]
    product_parts = []
    for line in lines:
        clean = re.sub(r"\s+", " ", line.strip())
        lowered = clean.lower()
        if (
            len(clean) >= 3
            and re.fullmatch(r"[A-Za-z][A-Za-z'& -]{1,40}", clean)
            and not any(keyword in lowered for keyword in skip_keywords)
        ):
            if lowered not in {part.lower() for part in product_parts}:
                product_parts.append(clean)
        if len(product_parts) == 3:
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
# Field validators
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
    v = re.sub(r"\s+", " ", (value or "").strip(" :.-\t"))
    if len(v) < 2 or not re.search(r"\d", v):
        return ""
    return v


def _clean_mrp(value):
    return value.replace(",", "").replace("/-", "").strip(" .,")


def is_valid_mrp(value):
    v = _clean_mrp(value or "")
    if not v or not _MRP_RE.match(v):
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
    if not value:
        return False
    v = value.strip()
    return bool(_DATE_RE.search(v) or _EXPIRY_DURATION_RE.search(v))


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

# Navigation and Sidebar Setup
st.sidebar.markdown(
    "<div class='side-brand'><strong>Legal Metrology</strong><small>Inspection Portal</small></div>",
    unsafe_allow_html=True,
)
selected_tab = st.sidebar.radio(
    "Navigation",
    [t["inspect"], t["dashboard"], t["history"], t["rules"], t["feedback"]],
)
st.sidebar.markdown("---")
inspector_name = st.sidebar.text_input("Inspector Name", "Inspector Officer")
shop_name = st.sidebar.text_input("Store Name", "Metro Retail Store")
location_gps = st.sidebar.text_input("Location", "Central Market")


# Rules catalog
RULES = [
    ("Maximum Retail Price (MRP)", "Rule 6(1)(e)", 25000, "MRP must be declared inclusive of all taxes."),
    ("Net Quantity Declaration", "Rule 6(1)(c)", 10000, "Net weight, measure, or volume must use recognized units."),
    ("Month & Year of Manufacture", "Rule 6(1)(d)", 15000, "Month and year of manufacture or packing must be present."),
    ("Manufacturer/Packer Details", "Rule 6(1)(a)", 25000, "Name and complete address of manufacturer."),
    ("Country of Origin", "Rule 6(1)(aa)", 50000, "Imported products should disclose the country of origin."),
    ("Consumer Care Contact", "Rule 6(2)", 10000, "Phone number or email address for consumer care."),
]


def display_rules():
    st.title("⚖️ Legal Metrology Rules & Guidance")
    for name, section, fine, explanation in RULES:
        st.markdown(
            f"<div class='rule-card'><h4>{name} <small>({section})</small></h4>"
            f"<p>{explanation}</p><p><b>Penalty reference:</b> ₹{fine:,}</p></div>",
            unsafe_allow_html=True,
        )


# ============================================================
# Live inspection
# ============================================================
def render_inspection():
    st.title(t["inspect"])
    category = st.selectbox(
        "🏷️ Product Category",
        ["Food & Beverages", "Cosmetics & Personal Care", "General Packaged Goods", "Medical Devices"],
    )
    input_type = st.radio("Input method", ["📷 Live Camera Capture", "📁 Upload Image File"], horizontal=True)
    uploaded_image = (
        st.camera_input("Take a live photo of the product label")
        if input_type.startswith("📷")
        else st.file_uploader("Upload product photo", type=["jpg", "jpeg", "png"])
    )

    _FORM_DEFAULTS = {
        "inp_product_name": "", "inp_mrp": "", "inp_net_qty": "",
        "inp_mfg_date": "", "inp_origin": "", "inp_manufacturer": "",
        "inp_care": "", "inp_fssai": "", "inp_expiry": "",
    }
    for _fk, _fv in _FORM_DEFAULTS.items():
        if _fk not in st.session_state:
            st.session_state[_fk] = _fv

    if uploaded_image is not None:
        img_bytes = uploaded_image.getvalue()
        if len(img_bytes) > 10 * 1024 * 1024:
            st.error("Image too large (max 10 MB).")
            st.stop()

        try:
            image_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            st.image(image_pil, caption="Selected Product Label", use_container_width=True)
        except Exception:
            st.error("Invalid image upload.")
            st.stop()

        img_hash = f"{hashlib.md5(img_bytes).hexdigest()}:vision-v1"

        if img_hash != st.session_state.get("ocr_last_hash", ""):
            with st.spinner("Extracting declarations using Google Cloud Vision API..."):
                (
                    name, mrp, net_quantity, manufacture_date,
                    fssai, country_origin, manufacturer, consumer_care, expiry
                ) = extract_label_values(img_bytes)

            values = {
                "product_name": name, "mrp": mrp, "net_qty": net_quantity,
                "mfg_date": manufacture_date, "origin": country_origin,
                "manufacturer": manufacturer, "care": consumer_care,
                "fssai": fssai, "expiry": expiry,
            }
            for field, value in values.items():
                st.session_state[f"ocr_{field}"] = value
                st.session_state[f"inp_{field}"] = value
            st.session_state["ocr_last_hash"] = img_hash

    st.markdown("<div class='section-title'><span>1</span>Verify mandatory declarations</div>", unsafe_allow_html=True)
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
            if category == "Food & Beverages" else "Not applicable"
        )
        exp_val = (
            st.text_input("Best Before / Expiry Date (food products only)", key="inp_expiry")
            if category == "Food & Beverages" else "Not applicable"
        )

    checks = [
        ("Maximum Retail Price (MRP)", mrp_val, "Rule 6(1)(e)", 25000, RULES[0][3], is_valid_mrp, "Enter numeric MRP"),
        ("Net Quantity Declaration", qty_val, "Rule 6(1)(c)", 10000, RULES[1][3], is_valid_quantity, "Use standard units (g, ml, kg, L)"),
        ("Month & Year of Manufacture", date_val, "Rule 6(1)(d)", 15000, RULES[2][3], is_valid_mfg_date, "Use e.g. 08/2025"),
        ("Manufacturer/Packer Details", mfg_val, "Rule 6(1)(a)", 25000, RULES[3][3], is_valid_generic, "Enter name + address"),
        ("Country of Origin", origin_val, "Rule 6(1)(aa)", 50000, RULES[4][3], is_valid_generic, "Enter country name"),
        ("Consumer Care Contact", care_val, "Rule 6(2)", 10000, RULES[5][3], is_valid_care, "Enter phone or email"),
    ]

    if st.button("💾 Save product to catalog"):
        save_product_to_database({
            "product_name": product_name, "category": category, "mrp": mrp_val,
            "net_quantity": qty_val, "manufacture_date": date_val, "manufacturer": mfg_val,
            "country_origin": origin_val, "consumer_care": care_val, "fssai": fssai_val, "expiry": exp_val,
        })
        st.success("Product saved to catalog successfully.")


# Route menu
if selected_tab == t["inspect"]:
    render_inspection()
elif selected_tab == t["rules"]:
    display_rules()