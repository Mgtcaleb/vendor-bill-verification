import sys

import streamlit as st
import pandas as pd
import tempfile
import os
import json
from pathlib import Path

# Set Hugging Face Token for authenticated Hub requests if defined in Streamlit secrets
if "HF_TOKEN" not in os.environ:
    try:
        if hasattr(st, "secrets") and "HF_TOKEN" in st.secrets:
            os.environ["HF_TOKEN"] = str(st.secrets["HF_TOKEN"])
    except Exception:
        pass

# Import core logic from the existing script
from invoice_savings_compare import (
    parse_invoice,
    load_vendor_sheet,
    match_items,
    _build_match
)

st.set_page_config(page_title="Invoice Savings Compare", layout="wide")

st.title("Invoice Savings Comparer & Analyzer")
st.markdown("Upload your vendor invoices (PDF or Images) and vendor pricing sheet to see how much you are saving and whether you should buy.")

# Initialize session state for mapping and data
if "mapping" not in st.session_state:
    mapping_file = Path("mapping.json")
    if mapping_file.exists():
        try:
            st.session_state["mapping"] = json.loads(mapping_file.read_text())
        except json.JSONDecodeError:
            st.session_state["mapping"] = {}
    else:
        st.session_state["mapping"] = {}

if "needs_review" not in st.session_state:
    st.session_state["needs_review"] = []
if "matched" not in st.session_state:
    st.session_state["matched"] = []
if "unmatched" not in st.session_state:
    st.session_state["unmatched"] = []
if "vendor_df" not in st.session_state:
    st.session_state["vendor_df"] = None
if "has_run" not in st.session_state:
    st.session_state["has_run"] = False
if "debug_info" not in st.session_state:
    st.session_state["debug_info"] = {}

# Sidebar for file uploads
st.sidebar.header("Upload Files")
invoice_files = st.sidebar.file_uploader(
    "Upload Invoices (PDF or Images)",
    type=["pdf", "png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True
)
vendor_file = st.sidebar.file_uploader("Upload Vendor Sheet", type=["xlsx", "xls"])

col_proc, col_clear = st.sidebar.columns(2)

with col_proc:
    sidebar_proc_btn = st.button("Process Invoices", type="primary", use_container_width=True)
    
with col_clear:
    if st.button("Clear Data", use_container_width=True):
        st.session_state["matched"] = []
        st.session_state["needs_review"] = []
        st.session_state["unmatched"] = []
        st.session_state["vendor_df"] = None
        st.session_state["has_run"] = False
        st.session_state["debug_info"] = {}
        st.rerun()

st.sidebar.divider()
st.sidebar.subheader("Advanced")
if st.sidebar.button("🗑️ Clear Saved Mappings", help="Deletes mapping.json. Use this if you made a mistake confirming a match and want to recheck everything from scratch.", use_container_width=True):
    mapping_file = Path("mapping.json")
    if mapping_file.exists():
        mapping_file.unlink()
    st.session_state["mapping"] = {}
    st.sidebar.success("Mappings cleared! You can now click Process Invoices to recheck everything.")

# Main page call-to-action when files are uploaded
main_proc_btn = False
if invoice_files and vendor_file:
    st.info(f"📁 **Files Ready to Check**: {len(invoice_files)} invoice(s) and 1 vendor sheet ({vendor_file.name}).")
    main_proc_btn = st.button("🚀 Run Invoice & Savings Check", type="primary", use_container_width=True)
elif not invoice_files and not vendor_file:
    st.info("👈 Please upload your invoice file(s) and vendor Excel sheet using the sidebar on the left to get started.")
elif not invoice_files:
    st.warning("👈 Please upload at least one invoice (PDF or Image) in the sidebar.")
elif not vendor_file:
    st.warning("👈 Please upload your vendor Excel sheet in the sidebar.")

if sidebar_proc_btn or main_proc_btn:
    if not invoice_files or not vendor_file:
        st.error("Please upload both the invoice file(s) and the vendor sheet before processing.")
    else:
        with st.spinner("Analyzing invoices and calculating savings..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                # Save vendor sheet
                vendor_path = os.path.join(tmpdir, vendor_file.name)
                with open(vendor_path, "wb") as f:
                    f.write(vendor_file.getbuffer())
                
                try:
                    vendor_df = load_vendor_sheet(vendor_path)
                    st.success(f"✅ Loaded vendor sheet with {len(vendor_df)} products.")
                except Exception as e:
                    st.error(f"❌ Failed to load vendor sheet: {e}")
                    st.stop()
                
                all_matched, all_needs_review, all_unmatched = [], [], []
                debug_info = {}
                
                for inv_file in invoice_files:
                    inv_path = os.path.join(tmpdir, inv_file.name)
                    with open(inv_path, "wb") as f:
                        f.write(inv_file.getbuffer())
                    
                    try:
                        invoice_items = parse_invoice(inv_path)
                        debug_info[inv_file.name] = len(invoice_items)
                        
                        if len(invoice_items) == 0:
                            st.warning(f"⚠️ No line items could be parsed from `{inv_file.name}`.")
                            try:
                                with pdfplumber.open(inv_path) as pdf:
                                    sample_text = "\n".join(page.extract_text() or "" for page in pdf.pages[:2])
                                    if sample_text.strip():
                                        with st.expander(f"🔍 View extracted text from `{inv_file.name}`"):
                                            st.code(sample_text[:2500])
                                    else:
                                        st.info(f"ℹ️ `{inv_file.name}` appears to be a scanned image PDF without embedded text.")
                            except Exception:
                                pass
                        else:
                            st.success(f"✅ Parsed {len(invoice_items)} line items from `{inv_file.name}`.")
                    except Exception as e:
                        st.error(f"❌ Error parsing `{inv_file.name}`: {e}")
                        continue
                        
                    matched, needs_review, unmatched = match_items(
                        invoice_items, 
                        vendor_df, 
                        st.session_state["mapping"], 
                        interactive=False
                    )
                    all_matched.extend(matched)
                    all_needs_review.extend(needs_review)
                    all_unmatched.extend(unmatched)
                
                # Update session state
                st.session_state["matched"] = all_matched
                st.session_state["needs_review"] = all_needs_review
                st.session_state["unmatched"] = all_unmatched
                st.session_state["vendor_df"] = vendor_df
                st.session_state["has_run"] = True
                st.session_state["debug_info"] = debug_info
                
                # Save mapping to disk
                with open("mapping.json", "w") as f:
                    json.dump(st.session_state["mapping"], f, indent=2)
                
                st.sidebar.success("Processing Complete!")

# Dashboard and Review UI
if st.session_state.get("has_run"):
    st.header("Dashboard")
    
    total_savings = sum(m["total_savings"] for m in st.session_state["matched"])
    total_items = len(st.session_state["matched"])
    total_unmatched = len(st.session_state["unmatched"])
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Savings", f"₹ {total_savings:,.2f}")
    col2.metric("Matched Items", total_items)
    col3.metric("Unmatched Items", total_unmatched)
    
    if total_items == 0 and total_unmatched == 0:
        st.warning("⚠️ No invoice items were detected to compare. If your invoice is a scanned PDF or photo, make sure the text is legible.")
    elif total_savings > 0:
        st.success("💰 Positive Savings! The current purchases are cost-effective (Recommendation: **Should Buy**).")
    elif total_savings < 0:
        st.error("⚠️ Negative Savings! You are paying more than the vendor sheet transfer price (Recommendation: **Re-evaluate / Do Not Buy**).")
    else:
        st.info("Neutral Savings. Cost is exactly matching the vendor sheet.")

    tabs = st.tabs(["Matched Items (" + str(len(st.session_state["matched"])) + ")", 
                    "Unmatched (" + str(len(st.session_state["unmatched"])) + ")"])
    
    with tabs[0]:
        st.subheader("Confirmed Matched Items")
        if st.session_state["matched"]:
            df_matched = pd.DataFrame(st.session_state["matched"])
            st.dataframe(df_matched, use_container_width=True)
            
            csv = df_matched.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Matched Items as CSV",
                data=csv,
                file_name='matched_savings.csv',
                mime='text/csv',
            )
        else:
            st.info("No matched items yet.")
            
    with tabs[1]:
        st.subheader("Unmatched Items")
        if st.session_state["unmatched"]:
            df_unmatched = pd.DataFrame(st.session_state["unmatched"])
            st.dataframe(df_unmatched, use_container_width=True)
            
            csv = df_unmatched.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Unmatched Items as CSV",
                data=csv,
                file_name='unmatched_items.csv',
                mime='text/csv',
            )
            
            # Create a dedicated text log for investigation
            log_text = "=== UNMATCHED ITEMS INVESTIGATION LOG ===\n\n"
            for row in st.session_state["unmatched"]:
                log_text += f"Invoice: {row['source_invoice']}\n"
                log_text += f"Item: {row['description']} (Brand: {row.get('brand_tag', 'None')})\n"
                log_text += f"Reason: {row.get('investigation_reason', 'No reason provided')}\n"
                log_text += "-"*50 + "\n"
                
            st.download_button(
                label="📄 Download Full Investigation Log (.txt)",
                data=log_text.encode('utf-8'),
                file_name='investigation_log.txt',
                mime='text/plain',
            )
        else:
            st.info("No unmatched items.")
