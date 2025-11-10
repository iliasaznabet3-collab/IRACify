# streamlit_app.py
import io
import json
import streamlit as st

from ai_irac_summarizer_v3 import (
    summarize_case_irac,
    summarize_from_url,
    summarize_case_essentie,
    LLMError,
)

# =========================
# Styling helpers
# =========================
def _inject_theme_css():
    if st.session_state.get("_theme_css_injected"):
        return
    st.markdown("""
    <style>
      /* ---------- Baseline Cards ---------- */
      .card {
        padding:18px 18px 14px;
        border-radius:14px;
        border:1px solid rgba(0,0,0,0.06);
        box-shadow:0 1px 6px rgba(0,0,0,0.04);
        margin:14px 0;
        transition:box-shadow 200ms ease;
      }
      .card:hover { box-shadow:0 4px 16px rgba(0,0,0,0.08); }
      .card h3 { margin:0 0 10px 0; font-size:1.05rem; }

      /* ---------- Section Colors ---------- */
      .card-irac     { background:linear-gradient(180deg,rgba(37,99,235,0.09),rgba(37,99,235,0.05)); border-color:rgba(37,99,235,0.25);}
      .card-essentie { background:linear-gradient(180deg,rgba(234,88,12,0.10),rgba(234,88,12,0.06)); border-color:rgba(234,88,12,0.30);}
      .card-ro       { background:linear-gradient(180deg,rgba(124,58,237,0.10),rgba(124,58,237,0.06)); border-color:rgba(124,58,237,0.28);}
      .card-bronnen  { background:linear-gradient(180deg,rgba(22,163,74,0.10),rgba(22,163,74,0.06)); border-color:rgba(22,163,74,0.28);}

      /* ---------- Title gradient ---------- */
      .title-gradient {
        background:linear-gradient(90deg,#2563eb,#7c3aed,#16a34a);
        -webkit-background-clip:text; background-clip:text; color:transparent;
        font-weight:800;
      }

      /* ---------- Badges ---------- */
      .badge {
        display:inline-block; padding:2px 8px; border-radius:12px;
        font-size:12px; font-weight:600; line-height:18px;
        color:#fff; margin-right:6px; vertical-align:middle;
      }
      .badge-rule  { background:#2563eb; }
      .badge-app   { background:#7c3aed; }
      .badge-concl { background:#16a34a; }
      .badge-over  { background:#6b7280; }

      .ro-header { font-weight:600; margin:8px 0 4px; }
    </style>
    """, unsafe_allow_html=True)
    st.session_state["_theme_css_injected"] = True


# =========================
# Data helpers
# =========================
def clean_irac_for_display(data: dict) -> dict:
    return {
        "Issue": data.get("Issue"),
        "Rule": data.get("Rule"),
        "Application": data.get("Application"),
        "Conclusion": data.get("Conclusion"),
        "Essentie": data.get("Essentie"),
        "Rechtsoverwegingen": data.get("Rechtsoverwegingen", []),
        "Bronnen": data.get("Bronnen", []),
    }

def _num_key(num: str):
    try:
        return tuple(int(p) for p in str(num).split("."))
    except Exception:
        return (10**9,)

# ---------- badge/emoji helpers ----------
def _role_badge_html(rol: str) -> str:
    r = (rol or "Overig").lower()
    if r.startswith("rule"):       cls = "badge-rule";  label = "Rule"
    elif r.startswith("app"):      cls = "badge-app";   label = "Application"
    elif r.startswith("concl"):    cls = "badge-concl"; label = "Conclusion"
    else:                          cls = "badge-over";  label = "Overig"
    return f'<span class="badge {cls}">{label}</span>'

def _role_emoji(rol: str) -> str:
    r = (rol or "Overig").lower()
    if r.startswith("rule"): return "🟦"
    if r.startswith("app"):  return "🟪"
    if r.startswith("concl"):return "🟩"
    return "⚪"

# =========================
# Renderers
# =========================
def render_irac(data: dict) -> None:
    _inject_theme_css()
    st.markdown('<div class="card card-irac">', unsafe_allow_html=True)
    st.markdown("<h3>IRAC</h3>", unsafe_allow_html=True)
    st.markdown(f"**Issue**\n\n{data.get('Issue','')}")
    st.markdown(f"**Rule**\n\n{data.get('Rule','')}")
    st.markdown(f"**Application**\n\n{data.get('Application','')}")
    st.markdown(f"**Conclusion**\n\n{data.get('Conclusion','')}")
    st.markdown("</div>", unsafe_allow_html=True)

    ess = data.get("Essentie")
    if isinstance(ess, dict):
        st.markdown('<div class="card card-essentie">', unsafe_allow_html=True)
        st.markdown("<h3>Essentie</h3>", unsafe_allow_html=True)
        if ess.get("Essentie"):
            st.write(ess["Essentie"])
        for kp in ess.get("Kernpunten", []):
            st.markdown(f"- {kp}")
        st.markdown("</div>", unsafe_allow_html=True)

def render_ros(data: dict) -> None:
    _inject_theme_css()
    ros = data.get("Rechtsoverwegingen", [])
    if not ros:
        return
    st.markdown('<div class="card card-ro">', unsafe_allow_html=True)
    st.markdown("<h3>Rechtsoverwegingen (r.o.)</h3>", unsafe_allow_html=True)
    with st.expander("Legenda (rollen)", expanded=False):
        st.markdown(
            _role_badge_html("Rule") + _role_badge_html("Application") +
            _role_badge_html("Conclusion") + _role_badge_html("Overig"),
            unsafe_allow_html=True
        )
    ros_sorted = sorted(ros, key=lambda x: _num_key(x.get("ro_nummer", "")))
    for ro in ros_sorted:
        rn, rol = ro.get("ro_nummer",""), ro.get("rol","Overig")
        quote, inhoud = ro.get("quote",""), ro.get("inhoud","")
        cits = ro.get("citaten",[]) or []
        with st.expander(f"{_role_emoji(rol)}  r.o. {rn}  ·  {rol}"):
            st.markdown(_role_badge_html(rol), unsafe_allow_html=True)
            st.markdown(f'<div class="ro-header">r.o. {rn}</div>', unsafe_allow_html=True)
            if quote: st.markdown(f"> {quote}")
            if inhoud: st.markdown(inhoud)
            if cits:
                st.markdown("**Verwijzingen:**")
                for c in cits: st.markdown(f"- {c}")
    st.markdown("</div>", unsafe_allow_html=True)

def render_bronnen(data: dict) -> None:
    _inject_theme_css()
    bronnen = data.get("Bronnen", [])
    if not bronnen: return
    st.markdown('<div class="card card-bronnen">', unsafe_allow_html=True)
    st.markdown("<h3>Bronnen</h3>", unsafe_allow_html=True)
    for b in bronnen: st.markdown(f"- {b}")
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# File upload helper
# =========================
def extract_text_from_upload(file) -> str:
    name = (file.name or "").lower()
    buf = file.read()
    if name.endswith(".pdf"):
        try:
            from pdfminer.high_level import extract_text
            return extract_text(io.BytesIO(buf))
        except Exception:
            return ""
    try:
        return buf.decode("utf-8", errors="replace")
    except Exception:
        return ""

# =========================
# UI
# =========================
st.set_page_config(page_title="IRACify", page_icon="⚖️", layout="centered")
st.markdown('<h1 class="title-gradient">IRACify ⚖️</h1>', unsafe_allow_html=True)
st.caption("AI-IRAC-samenvatter voor Nederlandse arresten • nu met kleur en flair ✨")

with st.sidebar:
    st.header("Invoer & instellingen")
    dev_mode = st.checkbox("Developer mode (debug)", value=False)
    model = st.text_input("Model (optioneel)", value="gpt-4o-mini")
    topk = st.number_input("Top-K R.O.'s", min_value=3, max_value=20, value=12, step=1)
    temp = st.slider("Temperatuur", 0.0, 1.0, 0.1, 0.1)
    st.divider()
    st.caption("API-key wordt uit omgeving gelezen (OPENAI_API_KEY).")

tabs = st.tabs(["Tekst", "URL", "Upload"])

# ---------- TAB: Tekst ----------
with tabs[0]:
    st.subheader("Plak de zaaktekst")
    tekst = st.text_area("Volledige tekst van het arrest", height=260, label_visibility="collapsed")
    if st.button("Samenvatten (Tekst)"):
        if not tekst.strip():
            st.warning("Voer eerst tekst in.")
        else:
            with st.spinner("Samenvatten…"):
                try:
                    irac = summarize_case_irac(
                        tekst=tekst, model=model, temperature=float(temp), max_ro_in_prompt=int(topk)
                    )
                    if "Essentie" not in irac:
                        try: irac["Essentie"] = summarize_case_essentie(tekst, model=model, temperature=float(temp))
                        except Exception: pass
                    show = clean_irac_for_display(irac)
                    render_irac(show); render_ros(show); render_bronnen(show)
                    st.download_button("Download iracify_output.json",
                        data=json.dumps(show, ensure_ascii=False, indent=2).encode("utf-8"),
                        file_name="iracify_output.json", mime="application/json")
                    if dev_mode:
                        with st.expander("Raw JSON (debug)"): st.json(irac)
                except Exception as e:
                    st.error(str(e))

# ---------- TAB: URL ----------
with tabs[1]:
    st.subheader("Geef een URL naar het arrest")
    url = st.text_input("https:// …", placeholder="Plak hier de link (ECLI-pagina of PDF).")
    if st.button("Samenvatten (URL)"):
        if not url.strip():
            st.warning("Voer eerst een geldige URL in.")
        else:
            with st.spinner("URL ophalen en samenvatten…"):
                try:
                    irac = summarize_from_url(
                        url=url, model=model, temperature=float(temp),
                        max_ro_in_prompt=int(topk), return_essentie=True
                    )
                    show = clean_irac_for_display(irac)
                    render_irac(show); render_ros(show); render_bronnen(show)
                    st.download_button("Download iracify_output.json",
                        data=json.dumps(show, ensure_ascii=False, indent=2).encode("utf-8"),
                        file_name="iracify_output.json", mime="application/json")
                    if dev_mode:
                        with st.expander("Raw JSON (debug)"): st.json(irac)
                except Exception as e:
                    st.error(str(e))

# ---------- TAB: Upload ----------
with tabs[2]:
    st.subheader("Upload een bestand (PDF of TXT/MD)")
    up = st.file_uploader("Kies een bestand", type=["pdf", "txt", "md"])
    if st.button("Samenvatten (Upload)"):
        if not up:
            st.warning("Upload eerst een bestand.")
        else:
            with st.spinner("Bestand verwerken en samenvatten…"):
                try:
                    text = extract_text_from_upload(up)
                    if not text or len(text.strip()) < 100:
                        st.error("Kon onvoldoende tekst uit het bestand halen.")
                    else:
                        irac = summarize_case_irac(
                            tekst=text, model=model, temperature=float(temp), max_ro_in_prompt=int(topk)
                        )
                        if "Essentie" not in irac:
                            try: irac["Essentie"] = summarize_case_essentie(text, model=model, temperature=float(temp))
                            except Exception: pass
                        show = clean_irac_for_display(irac)
                        render_irac(show); render_ros(show); render_bronnen(show)
                        st.download_button("Download iracify_output.json",
                            data=json.dumps(show, ensure_ascii=False, indent=2).encode("utf-8"),
                            file_name="iracify_output.json", mime="application/json")
                        if dev_mode:
                            with st.expander("Raw JSON (debug)"): st.json(irac)
                except Exception as e:
                    st.error(str(e))
