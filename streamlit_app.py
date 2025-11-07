# streamlit_app.py — IRACify demo UI
# -------------------------------------------------------------
# Vereist:
#   pip install streamlit openai>=1.42.0 jsonschema requests trafilatura pdfminer.six
#   export OPENAI_API_KEY=...  (of voer 'API sleutel' in de sidebar in)
# Run:
#   streamlit run streamlit_app.py
# -------------------------------------------------------------

import os
import json
import traceback
import streamlit as st

# Jouw module
import ai_irac_summarizer_v3 as irac

st.set_page_config(page_title="IRACify", page_icon="⚖️", layout="wide")

# ---------------- Sidebar ----------------
st.sidebar.title("⚙️ Instellingen")
api_key_input = st.sidebar.text_input("API sleutel (optioneel)", type="password", help="Laat leeg om OPENAI_API_KEY uit je environment te gebruiken.")
model = st.sidebar.text_input("Model", value=irac.MODEL_DEFAULT)
temperature = st.sidebar.slider("Temperatuur", 0.0, 1.0, 0.1, 0.05)
max_ro = st.sidebar.slider("Top-K r.o.", 3, 24, irac.TOPK_RO, 1)
timeout_s = st.sidebar.slider("Timeout (s)", 10, 180, 60, 5)
return_essentie = st.sidebar.checkbox("Voeg Essentie toe", value=True)

if api_key_input:
    st.session_state["OPENAI_API_KEY"] = api_key_input

api_key = st.session_state.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

st.sidebar.markdown("---")
st.sidebar.write("**Diepte-voorkeur**: de pipeline geeft vanzelf voorkeur aan specifieke nummers (bijv. 3.3 > 3).")

# ---------------- Tabs ----------------
st.title("IRACify — NL arresten naar IRAC ✨")
tab_url, tab_text = st.tabs(["🔗 Samenvatten vanaf URL", "📝 Samenvatten vanaf tekst"])

# Helper voor download

def _offer_download(name: str, data_obj):
    data = json.dumps(data_obj, ensure_ascii=False, indent=2)
    st.download_button(
        label=f"💾 Download {name}.json",
        data=data.encode("utf-8"),
        file_name=f"{name}.json",
        mime="application/json",
    )

# ---------------- URL Tab ----------------
with tab_url:
    st.subheader("Plak een link naar het arrest")
    url = st.text_input("URL", placeholder="https://... (ECLI of publicatie)")
    col1, col2 = st.columns([1,1])
    with col1:
        go_url = st.button("Samenvatten (URL)")
    with col2:
        st.caption("Ondersteunt HTML & PDF. Vereist 'requests', optioneel 'trafilatura' en 'pdfminer.six'.")

    if go_url and url:
        with st.spinner("Bezig met samenvatten vanaf URL..."):
            try:
                out = irac.summarize_from_url(
                    url,
                    model=model,
                    temperature=temperature,
                    timeout_s=timeout_s,
                    max_retries=5,
                    api_key=api_key,
                    max_ro_in_prompt=max_ro,
                    return_essentie=return_essentie,
                )
                st.success("Klaar ✅")
                st.json(out, expanded=False)
                _offer_download("iracify_output", out)

                # Mooie weergave
                st.markdown("---")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("ECLI's gedetecteerd", str(len(irac.extract_eclis(json.dumps(out)))) )
                c2.metric("# R.O.'s", str(len(out.get("Rechtsoverwegingen", []))))
                c3.metric("Model", model)
                c4.metric("Top-K", str(max_ro))

                st.markdown("### 📌 IRAC")
                colL, colR = st.columns([2,2])
                with colL:
                    st.markdown("**Issue**\n\n" + out.get("Issue", ""))
                    st.markdown("**Rule**\n\n" + out.get("Rule", ""))
                with colR:
                    st.markdown("**Application**\n\n" + out.get("Application", ""))
                    st.markdown("**Conclusion**\n\n" + out.get("Conclusion", ""))

                if return_essentie and isinstance(out.get("Essentie"), dict):
                    st.markdown("### 🧭 Essentie")
                    st.write(out["Essentie"].get("Essentie", ""))
                    kp = out["Essentie"].get("Kernpunten", [])
                    if kp:
                        st.markdown("**Kernpunten:**")
                        for k in kp:
                            st.write(f"- {k}")

                st.markdown("### 📚 Rechtsoverwegingen")
                for ro in out.get("Rechtsoverwegingen", []):
                    with st.expander(f"R.O. {ro.get('ro_nummer','?')} — {ro.get('rol','-')}"):
                        if ro.get("quote"):
                            st.markdown(f"> {ro['quote']}")
                        if ro.get("inhoud"):
                            st.write(ro["inhoud"])
                        cits = ro.get("citaten", [])
                        if cits:
                            st.caption("Citaten: " + ", ".join(cits))

                if out.get("Bronnen"):
                    st.markdown("### 🔗 Bronnen")
                    for b in out.get("Bronnen", []):
                        st.write("- " + str(b))

            except Exception as e:
                st.error(f"Er ging iets mis: {e}")
                st.code(traceback.format_exc())

# ---------------- Tekst Tab ----------------
with tab_text:
    st.subheader("Plak de arresttekst (of gedeelte)")
    tekst = st.text_area("Tekst", height=240, placeholder="Plak hier de volledige tekst of een relevant deel met r.o.-nummers…")
    c1, c2 = st.columns([1,1])
    with c1:
        go_text_irac = st.button("IRAC-samenvatting (Tekst)")
    with c2:
        go_text_ess = st.button("Alleen Essentie (Tekst)")

    # IRAC op tekst
    if go_text_irac and tekst.strip():
        with st.spinner("Bezig met IRAC (tekst)…"):
            try:
                out = irac.summarize_case_irac(
                    tekst=tekst,
                    model=model,
                    temperature=temperature,
                    timeout_s=timeout_s,
                    max_retries=5,
                    api_key=api_key,
                    max_ro_in_prompt=max_ro,
                )
                if return_essentie:
                    try:
                        out["Essentie"] = irac.summarize_case_essentie(
                            tekst=tekst,
                            model=model,
                            temperature=temperature,
                            timeout_s=max(30, timeout_s-10),
                            max_retries=4,
                            api_key=api_key,
                        )
                    except Exception as e:
                        st.warning(f"Essentie niet gelukt (ga verder met IRAC): {e}")

                st.success("Klaar ✅")
                st.json(out, expanded=False)
                _offer_download("iracify_output", out)

            except Exception as e:
                st.error(f"Er ging iets mis: {e}")
                st.code(traceback.format_exc())

    # Alleen Essentie
    if go_text_ess and tekst.strip():
        with st.spinner("Bezig met Essentie (tekst)…"):
            try:
                out = irac.summarize_case_essentie(
                    tekst=tekst,
                    model=model,
                    temperature=temperature,
                    timeout_s=timeout_s,
                    max_retries=4,
                    api_key=api_key,
                )
                st.success("Klaar ✅")
                st.json(out, expanded=False)
                _offer_download("iracify_essentie", out)
            except Exception as e:
                st.error(f"Er ging iets mis: {e}")
                st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.caption("IRACify • voorkeur voor specifieke r.o.-nummers • json-schema validatie • backoff + timeouts")
