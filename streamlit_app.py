# streamlit_app.py
import json
import streamlit as st
from ai_irac_summarizer_v2 import summarize_case_irac

# -------------------- UI Config --------------------
st.set_page_config(page_title="⚖️ IRAC Samenvatter NL", layout="wide")
st.title("⚖️ IRAC Samenvatter voor Nederlandse Arresten")

with st.sidebar:
    st.header("Instellingen")
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)
    temperature = st.slider("Creativiteit (temperature)", 0.0, 1.0, 0.1, 0.05)
    st.markdown("---")
    st.caption("Zorg dat `OPENAI_API_KEY` in je environment staat.")

# -------------------- Input --------------------
st.subheader("Zaaktekst")
tekst = st.text_area(
    "Plak hier je arrest of relevante passage (inclusief R.O.'s):",
    height=300,
    placeholder="Bijv. 'r.o. 3.1 De maatstaf volgt uit art. 6 EVRM...'",
)

# -------------------- Helper --------------------
def role_color(rol: str) -> str:
    if not rol:
        return "#d0d0d0"
    rol = rol.lower()
    if rol.startswith("rule"):
        return "#90cdf4"  # blauw
    if rol.startswith("application"):
        return "#f6ad55"  # oranje
    if rol.startswith("conclusion"):
        return "#68d391"  # groen
    return "#e2e8f0"     # grijs

# -------------------- Main Action --------------------
if st.button("Samenvatten", type="primary", use_container_width=True):
    if not tekst.strip():
        st.warning("Voer eerst een arresttekst in.")
    else:
        with st.spinner("Samenvatten en analyseren..."):
            try:
                data = summarize_case_irac(tekst=tekst, model=model, temperature=temperature)
            except Exception as e:
                st.error(f"Er is iets misgegaan: {e}")
                st.stop()

        st.success("Samenvatting voltooid ✅")

        # --------------- Layout ---------------
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("IRAC Analyse")
            st.markdown(f"### 🧩 **Issue**\n{data.get('Issue', '')}")
            st.markdown(f"### ⚖️ **Rule**\n{data.get('Rule', '')}")
            st.markdown(f"### 🧠 **Application**\n{data.get('Application', '')}")
            st.markdown(f"### ✅ **Conclusion**\n{data.get('Conclusion', '')}")

            st.markdown("---")
            st.subheader("📜 Relevante Rechtsoverwegingen")

            ros = data.get("Rechtsoverwegingen", [])
            if ros:
                for ro in ros:
                    rol = ro.get("rol", "")
                    kleur = role_color(rol)
                    st.markdown(
                        f"""
                        <div style='background-color:{kleur}; padding:10px; border-radius:10px; margin-bottom:8px;'>
                        <b>R.O. {ro.get('ro_nummer','')}</b> — <i>{rol}</i><br>
                        <blockquote>{ro.get('quote','')}</blockquote>
                        {ro.get('inhoud','')}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    cits = ro.get("citaten", [])
                    if cits:
                        st.caption("📖 Citaten: " + "; ".join(cits))
            else:
                st.info("Geen specifieke rechtsoverwegingen gevonden.")

        with col2:
            st.subheader("🔗 Bronnen")
            bronnen = data.get("Bronnen", [])
            if bronnen:
                for b in bronnen:
                    st.code(b)
            else:
                st.text("Geen ECLI of bronverwijzingen gevonden.")

            st.markdown("---")
            st.subheader("🧾 JSON-output")
            st.code(json.dumps(data, ensure_ascii=False, indent=2))

# -------------------- Footer --------------------
st.markdown("---")
st.caption("Gemaakt door Ilias — AI Legal Summarizer Prototype ⚖️")
