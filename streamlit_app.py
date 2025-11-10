# streamlit_app.py
import io
import os
import json
from typing import Dict, Any, List
import streamlit as st

# === Backend IRACify (zorg dat dit bestand in je project staat) ===
from ai_irac_summarizer_v3 import (
    summarize_case_irac,
    summarize_from_url,
    summarize_case_essentie,
    LLMError,
)

# ===============================
# SAFE SECRETS + ADMIN HANDLING
# ===============================
def _secret(key: str, default=None):
    """Lees uit st.secrets of env; val veilig terug op default zonder crash."""
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)

def _is_admin() -> bool:
    """Admin alleen als URL-token overeenkomt met ADMIN_TOKEN (uit secrets/env)."""
    try:
        token = st.query_params.get("admin", "")
    except Exception:
        token = ""
    admin_token = _secret("ADMIN_TOKEN", "")
    return bool(token) and admin_token and token == admin_token

def _sidebar_hide_css():
    st.markdown("""<style>
      [data-testid="stSidebar"], [data-testid="stSidebarNav"]{display:none!important}
      [data-testid="collapsedControl"]{display:none!important}
    </style>""", unsafe_allow_html=True)

def get_settings():
    """Veilige defaults + admin-instellingen (sidebar alleen voor admin)."""
    DEFAULT_MODEL = _secret("DEFAULT_MODEL", "gpt-4o-mini")
    DEFAULT_TOPK = int(_secret("DEFAULT_TOPK", 12))
    DEFAULT_TEMP = float(_secret("DEFAULT_TEMP", 0.1))
    DEFAULT_QUIZ_Q = int(_secret("DEFAULT_QUIZ_Q", 5))

    if _is_admin():
        with st.sidebar:
            st.header("Instellingen (admin)")
            dev_mode = st.checkbox("Developer mode (debug)", value=False)
            model = st.text_input("Model (samenvatten & quiz)", value=DEFAULT_MODEL)
            topk = st.number_input("Top-K R.O.'s", min_value=3, max_value=20, value=DEFAULT_TOPK, step=1)
            temp = st.slider("Temperatuur (samenvatten)", 0.0, 1.0, DEFAULT_TEMP, 0.05)
            quiz_q = st.number_input("Aantal quizvragen", min_value=4, max_value=5, value=DEFAULT_QUIZ_Q, step=1)
        return dict(model=model, topk=int(topk), temp=float(temp), quiz_q=int(quiz_q), dev_mode=dev_mode, is_admin=True)
    else:
        _sidebar_hide_css()
        return dict(model=DEFAULT_MODEL, topk=DEFAULT_TOPK, temp=DEFAULT_TEMP, quiz_q=DEFAULT_QUIZ_Q, dev_mode=False, is_admin=False)

# ===============================
# UI STYLING (kleur, badges)
# ===============================
def _inject_theme_css():
    if st.session_state.get("_theme_css_injected"):
        return
    st.markdown("""
    <style>
      .card{padding:18px 18px 14px;border-radius:14px;border:1px solid rgba(0,0,0,0.06);
            box-shadow:0 1px 6px rgba(0,0,0,0.04);margin:14px 0;transition:box-shadow .2s}
      .card:hover{box-shadow:0 4px 16px rgba(0,0,0,0.08)}
      .card h3{margin:0 0 10px 0;font-size:1.05rem}
      .card-irac{background:linear-gradient(180deg,rgba(37,99,235,.09),rgba(37,99,235,.05));border-color:rgba(37,99,235,.25)}
      .card-essentie{background:linear-gradient(180deg,rgba(234,88,12,.10),rgba(234,88,12,.06));border-color:rgba(234,88,12,.30)}
      .card-ro{background:linear-gradient(180deg,rgba(124,58,237,.10),rgba(124,58,237,.06));border-color:rgba(124,58,237,.28)}
      .card-bronnen{background:linear-gradient(180deg,rgba(22,163,74,.10),rgba(22,163,74,.06));border-color:rgba(22,163,74,.28)}
      .card-quiz{background:linear-gradient(180deg,rgba(2,132,199,.10),rgba(2,132,199,.06));border-color:rgba(2,132,199,.28)}
      .title-gradient{background:linear-gradient(90deg,#2563eb,#7c3aed,#16a34a);
        -webkit-background-clip:text;background-clip:text;color:transparent;font-weight:800}
      .badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:12px;font-weight:600;
              line-height:18px;color:#fff;margin-right:6px;vertical-align:middle}
      .badge-rule{background:#2563eb}.badge-app{background:#7c3aed}.badge-concl{background:#16a34a}.badge-over{background:#6b7280}
      .ro-header{font-weight:600;margin:8px 0 4px}
      .correct{background:rgba(22,163,74,.12);padding:8px 10px;border-radius:8px}
      .wrong{background:rgba(239,68,68,.12);padding:8px 10px;border-radius:8px}
    </style>
    """, unsafe_allow_html=True)
    st.session_state["_theme_css_injected"] = True

def _role_badge_html(rol: str) -> str:
    r = (rol or "Overig").lower()
    if r.startswith("rule"): cls,label="badge-rule","Rule"
    elif r.startswith("app"): cls,label="badge-app","Application"
    elif r.startswith("concl"): cls,label="badge-concl","Conclusion"
    else: cls,label="badge-over","Overig"
    return f'<span class="badge {cls}">{label}</span>'

def _role_emoji(rol: str) -> str:
    r = (rol or "Overig").lower()
    if r.startswith("rule"): return "🟦"
    if r.startswith("app"):  return "🟪"
    if r.startswith("concl"):return "🟩"
    return "⚪"

# ===============================
# RENDERERS
# ===============================
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
    try: return tuple(int(p) for p in str(num).split("."))
    except Exception: return (10**9,)

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
        if ess.get("Essentie"): st.write(ess["Essentie"])
        for kp in ess.get("Kernpunten", []): st.markdown(f"- {kp}")
        st.markdown("</div>", unsafe_allow_html=True)

def render_ros(data: dict) -> None:
    _inject_theme_css()
    ros = data.get("Rechtsoverwegingen", [])
    if not ros: return
    st.markdown('<div class="card card-ro">', unsafe_allow_html=True)
    st.markdown("<h3>Rechtsoverwegingen (r.o.)</h3>", unsafe_allow_html=True)
    with st.expander("Legenda (rollen)", expanded=False):
        st.markdown(
            _role_badge_html("Rule")+_role_badge_html("Application")+
            _role_badge_html("Conclusion")+_role_badge_html("Overig"),
            unsafe_allow_html=True
        )
    ros_sorted = sorted(ros, key=lambda x: _num_key(x.get("ro_nummer","")))
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

# ===============================
# FILE UPLOAD (PDF/TXT/MD)
# ===============================
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

# ===============================
# QUIZ (generatie + validatie)
# ===============================
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False
    OpenAI = None  # type: ignore

def _build_quiz_prompt(irac: Dict[str, Any], n_questions: int = 5) -> str:
    issue = irac.get("Issue",""); rule = irac.get("Rule","")
    app = irac.get("Application",""); concl = irac.get("Conclusion","")
    ro_snips=[]
    for ro in irac.get("Rechtsoverwegingen", [])[:8]:
        rn = ro.get("ro_nummer",""); inhoud = (ro.get("inhoud","") or "")[:300]
        ro_snips.append(f"[r.o. {rn}] {inhoud}")
    return (
        "Maak een multiple-choice quiz (4–5 vragen, A–D) op basis van dit arrest. "
        "Output STRIKT JSON: {\"quiz\":[{\"question\":\"...\",\"choices\":{\"A\":\"...\",\"B\":\"...\",\"C\":\"...\",\"D\":\"...\"},\"correct\":\"A|B|C|D\",\"explanation\":\"...\",\"ro_refs\":[\"3.1\"]}]}."
        f"\nAantal vragen: {n_questions}.\n\n=== IRAC ===\nIssue: {issue}\nRule: {rule}\nApplication: {app}\nConclusion: {concl}\n\n=== r.o.'s ===\n" + "\n".join(ro_snips)
    )

def _validate_quiz_payload(data: dict, n_questions: int) -> dict:
    quiz = data.get("quiz", []); cleaned=[]
    for q in quiz:
        question = str(q.get("question","")).strip()
        choices = q.get("choices", {}) or {}
        correct = str(q.get("correct","")).strip().upper()
        expl = str(q.get("explanation","")).strip()
        refs = [str(r) for r in (q.get("ro_refs", []) or [])]
        for k in ["A","B","C","D"]:
            choices[k] = str(choices.get(k,"")).strip()
        if not question or correct not in ["A","B","C","D"]:
            continue
        cleaned.append({"question":question,"choices":{k:choices[k] for k in ["A","B","C","D"]},
                        "correct":correct,"explanation":expl,"ro_refs":refs})
    if len(cleaned) < 4:
        raise RuntimeError("Onvoldoende valide quizvragen ontvangen van het model.")
    return {"quiz": cleaned[:n_questions]}

def generate_quiz_from_irac(irac: dict, model: str = "gpt-4o-mini", n_questions: int = 5) -> dict:
    if not _HAS_OPENAI:
        raise RuntimeError("OpenAI SDK niet geïnstalleerd. pip install -U openai")
    client = OpenAI()
    prompt = _build_quiz_prompt(irac, n_questions)
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.1, seed=0,
            response_format={"type": "json_object"},
            messages=[{"role":"system","content":"Output strikt JSON; geen extra tekst."},
                      {"role":"user","content":prompt}],
            timeout=45,
        )
        data = json.loads(resp.choices[0].message.content)
        return _validate_quiz_payload(data, n_questions)
    except Exception as e:
        raise RuntimeError(f"Quizgeneratie faalde: {e}")

# ===============================
# APP SHELL
# ===============================
st.set_page_config(page_title="IRACify", page_icon="⚖️", layout="centered", initial_sidebar_state="collapsed")
st.markdown('<h1 class="title-gradient">IRACify ⚖️</h1>', unsafe_allow_html=True)
st.caption("AI-IRAC-samenvatter voor Nederlandse arresten • IRAC + Essentie + R.O.'s + Quiz • verborgen instellingen")

cfg = get_settings()
model, topk, temp, quiz_q, dev_mode = cfg["model"], cfg["topk"], cfg["temp"], cfg["quiz_q"], cfg["dev_mode"]

tabs = st.tabs(["Tekst", "URL", "Upload", "Quiz"])

# ---------- TAB: Tekst ----------
with tabs[0]:
    _inject_theme_css()
    st.subheader("Plak de zaaktekst")
    tekst = st.text_area("Volledige tekst van het arrest", height=260, label_visibility="collapsed")
    if st.button("Samenvatten (Tekst)"):
        if not tekst.strip():
            st.warning("Voer eerst tekst in.")
        else:
            with st.spinner("Samenvatten…"):
                try:
                    irac = summarize_case_irac(tekst=tekst, model=model, temperature=float(temp), max_ro_in_prompt=int(topk))
                    if "Essentie" not in irac or not isinstance(irac["Essentie"], dict):
                        try: irac["Essentie"] = summarize_case_essentie(tekst, model=model, temperature=float(temp))
                        except Exception: pass
                    show = clean_irac_for_display(irac)
                    st.session_state.irac = show
                    render_irac(show); render_ros(show); render_bronnen(show)
                    st.download_button("Download iracify_output.json",
                        data=json.dumps(show, ensure_ascii=False, indent=2).encode("utf-8"),
                        file_name="iracify_output.json", mime="application/json")
                    if dev_mode:
                        with st.expander("Raw JSON (debug)"): st.json(irac)
                except LLMError as e:
                    st.error(f"Samenvatten faalde: {e}")
                except Exception as e:
                    st.error(str(e))

# ---------- TAB: URL ----------
with tabs[1]:
    _inject_theme_css()
    st.subheader("Geef een URL naar het arrest")
    url = st.text_input("https:// …", placeholder="Plak hier de link (ECLI-pagina of PDF).")
    if st.button("Samenvatten (URL)"):
        if not url.strip():
            st.warning("Voer eerst een geldige URL in.")
        else:
            with st.spinner("URL ophalen en samenvatten…"):
                try:
                    irac = summarize_from_url(url=url, model=model, temperature=float(temp),
                                              max_ro_in_prompt=int(topk), return_essentie=True)
                    show = clean_irac_for_display(irac)
                    st.session_state.irac = show
                    render_irac(show); render_ros(show); render_bronnen(show)
                    st.download_button("Download iracify_output.json",
                        data=json.dumps(show, ensure_ascii=False, indent=2).encode("utf-8"),
                        file_name="iracify_output.json", mime="application/json")
                    if dev_mode:
                        with st.expander("Raw JSON (debug)"): st.json(irac)
                except LLMError as e:
                    st.error(f"Samenvatten faalde: {e}")
                except Exception as e:
                    st.error(str(e))

# ---------- TAB: Upload ----------
with tabs[2]:
    _inject_theme_css()
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
                        irac = summarize_case_irac(tekst=text, model=model, temperature=float(temp), max_ro_in_prompt=int(topk))
                        if "Essentie" not in irac or not isinstance(irac["Essentie"], dict):
                            try: irac["Essentie"] = summarize_case_essentie(text, model=model, temperature=float(temp))
                            except Exception: pass
                        show = clean_irac_for_display(irac)
                        st.session_state.irac = show
                        render_irac(show); render_ros(show); render_bronnen(show)
                        st.download_button("Download iracify_output.json",
                            data=json.dumps(show, ensure_ascii=False, indent=2).encode("utf-8"),
                            file_name="iracify_output.json", mime="application/json")
                        if dev_mode:
                            with st.expander("Raw JSON (debug)"): st.json(irac)
                except LLMError as e:
                    st.error(f"Samenvatten faalde: {e}")
                except Exception as e:
                    st.error(str(e))

# ---------- TAB: Quiz ----------
with tabs[3]:
    _inject_theme_css()
    st.subheader("Quiz op basis van jouw arrest")
    if "irac" not in st.session_state:
        st.info("👋 Maak eerst een samenvatting in één van de andere tabs.")
    else:
        irac = st.session_state.irac
        st.markdown('<div class="card card-quiz">', unsafe_allow_html=True)
        st.markdown("<h3>Quizgenerator</h3>", unsafe_allow_html=True)

        if st.button("Genereer quizvragen"):
            try:
                with st.spinner("Quiz genereren…"):
                    quiz_data = generate_quiz_from_irac(irac, model=model, n_questions=int(quiz_q))
                    st.session_state.quiz = quiz_data
                    st.session_state.quiz_answers = {}
                    st.success("Quiz klaar! Scroll naar beneden om te beantwoorden.")
            except Exception as e:
                st.error(str(e))

        quiz = st.session_state.get("quiz", {})
        questions: List[Dict[str, Any]] = quiz.get("quiz", []) if isinstance(quiz, dict) else []
        if questions:
            st.markdown("---")
            st.markdown("**Vragen**")
            for i, q in enumerate(questions, start=1):
                st.markdown(f"**{i}. {q.get('question','')}**")
                choices = q.get("choices", {})
                key = f"q_{i}"
                current = st.session_state.get("quiz_answers", {}).get(key, None)
                choice = st.radio(
                    "Kies een antwoord",
                    options=["A","B","C","D"],
                    format_func=lambda x: f"{x}) {choices.get(x,'')}",
                    key=key,
                    index=["A","B","C","D"].index(current) if current in ["A","B","C","D"] else 0,
                )
                st.session_state.quiz_answers = st.session_state.get("quiz_answers", {})
                st.session_state.quiz_answers[key] = choice
                st.markdown("")

            if st.button("Inleveren en nakijken"):
                answers = st.session_state.get("quiz_answers", {})
                correct = 0
                st.markdown("---")
                st.markdown("### Uitslag")
                for i, q in enumerate(questions, start=1):
                    key = f"q_{i}"
                    user = answers.get(key, "A")
                    corr = q.get("correct", "A")
                    ok = (user == corr)
                    css = "correct" if ok else "wrong"
                    st.markdown(
                        f'<div class="{css}"><b>Vraag {i}:</b> {"✅ Goed" if ok else "❌ Fout"} '
                        f'(jouw antwoord: {user}; juist: {corr})</div>', unsafe_allow_html=True)
                    expl = q.get("explanation","")
                    if expl: st.markdown(f"*Uitleg:* {expl}")
                    refs = q.get("ro_refs", [])
                    if refs: st.markdown(f"*R.O.-verwijzing(en):* {', '.join(refs)}")
                    st.markdown("")
                    if ok: correct += 1
                st.success(f"Score: {correct} / {len(questions)}")
        st.markdown("</div>", unsafe_allow_html=True)
