"""
IRACify – ai_irac_summarizer_v3.py

Verbeteringen:
- Robuustere R.O.-segmentatie (r.o./rov. + genummerde koppen)
- Normalisatie en zachte truncatie per fragment
- Deterministische ranking met voorkeur voor specifieke nummers (3.3 > 3)
- Essentie-samenvatting (korte kern + bullets)
- URL-samenvatting (kopieer link → IRAC + Essentie)
- JSON-schema validatie, backoff met jitter, rol-minimumgarantie
- Guardrails tegen verzonnen r.o.-nummers
- f-string bugfix ✅
- OpenAI Chat Completions + fallback voor oudere SDK’s (zonder response_format)

Vereist:
  pip install -U "openai>=1.42.0" jsonschema requests trafilatura pdfminer.six
  export OPENAI_API_KEY=...
"""

from __future__ import annotations
import re
import json
import time
import random
import logging
from typing import Dict, Any, List, Optional, Tuple

from jsonschema import Draft202012Validator, ValidationError
from openai import OpenAI

# Optionele libs voor URL-extractie
try:
    import requests  # type: ignore
except Exception:
    requests = None

try:
    import trafilatura  # type: ignore
except Exception:
    trafilatura = None

# -------------------- Config --------------------
RO_MAX_CHARS = 1600
TOPK_RO = 12
MODEL_DEFAULT = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "Je bent een Nederlandse juridisch redacteur. "
    "Vat arresten samen in IRAC (Issue, Rule, Application, Conclusion) "
    "en benoem expliciet de relevante rechtsoverwegingen (R.O.'s). "
    "Gebruik juridisch correcte NL-terminologie. Wees kort, precies en feitelijk."
)

IRAC_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "Issue": {"type": "string"},
        "Rule": {"type": "string"},
        "Application": {"type": "string"},
        "Conclusion": {"type": "string"},
        "Rechtsoverwegingen": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ro_nummer": {"type": "string"},
                    "rol": {"type": "string", "enum": ["Rule", "Application", "Conclusion", "Overig"]},
                    "quote": {"type": "string"},
                    "inhoud": {"type": "string"},
                    "citaten": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["ro_nummer", "rol", "quote", "inhoud", "citaten"],
                "additionalProperties": False
            }
        },
        "Bronnen": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["Issue", "Rule", "Application", "Conclusion", "Rechtsoverwegingen", "Bronnen"],
    "additionalProperties": False
}

ESSENTIE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "Essentie": {"type": "string", "description": "Max ~120 woorden, in helder NL."},
        "Kernpunten": {"type": "array", "items": {"type": "string"}, "description": "3–5 bullets met feitelijke highlights."}
    },
    "required": ["Essentie", "Kernpunten"],
    "additionalProperties": False
}

# -------------------- Regex --------------------
ECLI_RE = re.compile(r"\bECLI:[A-Z]{2}:[A-Z]{2,}:\d{4}:[A-Z0-9]+(?:-\d+)?\b", re.IGNORECASE)
RO_HEADER_RE = re.compile(r"\b(?:r\.?o\.?|rov\.)\s*(\d+(?:\.\d+){0,3})\b", re.IGNORECASE)
NUM_HEADER_RE = re.compile(r"^(\d+(?:\.\d+){0,3})(?=[\s:—\-])", re.MULTILINE)

# -------------------- Helpers --------------------
def normalize_text(text: str) -> str:
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[\t\u00A0]+", " ", t)
    t = re.sub(r"\u201C|\u201D", '"', t)
    t = re.sub(r"\u2018|\u2019", "'", t)
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def clamp(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "…"

# -------------------- Parsing --------------------
def extract_eclis(text: str) -> List[str]:
    return list(dict.fromkeys(m.group(0).upper() for m in ECLI_RE.finditer(text)))

def _collect_headers(text: str) -> List[Tuple[str, int, int]]:
    headers = [(m.group(1), m.start(), m.end()) for m in RO_HEADER_RE.finditer(text)]
    if not headers:
        for m in NUM_HEADER_RE.finditer(text):
            num = m.group(1)
            try:
                first = int(num.split('.')[0])
            except ValueError:
                first = 0
            if 0 < first <= 50:
                headers.append((num, m.start(), m.end()))
    headers.sort(key=lambda x: x[1])
    return headers

def segment_rechtsoverwegingen(text: str) -> List[Tuple[str, str]]:
    t = normalize_text(text)
    headers = _collect_headers(t)
    blocks: List[Tuple[str, str]] = []
    if not headers:
        return blocks
    for i, (num, _start, end_hdr) in enumerate(headers):
        next_start = headers[i + 1][1] if i + 1 < len(headers) else len(t)
        content = t[end_hdr:next_start].strip()
        content = re.sub(r"^\s*[:\-—]\s*", "", content)
        content = re.sub(rf"^(?:{re.escape(num)}\.?\s*)+", "", content)
        if content:
            blocks.append((num, content))
    return blocks

# -------------------- Scoring --------------------
KEYWORDS_STRONG = [
    "rechtsregel", "toetsingskader", "maatstaf", "oordeelt", "overweegt",
    "schending", "niet-ontvankelijk", "verwerpt", "gegrond", "ongegrond",
    "cassatie", "sluit aan bij", "art.", "artikel", "evrm", "bw", "sr", "sv",
    "proportionaliteit", "subsidiariteit", "motiveringsgebrek", "belangenafweging", "kwalificatie"
]

def score_block(num: str, content: str, has_ecli: bool) -> float:
    text = f"{num} {content}".lower()
    score = 0.0
    for kw in KEYWORDS_STRONG:
        if kw in text:
            score += 2.0
    if re.search(r"\bart\.\s*\d+", text):
        score += 1.5
    if "evrm" in text:
        score += 1.0
    n = len(content)
    if 250 <= n <= 1200:
        score += 1.2
    elif n > 1800:
        score -= 0.6
    elif n < 120:
        score -= 0.4
    if has_ecli:
        score += 0.3
    # Sterkere bonus voor diepere nummers (3.1.2 > 3)
    depth = num.count('.')
    score += min(0.9, 0.3 * depth)
    return score

def _has_children(num: str, all_nums: List[str]) -> bool:
    prefix = num + "."
    return any(n.startswith(prefix) for n in all_nums)

def rank_ro_blocks(blocks: List[Tuple[str, str]], text: str, topk: int = TOPK_RO) -> List[Tuple[str, str]]:
    has_ecli = bool(extract_eclis(text))
    all_nums = [num for num, _ in blocks]
    MIN_PARENT_LEN = 220  # korte parent met kinderen -> liever kinderen tonen

    filtered: List[Tuple[str, str]] = []
    for num, content in blocks:
        if _has_children(num, all_nums) and len(content) < MIN_PARENT_LEN:
            continue
        filtered.append((num, content))

    scored = [(num, content, score_block(num, content, has_ecli), len(content)) for num, content in filtered]
    scored.sort(key=lambda x: (-x[2], -x[3], x[0]))
    top: List[Tuple[str, str]] = []
    seen = set()
    for num, content, _s, _l in scored:
        if num in seen:
            continue
        seen.add(num)
        top.append((num, clamp(content.strip(), RO_MAX_CHARS)))
        if len(top) >= topk:
            break
    return top

# -------------------- OpenAI helpers --------------------
class LLMError(RuntimeError):
    pass

def _client(api_key: Optional[str] = None) -> OpenAI:
    return OpenAI(api_key=api_key) if api_key else OpenAI()

def _json_schema_for_api() -> Dict[str, Any]:
    # Voor Chat Completions met json_schema (nieuwere SDK’s). Oudere SDK’s vallen terug op prompt-only.
    return {
        "type": "json_schema",
        "json_schema": {"name": "irac_schema", "schema": IRAC_SCHEMA, "strict": True},
    }

# -------------------- Postprocessing & Validatie --------------------
def _validate_irac(data: Dict[str, Any]) -> None:
    try:
        Draft202012Validator(IRAC_SCHEMA).validate(data)
    except ValidationError as ve:
        raise LLMError(f"JSON schema-validatie faalde: {ve.message}")

def _enforce_min_roles(data: Dict[str, Any]) -> None:
    roles = [ro.get("rol") for ro in data.get("Rechtsoverwegingen", [])]
    needed = {"Rule", "Application", "Conclusion"} - set(roles)
    if not needed:
        return
    for want in list(needed):
        idx = _pick_best_for_role(data.get("Rechtsoverwegingen", []), want)
        if idx is not None:
            data["Rechtsoverwegingen"][idx]["rol"] = want

def _pick_best_for_role(items: List[Dict[str, Any]], want: str) -> Optional[int]:
    want_kw = {
        "Rule": ["rechtsregel", "maatstaf", "toetsingskader", "volgt uit", "heeft te gelden"],
        "Application": ["toegepast", "in casu", "in dit geval", "het hof", "de rechtbank", "past toe"],
        "Conclusion": ["concludeert", "oordeelt", "veroordeelt", "vernietigt", "verwerpt", "gegrond", "ongegrond", "beslist"],
    }.get(want, [])
    best_i, best_score = None, 0
    for i, ro in enumerate(items):
        if ro.get("rol") == "Overig":
            text = (ro.get("inhoud", "") + " " + ro.get("quote", "")).lower()
            score = sum(1 for kw in want_kw if kw in text)
            if score > best_score:
                best_i, best_score = i, score
    return best_i

def _deny_unknown_ro_nums(data: Dict[str, Any], allowed_nums: List[str]) -> None:
    allowed = set(allowed_nums)
    for ro in data.get("Rechtsoverwegingen", []):
        if allowed and ro.get("ro_nummer") not in allowed:
            ro["rol"] = "Overig"

# -------------------- LLM-call (Chat Completions + fallback) --------------------
def call_llm_irac_with_ranking(
    tekst: str,
    ranked_blocks: List[Tuple[str, str]],
    eclis: List[str],
    model: str,
    temperature: float,
    timeout_s: int,
    max_retries: int,
    api_key: Optional[str],
) -> Dict[str, Any]:
    ro_list = "\n\n".join([f"[{num}]\n{content}" for num, content in ranked_blocks])
    hints_text = (" | ECLI’s: " + ", ".join(eclis)) if eclis else ""

    user_prompt = (
        "Je krijgt kandidaat-fragmenten uit rechtsoverwegingen (r.o.). "
        "Selecteer uitsluitend de overwegingen die dragend zijn voor rechtsregel en uitkomst. "
        "Voor elke gekozen r.o.:\n"
        " - Geef een KORTE LETTERLIJKE QUOTE (1–2 zinnen),\n"
        " - Geef daarna een SAMENVATTING met concrete details (max 3 zinnen),\n"
        " - Label de rol: Rule | Application | Conclusion | Overig,\n"
        " - Noteer expliciet wetsverwijzingen/HR-verwijzingen.\n"
        "Gebruik géén algemeenheden; benoem specifieke afwegingen, toetsingskaders en belangen.\n"
        "Zorg dat er minimaal één Rule, één Application en één Conclusion is.\n"
        "Baseer je uitsluitend op de aangeleverde fragmenten; verzin geen r.o.-nummers.\n\n"
        f"Context{hints_text}\n\n"
        "Beknopte zaaktekst:\n"
        f"{clamp(tekst.strip(), 3000)}\n\n"
        f"Kandidaat r.o.-fragmenten:\n{ro_list}\n\n"
        "Gebruik altijd het meest specifieke r.o.-nummer dat beschikbaar is (bijv. 3.3 i.p.v. 3).\n"
        "Lever uitsluitend JSON volgens het schema; geen extra tekst."
    )

    client = _client(api_key)
    backoff = 2.0
    last_err: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            try:
                # Nieuwere SDK: met response_format (json_schema)
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=min(temperature, 0.2),
                    timeout=timeout_s,
                    response_format=_json_schema_for_api(),
                )
                content = resp.choices[0].message.content
            except TypeError:
                # Oudere SDK: zonder response_format (we vragen JSON via prompt)
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=min(temperature, 0.2),
                    timeout=timeout_s,
                )
                content = resp.choices[0].message.content

            data = json.loads(content)
            _validate_irac(data)
            _enforce_min_roles(data)
            _deny_unknown_ro_nums(data, [n for n, _ in ranked_blocks])
            return data

        except Exception as e:
            last_err = e
            if attempt == max_retries:
                raise LLMError(f"LLM-call faalde: {e}")
            time.sleep(min(20.0, backoff + random.uniform(0, 0.75)))
            backoff *= 2.0

    raise LLMError(f"Onbekende fout: {last_err}")

# -------------------- Essentie --------------------
def summarize_case_essentie(
    tekst: str,
    model: str = MODEL_DEFAULT,
    temperature: float = 0.1,
    timeout_s: int = 45,
    max_retries: int = 4,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Korte, bondige samenvatting gericht op de kern van het arrest.
    Probeert eerst met response_format (json_schema). Bij TypeError valt hij terug op plain JSON.
    """
    tekst_norm = clamp(normalize_text(tekst), 6000)

    user_prompt = (
        "Vat de essentie van het arrest kort en feitelijk samen. "
        "Schrijf in helder Nederlands, zonder retoriek. "
        "Geef eerst een compacte alinea (≤ ~120 woorden) met probleem, rechtsregel en uitkomst; "
        "daarna 3–5 puntsgewijze kernpunten met concrete details (namen, artikelen, beslissingen). "
        "Lever uitsluitend JSON met velden 'Essentie' en 'Kernpunten' conform schema; geen extra tekst."
    )

    client = _client(api_key)
    backoff = 2.0
    last_err: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"{user_prompt}\n\nTEKST:\n{tekst_norm}"},
                    ],
                    temperature=min(temperature, 0.2),
                    timeout=timeout_s,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "essentie_schema", "schema": ESSENTIE_SCHEMA, "strict": True},
                    },
                )
                content = resp.choices[0].message.content
            except TypeError:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"{user_prompt}\n\nTEKST:\n{tekst_norm}"},
                    ],
                    temperature=min(temperature, 0.2),
                    timeout=timeout_s,
                )
                content = resp.choices[0].message.content

            data = json.loads(content)
            Draft202012Validator(ESSENTIE_SCHEMA).validate(data)
            return data

        except Exception as e:
            last_err = e
            if attempt == max_retries:
                raise LLMError(f"LLM essentie-call faalde: {e}")
            time.sleep(min(20.0, backoff + random.uniform(0, 0.5)))
            backoff *= 2.0
    raise LLMError(f"Onbekende fout essentie: {last_err}")

# -------------------- Publieke API --------------------
def summarize_case_irac(
    tekst: str,
    model: str = MODEL_DEFAULT,
    temperature: float = 0.1,
    timeout_s: int = 60,
    max_retries: int = 5,
    api_key: Optional[str] = None,
    max_ro_in_prompt: int = TOPK_RO,
) -> Dict[str, Any]:
    tekst_norm = normalize_text(tekst)
    eclis = extract_eclis(tekst_norm)
    blocks = segment_rechtsoverwegingen(tekst_norm)
    top_blocks = rank_ro_blocks(blocks, tekst_norm, topk=max_ro_in_prompt) if blocks else []
    data = call_llm_irac_with_ranking(
        tekst=tekst_norm,
        ranked_blocks=top_blocks,
        eclis=eclis,
        model=model,
        temperature=temperature,
        timeout_s=timeout_s,
        max_retries=max_retries,
        api_key=api_key,
    )
    bronnen = list(dict.fromkeys(data.get("Bronnen", []) + eclis))
    data["Bronnen"] = bronnen
    return data

# -------------------- URL → IRAC + Essentie --------------------
def _fetch_url_text(url: str, timeout: int = 20) -> Optional[str]:
    if requests is None:
        raise RuntimeError("'requests' is niet geïnstalleerd. pip install requests trafilatura pdfminer.six")

    headers = {"User-Agent": "IRACifyBot/1.0 (+https://iracify.app)"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "").lower()

    # PDF?
    if "application/pdf" in ctype or url.lower().endswith(".pdf"):
        try:
            from pdfminer.high_level import extract_text as pdf_extract_text  # type: ignore
            import io
            return pdf_extract_text(io.BytesIO(resp.content))
        except Exception:
            return None

    text = resp.text
    # HTML extraction met trafilatura indien beschikbaar
    if (("html" in ctype) or ("xml" in ctype) or ("<html" in text.lower())) and trafilatura is not None:
        try:
            extracted = trafilatura.extract(text, include_comments=False, include_tables=False)
            if extracted:
                return extracted
        except Exception:
            logging.debug("Trafilatura extractie faalde; val terug op raw text")
    return text

def summarize_from_url(
    url: str,
    *,
    model: str = MODEL_DEFAULT,
    temperature: float = 0.1,
    timeout_s: int = 60,
    max_retries: int = 5,
    api_key: Optional[str] = None,
    max_ro_in_prompt: int = TOPK_RO,
    return_essentie: bool = True,
) -> Dict[str, Any]:
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("URL moet met http(s) beginnen")

    text = _fetch_url_text(url)
    if not text or len(text.strip()) < 200:
        raise LLMError("Kon onvoldoende tekst uit de URL extraheren")

    irac = summarize_case_irac(
        tekst=text,
        model=model,
        temperature=temperature,
        timeout_s=timeout_s,
        max_retries=max_retries,
        api_key=api_key,
        max_ro_in_prompt=max_ro_in_prompt,
    )

    if return_essentie:
        try:
            irac["Essentie"] = summarize_case_essentie(
                text, model=model, temperature=temperature, timeout_s=timeout_s,
                max_retries=max_retries, api_key=api_key
            )
        except Exception as e:
            logging.warning(f"Essentie-samenvatting faalde (niet fataal): {e}")

    bronnen = irac.get("Bronnen", [])
    if url not in bronnen:
        bronnen.append(url)
    irac["Bronnen"] = bronnen
    return irac

# -------------------- CLI test --------------------
def pretty_print_irac(data: Dict[str, Any]) -> None:
    def sect(t): print(f"\n=== {t} ===")
    sect("ISSUE"); print(data.get("Issue",""))
    sect("RULE"); print(data.get("Rule",""))
    sect("APPLICATION"); print(data.get("Application",""))
    sect("CONCLUSION"); print(data.get("Conclusion",""))
    sect("RECHTSOVERWEGINGEN")
    for ro in data.get("Rechtsoverwegingen", []):
        rn, rol = ro.get('ro_nummer',''), ro.get('rol','')
        quote, inhoud = ro.get('quote',''), ro.get('inhoud','')
        cits = ro.get('citaten', [])
        print(f"- {rn} [{rol}]")
        if quote: print(f'  "{quote}"')
        if inhoud: print(f"  {inhoud}")
        if cits: print(f"  Citaten: {', '.join(cits)}")
    br = data.get("Bronnen", [])
    if br:
        sect("BRONNEN")
        for b in br:
            print(f"- {b}")
    ess = data.get("Essentie")
    if isinstance(ess, dict):
        sect("ESSENTIE")
        print(ess.get("Essentie",""))
        kp = ess.get("Kernpunten", [])
        for k in kp:
            print(f"- {k}")

if __name__ == "__main__":
    demo_text = '''
    ECLI:NL:HR:2022:9999
    De Hoge Raad overweegt als volgt.
    r.o. 3 In cassatie staat centraal of het weigeren van bewijsstukken het recht op een eerlijk proces schendt.
    r.o. 3.1 De maatstaf volgt uit art. 6 EVRM.
    r.o. 3.2 Het hof sloot stukken uit, maar motiveerde niet.
    r.o. 3.3 De Hoge Raad oordeelt dat het hof ontoereikend motiveerde.
    r.o. 4 Het middel slaagt; uitspraak wordt vernietigd.
    '''
    # Voorbeeld IRAC (vereist API key):
    # result = summarize_case_irac(demo_text)
    # pretty_print_irac(result)

    # Voorbeeld URL (vereist requests + optioneel trafilatura/pdfminer.six):
    # print(json.dumps(summarize_from_url("https://example.com/arrest"), indent=2, ensure_ascii=False))
