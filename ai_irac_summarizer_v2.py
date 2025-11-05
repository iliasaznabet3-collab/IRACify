"""
ai_irac_summarizer_v2.py
Verbeterde NL R.O.-extractie + IRAC:
- Deterministische segmentatie van r.o.-blokken (r.o./rov. + nummer)
- Heuristische ranking (juridische kernwoorden, wetsverwijzingen, lengte)
- LLM re-rank + labeling (Rule | Application | Conclusion | Overig)
- Strikt JSON-schema met quote + samenvatting + citaten
- Exponential backoff & timeouts

Vereist:
- pip install -U "openai>=1.42.0"
- OPENAI_API_KEY als environment variable (of geef api_key=... mee)
"""

import re
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from openai import OpenAI

# -------------------- Config --------------------

RO_MAX_CHARS = 1600   # max. tekens per r.o.-fragment dat naar het model gaat
TOPK_RO = 12          # aantal r.o.-kandidaten dat we aan het model tonen (meer detail)

SYSTEM_PROMPT = (
    "Je bent een Nederlandse juridisch redacteur. "
    "Vat arresten samen in IRAC (Issue, Rule, Application, Conclusion) en benoem expliciet de relevante rechtsoverwegingen (R.O.'s). "
    "Gebruik juridisch correcte NL-terminologie. Wees kort, precies en feitelijk."
)

IRAC_SCHEMA = {
    "name": "irac_schema",
    "schema": {
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
                        "rol": {"type": "string", "description": "Rule|Application|Conclusion|Overig"},
                        "quote": {"type": "string", "description": "Korte letterlijke quote (1–2 zinnen)"},
                        "inhoud": {"type": "string", "description": "Samenvatting met concrete details (max 3 zinnen)"},
                        "citaten": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Wets/HR-verwijzingen, bv. 'art. 6 EVRM', 'art. 359a Sv'"
                        }
                    },
                    # strict=True -> alle properties moeten ook required zijn
                    "required": ["ro_nummer", "rol", "quote", "inhoud", "citaten"],
                    "additionalProperties": False
                }
            },
            "Bronnen": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        # strict=True -> alle top-level properties ook required
        "required": ["Issue", "Rule", "Application", "Conclusion", "Rechtsoverwegingen", "Bronnen"],
        "additionalProperties": False
    },
    "strict": True,
}

# -------------------- Regexen --------------------

ECLI_RE = re.compile(r"\bECLI:[A-Z]{2}:[A-Z]{2,}:\d{4}:[A-Z0-9]+(?:-\d+)?\b", re.IGNORECASE)

# Matcht r.o./rov. met 1-4 niveau’s, bv. 3, 3.1, 3.4.1, etc.
RO_HEADER_RE = re.compile(r"\b(?:r\.?o\.?|rov\.)\s*(\d+(?:\.\d+){0,3})\b", re.IGNORECASE)

# -------------------- Deterministische parser --------------------

def extract_eclis(text: str) -> List[str]:
    return list(dict.fromkeys(m.group(0).upper() for m in ECLI_RE.finditer(text)))

def segment_rechtsoverwegingen(text: str) -> List[Tuple[str, str]]:
    """
    Vind r.o.-headers en segmenteer de tekst tot de volgende header.
    Return: list[(nummer, inhoud)]
    """
    matches = list(RO_HEADER_RE.finditer(text))
    blocks: List[Tuple[str, str]] = []
    if not matches:
        return blocks
    for i, m in enumerate(matches):
        num = m.group(1)
        start = m.end()
        end = matches[i+1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        content = re.sub(r"^\s*[:\-\—]\s*", "", content)  # haal eventuele dubbele punctuatie weg
        blocks.append((num, content))
    return blocks

# -------------------- Heuristische scoring --------------------

KEYWORDS_STRONG = [
    "rechtsregel", "toetsingskader", "maatstaf", "oordeelt", "overweegt",
    "schending", "niet-ontvankelijk", "verwerpt", "gegrond", "ongegrond",
    "cassatie", "sluit aan bij", "art.", "artikel", "evrm", "bw", "sr", "sv",
    "proportionaliteit", "subsidiariteit", "motiveringsgebrek", "belangenafweging", "kwalificatie"
]

def score_block(num: str, content: str, has_ecli: bool) -> float:
    text = f"{num} {content}".lower()
    score = 0.0
    # kernwoorden
    for kw in KEYWORDS_STRONG:
        if kw in text:
            score += 2.0
    # citaten/wetsverwijzingen
    if re.search(r"\bart\.\s*\d+", text):
        score += 1.5
    if "evrm" in text:
        score += 1.0
    # lengte-heuriek
    n = len(content)
    if 250 <= n <= 1200:
        score += 1.2
    elif n > 1800:
        score -= 0.6
    elif n < 120:
        score -= 0.4
    # aanwezigheid ECLI in geheel (zwak signaal)
    if has_ecli:
        score += 0.3
    return score

def rank_ro_blocks(blocks: List[Tuple[str, str]], text: str, topk: int = TOPK_RO) -> List[Tuple[str, str]]:
    has_ecli = bool(extract_eclis(text))
    scored = [(num, content, score_block(num, content, has_ecli)) for num, content in blocks]
    scored.sort(key=lambda x: x[2], reverse=True)
    return [(num, content) for num, content, _ in scored[:topk]]

# -------------------- OpenAI helper --------------------

def _client(api_key: Optional[str] = None) -> OpenAI:
    return OpenAI(api_key=api_key) if api_key else OpenAI()

def call_llm_irac_with_ranking(
    tekst: str,
    ranked_blocks: List[Tuple[str, str]],
    eclis: List[str],
    model: str,
    temperature: float,
    timeout_s: int,
    max_retries: int,
    api_key: Optional[str]
) -> Dict[str, Any]:
    """
    Geeft IRAC + geselecteerde R.O.'s met rol terug (constrained JSON).
    LLM krijgt alléén topk-blokken om te kiezen.
    """
    def clamp(s: str, n: int) -> str:
        return s if len(s) <= n else s[:n] + "…"

    ro_list = "\n\n".join([f"[{num}]\n{clamp(content.strip(), RO_MAX_CHARS)}" for num, content in ranked_blocks])

    hints = []
    if eclis:
        hints.append(f"ECLI(‘s): {', '.join(eclis)}")
    hints_text = (" | " + " | ".join(hints)) if hints else ""

    user_prompt = (
        "Je krijgt kandidaat-fragmenten uit rechtsoverwegingen (r.o.). "
        "Selecteer uitsluitend de overwegingen die dragend zijn voor rechtsregel en uitkomst. "
        "Voor elke gekozen r.o.:\n"
        " - Geef een KORTE LETTERLIJKE QUOTE (1–2 zinnen) uit het fragment (tussen aanhalingstekens),\n"
        " - Geef daarna een SAMENVATTING met concrete details (max 3 zinnen),\n"
        " - Label de rol: Rule | Application | Conclusion | Overig,\n"
        " - Noteer expliciet wetsverwijzingen/HR-verwijzingen als 'citaten' (bv. 'art. 6 EVRM', 'art. 359a Sv').\n"
        "Gebruik géén algemeenheden; benoem specifieke afwegingen, toetsingskaders en belangen. "
        "Zorg dat er minimaal één Rule, één Application en één Conclusion is. "
        "Baseer je uitsluitend op de aangeleverde fragmenten; verzin geen r.o.-nummers.\n\n"
        f"Context{hints_text}\n\n"
        "Beknopte zaaktekst (voor IRAC-scheidingskader):\n"
        f"{tekst[:3000].strip()}\n\n"
        "Kandidaat r.o.-fragmenten:\n"
        f"{ro_list}\n\n"
        "Lever uitsluitend JSON volgens het schema; geen extra tekst."
    )

    client = _client(api_key)
    backoff = 2
    last = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": (
                        "Je bent een NL juridisch redacteur. Schrijf precies, citeer kort en benoem bepalingen/HR."
                    )},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=min(temperature, 0.2),  # houd het feitelijk
                timeout=timeout_s,
                response_format={"type": "json_schema", "json_schema": IRAC_SCHEMA},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            last = e
            if attempt == max_retries:
                raise
            time.sleep(min(20, backoff))
            backoff *= 2
    raise RuntimeError(f"LLM-call faalde na {max_retries} pogingen: {last}")

# -------------------- Publieke API --------------------

def summarize_case_irac(
    tekst: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.1,
    timeout_s: int = 60,
    max_retries: int = 5,
    api_key: Optional[str] = None,
    max_ro_in_prompt: int = TOPK_RO
) -> Dict[str, Any]:
    """
    Pipeline:
      1) Segmenteer r.o.-blokken
      2) Heuristisch ranken en top-k selecteren
      3) LLM re-rank & IRAC synthese met strikt schema
    """
    eclis = extract_eclis(tekst)
    blocks = segment_rechtsoverwegingen(tekst)
    if blocks:
        top_blocks = rank_ro_blocks(blocks, tekst, topk=max_ro_in_prompt)
    else:
        top_blocks = []  # geen expliciete r.o.-nummering

    data = call_llm_irac_with_ranking(
        tekst=tekst,
        ranked_blocks=top_blocks,
        eclis=eclis,
        model=model,
        temperature=temperature,
        timeout_s=timeout_s,
        max_retries=max_retries,
        api_key=api_key,
    )

    # Voeg gedetecteerde ECLI's toe aan Bronnen (non-destructief)
    bronnen = data.get("Bronnen", [])
    for e in eclis:
        if e not in bronnen:
            bronnen.append(e)
    data["Bronnen"] = bronnen
    return data

# -------------------- CLI test --------------------

def pretty_print_irac(data: Dict[str, Any]) -> None:
    def sect(t): print(f"\n=== {t} ===")
    sect("ISSUE"); print(data.get("Issue",""))
    sect("RULE"); print(data.get("Rule",""))
    sect("APPLICATION"); print(data.get("Application",""))
    sect("CONCLUSION"); print(data.get("Conclusion",""))
    sect("RECHTSOVERWEGINGEN")
    for ro in data.get("Rechtsoverwegingen", []):
        rn = ro.get('ro_nummer','')
        rol = ro.get('rol','')
        quote = ro.get('quote','')
        inhoud = ro.get('inhoud','')
        cits = ro.get('citaten', [])
        print(f"- {rn} [{rol}]")
        if quote:
            print(f'  "{quote}"')
        if inhoud:
            print(f"  {inhoud}")
        if cits:
            print(f"  Citaten: {', '.join(cits)}")
    br = data.get("Bronnen", [])
    if br:
        sect("BRONNEN")
        for b in br:
            print(f"- {b}")

if __name__ == "__main__":
    # Demo met kunstmatige r.o.'s — vervang door echte arresttekst voor real-world test
    text = """
    ECLI:NL:HR:2022:9999
    De Hoge Raad overweegt als volgt. r.o. 3 In cassatie staat centraal of het weigeren van bepaalde bewijsstukken het recht op een eerlijk proces schendt.
    r.o. 3.1 De maatstaf volgt uit art. 6 EVRM: de verdachte moet in staat zijn relevante stukken in te brengen.
    r.o. 3.2 Het hof sloot de stukken uit ter bevordering van de procesorde, maar motiveerde niet waarom dit zwaarder woog dan het verdedigingsbelang.
    r.o. 3.3 De Hoge Raad oordeelt dat het hof ontoereikend heeft gemotiveerd; de rechtsregel vergt een kenbare belangenafweging.
    r.o. 4 Het middel slaagt; de bestreden uitspraak wordt vernietigd en de zaak wordt teruggewezen.
    """
    result = summarize_case_irac(text)
    pretty_print_irac(result)

