# test_iracify_run.py
# Doel: deterministische testrun voor ai_irac_summarizer_v3.py
# - Mockt de LLM-calls -> altijd dezelfde output
# - Test dat specifieke r.o.-nummers (bv. 3.3) worden gekozen
# - Schrijft resultaat naar irac_test_output.json

import json
from ai_irac_summarizer_v3 import (
    summarize_case_irac,
    summarize_case_essentie,
    summarize_from_url,   # niet gebruikt in deze test, maar aanwezig
    segment_rechtsoverwegingen,
    rank_ro_blocks,
    extract_eclis,
)

# ----------------- Demo-arrest (semi-verzonnen, lijkt op HR-lijn) -----------------
DEMO_TEXT = """
ECLI:NL:HR:2023:1234
De Hoge Raad overweegt als volgt.
r.o. 3 In cassatie staat centraal of het weigeren van bepaalde bewijsstukken het recht op een eerlijk proces (art. 6 EVRM) schendt.
r.o. 3.1 Het toetsingskader houdt in dat de rechter een kenbare belangenafweging moet maken en die toereikend motiveert; art. 6 EVRM is leidend.
r.o. 3.2 Het hof sloot de stukken uit ter bevordering van de procesorde, zonder te verduidelijken waarom dit zwaarder woog dan het verdedigingsbelang.
r.o. 3.3 De Hoge Raad oordeelt dat het hof ontoereikend heeft gemotiveerd; de rechtsregel vergt een concrete afweging en kenbare motivering.
r.o. 4 Het middel slaagt; de bestreden uitspraak wordt vernietigd en de zaak teruggewezen.
"""

# ----------------- Mocks voor LLM-calls (deterministisch) -----------------
def _mock_call_llm_irac_with_ranking(tekst, ranked_blocks, eclis, **kwargs):
    # Kies eerste plausibele blokken voor Rule/Application/Conclusion
    rule_idx = app_idx = concl_idx = None
    for i, (num, content) in enumerate(ranked_blocks):
        low = content.lower()
        if rule_idx is None and any(k in low for k in ["maatstaf", "rechtsregel", "toetsingskader", "art."]):
            rule_idx = i
        if app_idx is None and any(k in low for k in ["in casu", "in dit geval", "past toe", "het hof", "de rechtbank"]):
            app_idx = i
        if concl_idx is None and any(k in low for k in ["oordeelt", "vernietigt", "verwerpt", "gegrond", "ongegrond", "middel slaagt"]):
            concl_idx = i
    def _safe(i): return 0 if i is None else i
    rule_idx, app_idx, concl_idx = _safe(rule_idx), _safe(app_idx), _safe(concl_idx)

    def ro_obj(num, content, rol):
        quote = (content.split(".")[0] + ".") if "." in content else content[:120]
        cits = []
        if "art." in content.lower():
            cits.append("art. 6 EVRM")
        return {
            "ro_nummer": num,
            "rol": rol,
            "quote": quote[:280],
            "inhoud": content[:400],
            "citaten": cits,
        }

    ros = []
    for role, idx in [("Rule", rule_idx), ("Application", app_idx), ("Conclusion", concl_idx)]:
        num, content = ranked_blocks[idx]
        ros.append(ro_obj(num, content, role))

    # Voeg nog één Overig toe als voorbeeld
    for i, (num, content) in enumerate(ranked_blocks):
        if i not in {rule_idx, app_idx, concl_idx}:
            ros.append(ro_obj(num, content, "Overig"))
            break

    return {
        "Issue": "Centraal staat of het uitsluiten van stukken de verdedigingsrechten (art. 6 EVRM) schendt en welk toetsingskader geldt.",
        "Rule": "De rechter past het toetsingskader toe (o.a. art. 6 EVRM) en maakt een kenbare belangenafweging die toereikend wordt gemotiveerd.",
        "Application": "In deze zaak is beoordeeld of procesorde zwaarder weegt dan verdedigingsbelang, met weging van concrete omstandigheden.",
        "Conclusion": "Het middel slaagt wegens ontoereikende motivering; de uitspraak wordt vernietigd/teruggewezen.",
        "Rechtsoverwegingen": ros,
        "Bronnen": list(eclis),
    }

def _mock_summarize_case_essentie(tekst, **kwargs):
    return {
        "Essentie": "Kern: art. 6 EVRM vereist een kenbare belangenafweging; het hof motiveerde ontoereikend, waardoor de uitspraak wordt vernietigd.",
        "Kernpunten": [
            "Toetsingskader: art. 6 EVRM.",
            "Hof sloot stukken uit voor procesorde.",
            "Belangenafweging onvoldoende gemotiveerd.",
            "Gevolg: middel slaagt, vernietiging/terugwijzing."
        ]
    }

# ----------------- Patchen: we monkeypatchen de functies in het geïmporteerde module-object -----------------
import ai_irac_summarizer_v3 as mod
mod.call_llm_irac_with_ranking = _mock_call_llm_irac_with_ranking
mod.summarize_case_essentie = _mock_summarize_case_essentie

# ----------------- Pipeline testrun -----------------
def main():
    # 1) Laat zien dat segmentatie en ranking specifieke nummers opleveren
    blocks = mod.segment_rechtsoverwegingen(DEMO_TEXT)
    ranked = mod.rank_ro_blocks(blocks, DEMO_TEXT, topk=12)
    eclis = mod.extract_eclis(DEMO_TEXT)

    print("Gedetecteerde ECLI's:", eclis)
    print("Top r.o.-kandidaten (geordend):", [n for n,_ in ranked])

    # Expectation: 3.1/3.2/3.3 verschijnen, en 3 (parent) krijgt geen voorkeur
    assert any(n.startswith("3.3") for n,_ in ranked), "Specifieke r.o. 3.3 ontbreekt in ranking"
    assert "3" in [n for n,_ in blocks], "Parent r.o. 3 is niet gedetecteerd (segmentatie-check)"

    # 2) Draai IRAC (met mock LLM) en verrijk met Essentie
    irac = mod.summarize_case_irac(DEMO_TEXT)
    irac["Essentie"] = mod.summarize_case_essentie(DEMO_TEXT)

    # 3) Checks op vorm en inhoud
    roles = [ro["rol"] for ro in irac.get("Rechtsoverwegingen", [])]
    assert {"Rule", "Application", "Conclusion"}.issubset(set(roles)), "Niet alle verplichte rollen aanwezig"
    assert any(ro["ro_nummer"].startswith("3.3") for ro in irac["Rechtsoverwegingen"]), "Geen r.o. met specifiek nummer (3.3) in output"

    # 4) Schrijf JSON-output
    with open("irac_test_output.json", "w", encoding="utf-8") as f:
        json.dump(irac, f, ensure_ascii=False, indent=2)

    print("\nOK ✅  — Testrun voltooid. Output geschreven naar irac_test_output.json")
    print("Samenvatting (Essentie):", irac["Essentie"]["Essentie"])
    print("\nGekozen R.O.'s (nummer → rol):")
    for ro in irac["Rechtsoverwegingen"]:
        print(f"  - {ro['ro_nummer']}: {ro['rol']}")

if __name__ == "__main__":
    main()

