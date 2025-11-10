# ⚖️ IRACify – AI Juridische Samenvatter

**IRACify** is een AI-tool die Nederlandse arresten automatisch samenvat in het klassieke **IRAC-formaat**  
(**Issue, Rule, Application, Conclusion**), inclusief rechtsoverwegingen (r.o.’s), bronnen en kernpunten.

🚀 Probeer het zelf:
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://iracify.streamlit.app)

---

## ✨ Functies

✅ **Automatische IRAC-samenvatting**  
– Herkent relevante r.o.’s en benoemt hun rol (Rule, Application, Conclusion)  
– Geeft kernachtige Issue, Rule, Application en Conclusion in juridisch Nederlands  

🧩 **Essentie & kernpunten**  
– Compacte samenvatting van de zaak in ±120 woorden  
– 3–5 bullet-highlights met concrete feiten en beslissingen  

📚 **Quiz-modus**  
– Genereert 4–5 multiple-choice vragen over het arrest  
– Toont uitleg en verwijzing naar r.o.’s  

📁 **Upload-, URL- en tekstinvoer**  
– Verwerk arresten via URL, PDF of handmatige tekstinvoer  

🎨 **Kleurrijke interface**  
– Duidelijke badges voor r.o.-rollen  
– Moderne layout zonder instellingenbalk voor gebruikers  

🔒 **Veilige adminmodus**  
– Alleen zichtbaar via `?admin=SECRETTOKEN`  
– Sidebar met model- en quizinstellingen  

---

## 🧠 Technologie

- **Frontend:** Streamlit (Python)  
- **Backend:** OpenAI GPT-4o-mini API  
- **Parser:** Python + regex + JSON Schema validatie  
- **Extractie:** `requests`, `pdfminer.six`, `trafilatura`

---

## 🧰 Installatie (lokaal)

```bash
git clone https://github.com/<jouw-gebruikersnaam>/iracify.git
cd iracify
pip install -r requirements.txt
streamlit run streamlit_app.py
