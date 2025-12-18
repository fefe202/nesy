# Business Intelligence Co-pilot 🚀

Applicazione Streamlit per l'analisi di idee di business usando Google Gemini e RAG (Retrieval-Augmented Generation).

## Prerequisiti

- Python 3.11+
- Account Google Cloud con API Gemini abilitata

## Setup

### 1. Installazione dipendenze

```powershell
pip install -r requirements.txt
```

### 2. Configurazione API Key Google Gemini

#### Opzione A: API Key (consigliata per sviluppo)

1. Ottieni la tua API key da [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Imposta la variabile d'ambiente:

**PowerShell (sessione corrente):**
```powershell
$env:GOOGLE_API_KEY="la-tua-api-key-qui"
```

**PowerShell (permanente - solo utente corrente):**
```powershell
[System.Environment]::SetEnvironmentVariable('GOOGLE_API_KEY', 'la-tua-api-key-qui', 'User')
```

**Windows GUI (permanente):**
1. Cerca "Variabili d'ambiente" nel menu Start
2. Clicca "Modifica variabili d'ambiente di sistema"
3. Clicca "Variabili d'ambiente..."
4. In "Variabili utente", clicca "Nuova..."
5. Nome variabile: `GOOGLE_API_KEY`
6. Valore variabile: la tua API key
7. Clicca OK e riavvia il terminale

#### Opzione B: Service Account (consigliata per produzione)

1. Crea un service account in Google Cloud Console
2. Scarica il file JSON delle credenziali
3. Imposta la variabile d'ambiente:

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\service-account.json"
```

### 3. Prepara i dati CSV

Assicurati che il file `imprese_attive_2021.csv` sia presente nella cartella `LLM_Lab/`.

## Avvio

```powershell
cd d:\nesy\LLM_Lab
streamlit run project_llm_lab.py
```

L'app si aprirà automaticamente nel browser all'indirizzo `http://localhost:8501`

## Modelli utilizzati

- **gemini-1.5-flash**: Per task veloci (estrazione contesto, sentiment analysis)
- **gemini-1.5-pro**: Per task complessi (analisi di mercato, report strategici)

## Funzionalità

1. **Analisi Contesto**: Estrae informazioni chiave dall'idea di business
2. **RAG Database**: Ricerca in database vettoriale di imprese italiane
3. **Web Search**: Ricerca trend aggiornati online
4. **Visualizzazioni**: Grafici interattivi (Pie e Bar chart)
5. **Risk Scoring**: Calcolo rischio qualitativo con matrice
6. **Report SWOT**: Analisi strategica completa con raccomandazioni

## Troubleshooting

### Errore: "GOOGLE_API_KEY non trovata"
- Verifica di aver impostato la variabile d'ambiente
- Riavvia il terminale dopo aver impostato la variabile
- Verifica con: `echo $env:GOOGLE_API_KEY` (PowerShell)

### Errore: "ModuleNotFoundError"
- Assicurati di aver installato tutte le dipendenze: `pip install -r requirements.txt`

### Errore rate limit API
- Gemini ha limiti gratuiti. Attendi qualche minuto o passa a un piano a pagamento.
