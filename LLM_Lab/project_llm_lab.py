import streamlit as st
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from ddgs import DDGS
import json
import re 
from streamlit_elements import elements, mui, nivo 
import numpy as np


GEMMA_MODEL = "gemma3:4b"
LLAMA_MODEL = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434"


def extract_json_from_response(text: str) -> (dict | list):
    """
    Estrae il primo oggetto JSON o array JSON valido da una stringa, 
    ignorando il testo verboso degli llm.
    """
    start_index = -1
    first_brace = text.find('{')
    first_bracket = text.find('[')

    if first_brace == -1 and first_bracket == -1:
        st.warning("Nessun oggetto o lista JSON trovato nella risposta del modello.")
        st.write("Testo grezzo:", text)
        return {}

    if first_brace != -1 and (first_brace < first_bracket or first_bracket == -1):
        start_index = first_brace
        start_char = '{'
        end_char = '}'
    else:
        start_index = first_bracket
        start_char = '['
        end_char = ']'
        
    balance = 0
    end_index = -1

    for i in range(start_index, len(text)):
        char = text[i]
        if char == start_char:
            balance += 1
        elif char == end_char:
            balance -= 1
        
        if balance == 0:
            end_index = i
            break
    
    if end_index == -1:
        st.error("Trovato inizio JSON ma non una fine bilanciata.")
        st.write("Testo grezzo:", text)
        return {}

    json_str = text[start_index : end_index + 1]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        st.error(f"Errore di decodifica JSON. Stringa estratta:\n{json_str}\n---\nTesto grezzo:\n{text}")
        return {}
    except Exception as e:
        st.error(f"Errore imprevisto durante l'estrazione del JSON: {e}")
        return {}

@st.cache_resource
def setup_rag(percorso_csv="imprese_attive_2021.csv"):
    """
    Prepara la base di conoscenza RAG usando un file CSV reale, un modello di embedding italiano e un database  vettoriale ChromaDB.
    La divisione in chunks non è richiesta poiché ogni riga è già un'unità logica.
    File csv: Imprese attive 2021, Ministro per la Pubblica Amministrazione.
    """
    try:
        df = pd.read_csv(percorso_csv)
    except FileNotFoundError:
        st.error(f"Errore: file '{percorso_csv}' non trovato. Assicurati che sia nella stessa cartella.")
        return None, None
    except Exception as e:
        st.error(f"Errore durante la lettura del CSV: {e}")
        return None, None

    client = chromadb.Client()
    collection = client.get_or_create_collection("imprese_attive_italia")
    embedding_model = SentenceTransformer("nickprock/sentence-bert-base-italian-uncased") 

    for _, row in df.iterrows():
        doc_text = f"Regione: {row['regione']}, Settore: {row['descrizione_ateco']}, Numero imprese attive: {row['numero_imprese']}"
        embedding = embedding_model.encode(doc_text).tolist()
        collection.add(
            ids=[f"{row['regione']}_{row['codice_ateco']}_{row.name}"],
            embeddings=[embedding],
            documents=[doc_text],
            metadatas=[{
                "regione": row['regione'],
                "settore": row['descrizione_ateco'],
                "codice_ateco": row['codice_ateco']
            }]
        )

    return collection, embedding_model


def search_web_function(query: str, max_results: int = 5) -> str:
    """Effettua una ricerca su DuckDuckGo e restituisce i risultati in formato testo."""
    try:
        ddgs = DDGS()
        results = ddgs.text(query, max_results=max_results)
        output = f"Risultati ricerca per: {query}\n\n"
        for i, r in enumerate(results, 1):
            output += f"{i}. {r.get('title', 'N/A')}\n{r.get('body', '')}\nURL: {r.get('href', '')}\n\n"
        return output
    except Exception as e:
        return f"Errore durante la ricerca web: {str(e)}"


def calculate_risk_matrix(
    regione_specifica: str,
    settore_ateco_pulito: str,
    pie_data_list: list,
    bar_data_list: list,
    sentiment_score: int
) -> dict:
    """
    Tool deterministico per calcolare il livello di rischio qualitativo (Basso, Medio, Alto, Critico)
    basandosi su dati RAG e un punteggio di sentiment pre-calcolato.
    """
    
    kri_values = {
        "concorrenza_locale": 0.0,
        "concorrenza_nazionale_media": 0.0,
        "dominanza_locale": 0.0, # %
        "sentiment_web": float(sentiment_score)
    }
    
    # --- Estrai KRI (Logica di calcolo) ---
    total_imprese_pie = 0
    value_settore_utente_pie = 0
    if pie_data_list:
        for item in pie_data_list:
            item_value = item.get('value', 0)
            item_label = item.get('label', '')
            total_imprese_pie += item_value
            if settore_ateco_pulito and item_label.strip().lower() == settore_ateco_pulito.lower():
                value_settore_utente_pie = item_value
                kri_values["concorrenza_locale"] = item_value
        if total_imprese_pie > 0 and value_settore_utente_pie > 0:
            kri_values["dominanza_locale"] = (value_settore_utente_pie / total_imprese_pie) * 100

    altre_regioni_values = []
    if bar_data_list:
        for item in bar_data_list:
            if regione_specifica and item.get('label', '').upper() != regione_specifica.upper():
                altre_regioni_values.append(item.get('value', 0))
            elif not regione_specifica:
                 altre_regioni_values.append(item.get('value', 0))
        if altre_regioni_values:
            kri_values["concorrenza_nazionale_media"] = np.mean(altre_regioni_values) # pyright: ignore[reportArgumentType]

    details = {k: round(v,1) for k, v in kri_values.items()}

    # --- Mappa KRI a scale qualitative ---
    probabilita = "Bassa"
    concorrenza_norm = min((kri_values["concorrenza_locale"] / 50000) * 100, 100)
    if concorrenza_norm > 66:
        probabilita = "Alta"
    elif concorrenza_norm > 33:
        probabilita = "Media"

    impatto = "Basso"
    dominanza = kri_values["dominanza_locale"]
    sentiment = kri_values["sentiment_web"]
    
    if dominanza < 15 and sentiment < 0:
         impatto = "Alto"
    elif (kri_values["concorrenza_nazionale_media"] / 50000 * 100) > 50 and sentiment < 0:
         impatto = "Alto"
    elif dominanza > 30 or sentiment >= 0:
         impatto = "Medio"

    # --- Definisci la Matrice di Rischio ---
    risk_matrix = {
        ("Bassa", "Basso"): "Basso",
        ("Bassa", "Medio"): "Basso",
        ("Bassa", "Alto"): "Medio",
        ("Media", "Basso"): "Basso",
        ("Media", "Medio"): "Medio",
        ("Media", "Alto"): "Alto",
        ("Alta", "Basso"): "Medio",
        ("Alta", "Medio"): "Alto",
        ("Alta", "Alto"): "Critico",
    }

    livello_rischio = risk_matrix.get((probabilita, impatto), "Indefinito")

    return {
        "livello_rischio": livello_rischio,
        "probabilita": probabilita,
        "impatto": impatto,
        "dettaglio_kri": details
    }

class DeconstructorAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = SystemMessage(content="""
Sei un analista di business. Estrai le informazioni chiave dall'idea fornita.
La tua risposta DEVE contenere unicamente un oggetto JSON valido. NON includere testo prima o dopo il JSON.
Le chiavi devono essere: 'settore', 'tipo_business', 'target', 'dimensione_geografica', 'regione_specifica'.

IMPORTANTE per 'regione_specifica':
- Se l'utente menziona una regione specifica (es: "in Lombardia", "nel Lazio"), inserisci il nome della regione
- Se l'utente menziona una città (es: "a Napoli", "a Firenze"), mappa correttamente alla regione corrispondente e inserisci il nome della regione
- Se l'utente NON specifica una regione, inserisci null
                                    
IMPORTANTE per 'settore':
- Estrai il settore economico principale (es: "ristorazione", "tecnologia", "moda", ecc.)

""")

    def analyze(self, idea: str) -> dict:
        messages = [self.prompt, HumanMessage(content=idea)]
        try:
            response_text = self.llm.invoke(messages).content
        except Exception as e:
            st.error(f"Errore connessione LLM (Agente 1): {e}")
            return {}
        
        json_data = extract_json_from_response(response_text)
        if not isinstance(json_data, dict) or not json_data:
            st.error("L'Agente Deconstructor non è riuscito a generare un JSON valido.")
            return {}
        return json_data


class IntelligenceAgent:
    def __init__(self, llm, collection, embedding_model):
        self.llm = llm
        self.second_llm = ChatOllama(model=GEMMA_MODEL, base_url=OLLAMA_BASE_URL)
        self.collection = collection
        self.embedding_model = embedding_model
        self.synthesis_prompt = SystemMessage(content="""
Sei un analista di mercato esperto. Dati i dati interni (RAG) e le ricerche web, scrivi un brief di mercato conciso e professionale.
Fai un'analisi qualitativa dei dati. NON ripetere i numeri grezzi.
Concentrati su trend, opportunità e contesto competitivo.
La tua risposta deve essere solo prosa.
""")
        
    def _find_best_ateco_description(self, settore: str, regione_filtro: str) -> str | None:
        """
        Traduce un settore generico nella descrizione ATECO ufficiale.
        Recupera più candidati e usa keyword matching per selezionare il migliore.
        """
        try:
            query_text = f"descrizione ATECO per {settore}"
            results = self.collection.query(
                query_embeddings=[self.embedding_model.encode(query_text).tolist()],
                n_results=5,
                where={"regione": regione_filtro}
            )

            if results["ids"] and results["ids"][0]:
                best_ateco_match = None
                highest_score = -1 


                settore_keywords = set(re.findall(r'\b\w+\b', settore.lower()))

                for i, doc_str in enumerate(results["documents"][0]):
                    match = re.search(r"Settore: (.*?)(?:, Numero imprese attive:|$)", doc_str)
                    if match:
                        descrizione_ateco = match.group(1).strip()
                        
                        ateco_keywords = set(re.findall(r'\b\w+\b', descrizione_ateco.lower()))
                        common_keywords = settore_keywords.intersection(ateco_keywords)
                        score = len(common_keywords)

                        if settore.lower() in descrizione_ateco.lower():
                             score += 10 

                        if score > highest_score:
                            highest_score = score
                            best_ateco_match = descrizione_ateco
                
                if best_ateco_match and highest_score > 0: 
                    return best_ateco_match
                elif results["documents"][0]: 

                     first_doc = results["documents"][0][0]
                     match = re.search(r"Settore: (.*?)(?:, Numero imprese attive:|$)", first_doc)
                     if match:
                          fallback_ateco = match.group(1).strip()
                          return fallback_ateco

            else:
                 st.warning(f"Traduzione: Nessun documento trovato per '{settore}' in '{regione_filtro}'")

        except Exception as e:
            st.error(f"Errore grave durante la traduzione del settore: {e}")

        st.error(f"Traduzione fallita: non trovo un settore ATECO per '{settore}'")
        return None

    def gather_data(self, context: dict) -> tuple[str, str, str, str]:
        """
        Esegue le query RAG e Web e restituisce i dati grezzi.
        """
        regione_specifica = context.get('regione_specifica')
        settore = context.get('settore')
        tipo_business = context.get('tipo_business', '')


        regione_filtro = None
        if regione_specifica:
            regione_filtro = regione_specifica.upper()

        rag_data_pie = ""
        rag_data_bar = ""
        web_results = ""


        if regione_filtro:
            try:
                debug_results = self.collection.get(
                    where={"regione": regione_filtro},
                    limit=1,
                    include=["metadatas"]
                )
                if debug_results and debug_results["metadatas"]:
                    pass
                else:
                    st.warning("Debug Metadati: Nessun documento trovato per la regione.")
            except Exception as e:
                st.error(f"Debug Metadati: Errore durante il recupero: {e}")

            settore_ateco_raw = self._find_best_ateco_description(settore, regione_filtro) # type: ignore
            settore_ateco_filtrabile = None
            if settore_ateco_raw:
                 settore_ateco_filtrabile = settore_ateco_raw.strip()
            else:
                 st.warning("Traduzione ATECO fallita, procedo con query semantica.")


            final_pie_docs = []
            user_sector_data_pie = None 

            # Query A: Prendi il settore specifico dell'utente
            where_filter_user_pie = {"regione": regione_filtro}
            if settore_ateco_filtrabile:
                 where_filter_user_pie = {"$and": [
                     {"regione": regione_filtro},
                     {"settore": settore_ateco_filtrabile}
                 ]}

            results_user_sector_pie = self.collection.query(
                query_embeddings=[self.embedding_model.encode(f"settore {settore}").tolist()],
                n_results=1,
                where=where_filter_user_pie
            )
            if results_user_sector_pie["documents"] and results_user_sector_pie["documents"][0]:
                 user_sector_data_pie = results_user_sector_pie["documents"][0][0]


            # Query B: Prendi i top 20 candidati (altri settori) per trovare i top 3
            where_filter_top_pie = {"regione": regione_filtro}
            if settore_ateco_filtrabile:
                where_filter_top_pie = {"$and": [
                    {"regione": regione_filtro},
                    {"settore": {"$ne": settore_ateco_filtrabile}}
                ]}

            results_top_sectors_pie = self.collection.query(
                query_embeddings=[self.embedding_model.encode(f"{regione_filtro} settori economici imprese").tolist()],
                n_results=20,
                where=where_filter_top_pie
            )

            top_others_selected_docs = []
            if results_top_sectors_pie["documents"] and results_top_sectors_pie["documents"][0]:
                temp_extractor = DataExtractorAgent(None) 
                parsed_candidates = []
                for doc_str in results_top_sectors_pie["documents"][0]:
                     label, value = temp_extractor._parse_rag_string(doc_str, extract_key="Settore")
                     if label and value is not None:
                          parsed_candidates.append({"doc": doc_str, "value": value})

                parsed_candidates.sort(key=lambda x: x["value"], reverse=True)

                top_others_selected_docs = [cand["doc"] for cand in parsed_candidates[:3]]

            final_pie_docs.extend(top_others_selected_docs)
            

            if user_sector_data_pie and user_sector_data_pie not in final_pie_docs:
                final_pie_docs.append(user_sector_data_pie)
            elif user_sector_data_pie:
                 st.write("**Settore Utente era già nei Top 3.**")

            
            final_pie_docs = final_pie_docs[:4]
            rag_data_pie = "\n".join(list(dict.fromkeys(final_pie_docs))) 



            bar_docs_list = []

            # Query A: Regione Utente - DEDICATA
            if settore_ateco_filtrabile:
                where_filter_user_bar_dedicated = {"$and": [
                    {"regione": regione_filtro},
                    {"settore": settore_ateco_filtrabile}
                ]}
                results_user_sector_bar_dedicated = self.collection.query(
                    query_embeddings=[self.embedding_model.encode(f"settore {settore}").tolist()],
                    n_results=1,
                    where=where_filter_user_bar_dedicated
                )
                if results_user_sector_bar_dedicated["documents"] and results_user_sector_bar_dedicated["documents"][0]:
                    bar_docs_list.extend(results_user_sector_bar_dedicated["documents"][0])
            else:
                 st.warning("Query Bar (User Region - DEDICATA) saltata: ATECO non trovato.")


            # Query B: Top 4 Regioni NAZIONALI
            if settore_ateco_filtrabile:
                where_filter_other_regions_bar = {"$and": [
                    {"regione": {"$ne": regione_filtro}},
                    {"settore": settore_ateco_filtrabile}
                ]}

                results_bar_other_regions = self.collection.query(
                    query_embeddings=[self.embedding_model.encode(f"imprese settore {settore}").tolist()],
                    n_results=4,
                    where=where_filter_other_regions_bar
                )
                if results_bar_other_regions["documents"] and results_bar_other_regions["documents"][0]:
                    bar_docs_list.extend(results_bar_other_regions["documents"][0])
            else:
                 st.warning("Query Bar (Other Regions) saltata: ATECO non trovato.")

            rag_data_bar = "\n".join(list(dict.fromkeys(bar_docs_list)))
            st.divider() 

        else: 
             rag_data_pie = "Nessuna regione specifica."
             st.warning("Nessuna regione specificata, il grafico a barre mostrerà i top nazionali.")
             settore_ateco_trovato = self._find_best_ateco_description(settore, "ITALIA") # pyright: ignore[reportArgumentType]
             settore_ateco_filtrabile = None
             if settore_ateco_trovato:
                  settore_ateco_filtrabile = settore_ateco_trovato.strip()

             where_filter_bar_no_region = {}
             if settore_ateco_filtrabile:
                  where_filter_bar_no_region = {"settore": settore_ateco_filtrabile} 

             results_bar_no_region = self.collection.query(
                 query_embeddings=[self.embedding_model.encode(f"numero imprese settore {settore}").tolist()],
                 n_results=5,
                 where=where_filter_bar_no_region
             )
             if results_bar_no_region["documents"] and results_bar_no_region["documents"][0]:
                 rag_data_bar = "\n".join(results_bar_no_region["documents"][0])
             else:
                  rag_data_bar = "Nessun dato trovato per il settore specificato."


        query_gen_prompt = [
            SystemMessage(content="""
                Sei un esperto di ricerche Google. Il tuo obiettivo è trovare notizie fresche e dati reali.
                Genera 3 query di ricerca SEMPLICI e DIRETTE per DuckDuckGo.

                REGOLE CRITICHE:
                1. NON usare termini complessi come "analisi", "casi studio", "panoramica", "report", "overview".
                2. Usa un linguaggio naturale, come se stessi cercando notizie su Google News.
                3. Includi sempre la ZONA geografica e l'ANNO corrente.

                STRUTTURA RICHIESTA:
                1. Query per la concorrenza (usa parole come: "migliori", "classifica", "elenco").
                2. Query per le novità (usa parole come: "nuove aperture", "notizie", "chiusure").
                3. Query per i dati (usa parole come: "statistiche", "numeri", "quanti sono").

                Rispondi ESCLUSIVAMENTE con le 3 query, una per riga, senza numeri o elenchi puntati.
                Usa parole chiave specifiche in italiano"""),

            HumanMessage(content=f"Business: {tipo_business}\nSettore: {settore}\nZona: {regione_specifica}\nQuery:")
        ]

        web_results = ""

        try:
            web_query = self.second_llm.invoke(query_gen_prompt).content.strip().replace("\"", "") # pyright: ignore[reportAttributeAccessIssue]
            
            queries = [q.strip() for q in web_query.split('\n') if q.strip()]
            queries = queries[:3]

            with st.spinner(f"🌐 Ricerca IA: '{web_query}'..."):
                for i, q in enumerate(queries):
                    clean_q = q.replace('"', '').replace("'", "").strip()
                    web_result = search_web_function(clean_q, max_results=3)
                    web_results += f"Risultati per query: {clean_q}\n\n" + web_result + "\n\n"

        except Exception as e:
            st.warning(f"Ricerca web non disponibile: {e}")
            web_results = "Nessun dato web disponibile."

        return rag_data_pie, rag_data_bar, web_results, settore_ateco_filtrabile # pyright: ignore[reportReturnType]
    
    def synthesize_brief(self, context, rag_data_pie, rag_data_bar, web_results) -> str:
        """
        Scrive il brief in prosa basandosi sui dati raccolti.
        """
        combined_context = f"""
Idea utente: {context}

=== DATI INTERNI RAG (PER CONTESTO) ===
Dati Locali (per Pie Chart):
{rag_data_pie}

Dati Nazionali (per Bar Chart):
{rag_data_bar}

=== INFORMAZIONI DAL WEB (PER CONTESTO) ===
{web_results}

Scrivi ora il tuo brief in prosa, analizzando i dati forniti.
"""
        
        messages = [self.synthesis_prompt, HumanMessage(content=combined_context)]
        try:
            return self.llm.invoke(messages).content
        except Exception as e:
            st.error(f"Errore connessione LLM (Agente 2): {e}")
            return "Errore durante la generazione del brief."
        

class RiskScoringAgent:
    def __init__(self, llm):
        self.llm = llm # Usato solo per il sentiment
        self.sentiment_prompt = SystemMessage(content="""
Analizza il seguente testo estratto da ricerche web.
Valuta il sentiment generale riguardo al mercato/settore descritto.
Rispondi SOLO con un numero: -1 (negativo), 0 (neutro), 1 (positivo).
""")

    def _analyze_sentiment(self, web_results: str) -> int:
        """Analizza il sentiment dei risultati web."""
        if not web_results or web_results == "Nessun dato web disponibile.":
            return 0 # Neutro
        
        messages = [self.sentiment_prompt, HumanMessage(content=web_results[:1000])]
        try:
            response = self.llm.invoke(messages).content.strip()
            sentiment_score = int(response)
            return sentiment_score if sentiment_score in [-1, 0, 1] else 0
        except Exception as e:
            st.warning(f"Analisi sentiment fallita: {e}")
            return 0

    def _get_cleaned_ateco_label(self, context: dict) -> str | None:
        """
        Chiama l'Agente 2 per ottenere l'ATECO tradotto e lo pulisce.
        """
        regione_specifica = context.get('regione_specifica')
        settore_utente_label_cleaned = None
        
        # Dipendenza esterna: assume che 'agent2' esista nello scope globale
        global agent2 
        if agent2 and context.get('settore'):
            regione_filtro = regione_specifica.upper() if regione_specifica else "ITALIA"
            settore_ateco_tradotto_raw = agent2._find_best_ateco_description(context.get('settore'), regione_filtro) # pyright: ignore[reportArgumentType]
            
            if settore_ateco_tradotto_raw:
                settore_utente_label_raw = settore_ateco_tradotto_raw.strip()
                match = re.match(r"^[A-Z]\s+(.*)", settore_utente_label_raw)
                settore_utente_label_cleaned = match.group(1).strip() if match else settore_utente_label_raw
        
        # Fallback al settore generico se la traduzione ATECO fallisce
        if not settore_utente_label_cleaned:
             settore_utente_label_cleaned = context.get('settore')
             
        return settore_utente_label_cleaned

    def calculate_score(self, context: dict, pie_data_list: list, bar_data_list: list, web_results: str) -> dict:
        """
        Metodo principale dell'agente: prepara gli input e chiama il tool.
        """
        
        # 1. Compito Agente: Analizzare sentiment (LLM)
        sentiment_score = self._analyze_sentiment(web_results)
        
        # 2. Compito Agente: Preparare altri argomenti
        regione_specifica = context.get('regione_specifica')
        settore_ateco_pulito = self._get_cleaned_ateco_label(context)
        
        if not settore_ateco_pulito:
             st.error("RiskAgent: Impossibile determinare il settore ATECO pulito. Calcolo del rischio annullato.")
             return {}

        # 3. Compito Agente: Chiamare il Tool esterno
        # (Chiamata Python diretta, non una chiamata LLM tool-calling)
        risk_data = calculate_risk_matrix(
            regione_specifica=regione_specifica, # pyright: ignore[reportArgumentType]
            settore_ateco_pulito=settore_ateco_pulito,
            pie_data_list=pie_data_list,
            bar_data_list=bar_data_list,
            sentiment_score=sentiment_score
        )
        
        return risk_data

class DataExtractorAgent:
    """
    Agente potenziato con parser regex più robusti che estraggono
    le etichette corrette (Settore per Pie, Regione per Bar).
    """
    def __init__(self, llm):
        self.llm = llm
        
        self.pie_prompt = SystemMessage(content="""
Dati i seguenti dati RAG, estrai le informazioni per un grafico a torta.
La tua risposta DEVE essere unicamente una LISTA JSON di oggetti.
FORMATO RICHIESTO: [{"id": "Label Settore", "label": "Label Settore", "value": 123}]

DATI RAG: {raw_data}
LISTA JSON:
""")
        
        self.bar_prompt = SystemMessage(content="""
Dati i seguenti dati RAG, estrai le informazioni per un grafico a barre.
La tua risposta DEVE essere unicamente una LISTA JSON di oggetti.
FORMATO RICHIESTO: [{"label": "Label Regione", "value": 456}]

DATI RAG: {raw_data}
LISTA JSON:
""")

    def _parse_rag_string(self, text: str, extract_key: str = "Settore") -> tuple[str | None, float | None]:
        """
        Estrae l'etichetta (Settore o Regione) e il valore da una stringa RAG.
        Gestisce correttamente il valore 0.
        """
        try:
            label, value_str = None, None
            value = None 

            value_match = re.search(r"Numero imprese attive: (\d*)", text)
            if value_match:
                value_str = value_match.group(1).strip()
                
                try:
                    value = float(value_str) if value_str else 0.0 
                except ValueError:
                    value = None 

            if extract_key == "Settore":
                label_match = re.search(r"Settore: (.*?),", text)
            else: 
                label_match = re.search(r"Regione: (.*?),", text)

            if label_match:
                label = label_match.group(1).strip()
                
                if extract_key == "Settore" and re.match(r"^[A-Z]\s", label):
                    label = label[2:]

            
            if label is not None and value is not None:
                return label, value
        except Exception as e:
            st.warning(f"Errore Regex Parser: {e} su testo: {text}")
            pass  
        return None, None


    def extract_pie_data(self, context: dict, raw_data_pie: str) -> list:
        try:
            messages = [
                SystemMessage(content=self.pie_prompt.content.format(raw_data=raw_data_pie)), # pyright: ignore[reportAttributeAccessIssue]
                HumanMessage(content="Estrai la lista JSON ora.")
            ]
            response = self.llm.invoke(messages).content
            json_list = extract_json_from_response(response)
            
            if isinstance(json_list, list) and len(json_list) > 0 and 'id' in json_list[0]:
                st.success("Estrazione Pie Chart (LLM) riuscita.")
                return json_list
            else:
                raise Exception("LLM ha restituito una lista vuota o non valida (es. mancava 'id')")
                
        except Exception as e:
            pie_data = []
            for line in raw_data_pie.split('\n'):
                if not line.strip(): continue
                label, value = self._parse_rag_string(line, extract_key="Settore") 
                if label and value:
                    pie_data.append({"id": label, "label": label, "value": value})
            
            if pie_data:
                pass
            else:
                st.error("Fallback Regex Pie fallito. Dati RAG potrebbero essere vuoti.")
            return pie_data


    def extract_bar_data(self, context: dict, raw_data_bar: str) -> list:
        try:
            messages = [
                SystemMessage(content=self.bar_prompt.content.format(raw_data=raw_data_bar)), # pyright: ignore[reportAttributeAccessIssue]
                HumanMessage(content="Estrai la lista JSON ora.")
            ]
            response = self.llm.invoke(messages).content
            json_list = extract_json_from_response(response)
            
            if isinstance(json_list, list) and len(json_list) > 0 and 'label' in json_list[0]:
                st.success("Estrazione Bar Chart (LLM) riuscita.")
                return json_list
            else:
                raise Exception("LLM ha restituito una lista vuota o non valida (es. mancava 'label')")
                
        except Exception as e:
            bar_data = []
            for line in raw_data_bar.split('\n'):
                if not line.strip(): continue
                label, value = self._parse_rag_string(line, extract_key="Regione") 
                if label and value:
                    bar_data.append({"label": label, "value": value})
            
            if bar_data:
                pass
            else:
                st.error("Fallback Regex Bar fallito. Dati RAG potrebbero essere vuoti.")
            return bar_data
        
class AnalystAgent:
    def __init__(self, llm):
        self.llm = llm
        self.draft_prompt = SystemMessage(content="""
Sei un analista strategico specializzato in valutazione di nuove opportunità di business.
**Il report deve includere le seguenti sezioni:**
- Contestualizzazione Idea
- Analisi Mercato (Grafico Pie)
- Analisi Concorrenza (Grafico Bar)
- Valutazione del Rischio (basata sul punteggio e sui KRI)
- **Analisi SWOT:**
    - **Strengths (Punti di Forza):** Quali vantaggi ha l'idea specifica dell'utente (es. nicchia, specializzazione menzionata) nel contesto dei dati trovati?
    - **Weaknesses (Punti di Debolezza):** Quali svantaggi o sfide ha l'idea specifica (es. settore molto affollato evidenziato dai grafici, rischio alto)?
    - **Opportunities (Opportunità):** Quali trend positivi (dalla ricerca web nel brief) o lacune di mercato (bassa concorrenza in nicchie specifiche) emergono?
    - **Threats (Minacce):** Quali rischi esterni (alta concorrenza generale mostrata dai grafici, trend negativi dal web, punteggio di rischio elevato) possono impattare l'idea?
- Raccomandazione Finale (che consideri SWOT e rischio)

Basa ogni punto dello SWOT sui dati concreti forniti nel contesto, brief, grafici e punteggio di rischio.
IMPORTANTE: Non scrivere un report generico. Ogni sezione deve fare riferimento esplicito all'idea originale dell'utente e ai dati visti.
""")
        self.critique_prompt = SystemMessage(content="""
Sei un consulente di business senior. Rivedi criticamente la bozza.
1. COERENZA: Il report collega l'idea ai dati del brief e ai DUE grafici?
2. SPECIFICITÀ: Il report è personalizzato o generico?
3. CONCRETEZZA: Le raccomandazioni sono actionable?
4. COMPLETEZZA: L'analisi dei grafici è corretta?

Bozza da analizzare:
{draft}

Identifica le 3 principali debolezze.
""")
        self.refine_prompt = SystemMessage(content="""
Basandoti sulla bozza, la critica ricevuta e tutti i dati disponibili, scrivi la versione finale del report strategico.

**Il report finale deve:**
- Integrare l'analisi dei grafici e del punteggio di rischio.
- Includere una sezione **Analisi SWOT** ben strutturata e basata sui dati. Ogni punto (Strengths, Weaknesses, Opportunities, Threats) deve fare riferimento ai dati del contesto, brief, grafici o punteggio di rischio.
- Fornire insight specifici e actionable.
- Concludere con una raccomandazione motivata che consideri SWOT e rischio.

Dati completi disponibili:
{full_context}

Bozza iniziale:
{draft}

Critica ricevuta:
{critique}

Scrivi ora il report finale ottimizzato includendo la sezione SWOT.
""")

    def report(self, context, market_brief, charts_data, risk_score_data):
        full_context = f"Contesto: {context}\nBrief: {market_brief}\nDati Grafici: {charts_data}\nDati di Rischio: {risk_score_data}"

        try:
            draft_messages = [self.draft_prompt, HumanMessage(content=full_context)]
            draft_report = self.llm.invoke(draft_messages).content
            
            critique_messages = [self.critique_prompt, HumanMessage(content=draft_report)]
            critique = self.llm.invoke(critique_messages).content
            
            refine_input = f"Dati: {full_context}\nBozza: {draft_report}\nCritica: {critique}"
            refine_messages = [self.refine_prompt, HumanMessage(content=refine_input)]
            
            return self.llm.stream(refine_messages)
            
        except Exception as e:
            st.error(f"Errore connessione LLM (Agente 4): {e}")
            def error_stream():
                yield f"Errore during la generazione del report: {e}"
            return error_stream()



st.set_page_config(layout="wide", page_title="Business Intelligence Co-pilot")

st.title("🚀 Business Intelligence Co-pilot")
st.markdown("Inserisci la tua idea di business e lascia che l'IA la analizzi per te.")

collection, embedding_model = None, None
try:
    collection, embedding_model = setup_rag()
except Exception as e:
    st.error(f"Errore fatale durante l'inizializzazione di ChromaDB. Dettagli: {e}")
    st.stop()

if collection is None or embedding_model is None:
    st.error("Setup RAG non riuscito. Impossibile avviare l'analisi.")
    st.stop()

user_idea = st.text_area("Descrivi la tua idea di business qui:", height=100, placeholder="Es: Voglio aprire una pizzeria gourmet a Roma specializzata in impasti con grani antichi.")

if st.button("Avvia Analisi", type="primary"):
    if not user_idea:
        st.warning("Per favore, inserisci un'idea di business.")
    else:
        llm_gemma = ChatOllama(model=GEMMA_MODEL, base_url=OLLAMA_BASE_URL)
        llm_llama = ChatOllama(model=LLAMA_MODEL, base_url=OLLAMA_BASE_URL)

        agent1 = DeconstructorAgent(llm=llm_gemma)
        agent2 = IntelligenceAgent(llm=llm_llama, collection=collection, embedding_model=embedding_model)
        agent3 = DataExtractorAgent(llm=llm_llama) 
        agent4 = AnalystAgent(llm=llm_llama)
        agent5 = RiskScoringAgent(llm=llm_gemma)

 
        context_data = {}
        rag_data_pie, rag_data_bar, web_results = "", "", ""
        market_brief_data = ""
        charts_data_object = {} 
        pie_data_list = []
        bar_data_list = []
        risk_score_data = {}

        with st.spinner("Agente 1: Analisi del contesto..."):
            context_data = agent1.analyze(user_idea)
        
        if context_data:
            st.success("Fase 1 completata: Contesto Estratto")
            
            with st.spinner("Agente 2: Raccolta Dati (RAG + Web)..."):
                rag_data_pie, rag_data_bar, web_results, settore_ateco_filtrabile = agent2.gather_data(context_data)
            st.success("Fase 2a completata: Dati Grezzi Raccolti")

            with st.spinner("Agente 2: Sintesi Brief..."):
                market_brief_data = agent2.synthesize_brief(context_data, rag_data_pie, rag_data_bar, web_results)
            st.success("Fase 2b completata: Brief di Mercato Generato")
            

            with st.spinner("Agente 3: Estrazione Dati Grafici..."):
                pie_data_list = agent3.extract_pie_data(context_data, rag_data_pie)
                bar_data_list = agent3.extract_bar_data(context_data, rag_data_bar)
            
            charts_data_object = {
                "pie": {
                    "type": "pie",
                    "title": f"Distribuzione Settori in {context_data.get('regione_specifica', 'Italia')}",
                    "data": pie_data_list
                },
                "bar": {
                    "type": "bar",
                    "title": f"Confronto Settore '{settore_ateco_filtrabile}' nelle Regioni",
                    "data": bar_data_list,
                    "x_axis_label": "Regione/Settore", 
                    "y_axis_label": "Numero Imprese"
                }
            }
            
            if charts_data_object and (pie_data_list or bar_data_list):
                st.success(f"Fase 3 completata: Grafici Finalizzati")

                with st.spinner("Agente 5: Calcolo Punteggio di Rischio..."):
                    risk_score_data = agent5.calculate_score(context_data, pie_data_list, bar_data_list, web_results)
                st.success("Fase 5 completata: Punteggio di Rischio Calcolato")
                
                st.divider()
                st.header("Risultati dell'Analisi")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📊 Grafici di Mercato (Nivo)")
                    if not charts_data_object or not isinstance(charts_data_object, dict):
                        st.warning("Grafici non disponibili o dati mancanti.")
                    else:
                        chart_pie = charts_data_object.get("pie", {})
                        chart_bar = charts_data_object.get("bar", {})
                        
                        if chart_pie and chart_pie.get("data"):
                            with elements(f"nivo_chart_pie_{pd.Timestamp.now().isoformat()}"): # pyright: ignore[reportGeneralTypeIssues]
                                mui.Typography(chart_pie.get('title'), variant="h6", gutterBottom=True, sx={"textAlign": "center"})
                                with mui.Box(sx={"height": 350}):
                                    nivo.Pie(data=chart_pie['data'], margin={"top": 20, "right": 80, "bottom": 80, "left": 80}, innerRadius=0.5, padAngle=0.7, cornerRadius=3, activeOuterRadiusOffset=8, borderWidth=1, borderColor={"from": "color", "modifiers": [["darker", 0.2]]}, arcLinkLabelsSkipAngle=10, arcLinkLabelsTextColor="#fff", arcLinkLabelsThickness=2, arcLinkLabelsColor={"from": "color"}, arcLabelsSkipAngle=10, arcLabelsTextColor={"from": "color", "modifiers": [["darker", 2]]}, theme={"tooltip": {"container": {"background": "#333"}}, "text": {"fill": "#fff"}})
                            with st.expander("Vedi dati Grafico Pie (JSON)"):
                                st.json(chart_pie)
                        else:
                            st.info("Grafico 1 (Pie) non disponibile.")

                        if chart_bar and chart_bar.get("data"):
                            raw_bar_data = chart_bar['data']
                            
                            aggregated_bar_data = {}
                            for item in raw_bar_data:
                                label = item['label']
                                value = item['value']
                                if label in aggregated_bar_data:
                                    aggregated_bar_data[label] += value
                                else:
                                    aggregated_bar_data[label] = value
                            
                            nivo_bar_data = [{"label": label, "value": value} for label, value in aggregated_bar_data.items()]
                            chart_bar['data'] = nivo_bar_data
                            
                            with elements(f"nivo_chart_bar_{pd.Timestamp.now().isoformat()}"): # pyright: ignore[reportGeneralTypeIssues]
                                mui.Typography(chart_bar.get('title'), variant="h6", gutterBottom=True, sx={"textAlign": "center"})
                                with mui.Box(sx={"height": 350}):
                                    nivo.Bar(data=chart_bar['data'], keys=["value"], color={"scheme": "category10"} ,indexBy="label", margin={"top": 10, "right": 60, "bottom": 50, "left": 60}, padding=0.3, valueScale={"type": "linear"}, indexScale={"type": "band", "round": True}, colors={"scheme": "nivo"}, axisBottom={"legend": chart_bar.get('x_axis_label', 'Categoria'), "legendPosition": "middle", "legendOffset": 32}, axisLeft={"legend": chart_bar.get('y_axis_label', 'Valore'), "legendPosition": "middle", "legendOffset": -45}, theme={"tooltip": {"container": {"background": "#333"}}, "text": {"fill": "#fff"}})
                            with st.expander("Vedi dati Grafico Bar (JSON)"):
                                st.json(chart_bar)
                        else:
                             st.info("Grafico 2 (Bar) non disponibile.")

                with col2:
                    st.subheader("🧠 Contesto Estratto")
                    st.json(context_data)

                    st.subheader("⚖️ Punteggio di Rischio")
                    if risk_score_data and risk_score_data.get("livello_rischio"):
                        livello = risk_score_data["livello_rischio"]
                        prob = risk_score_data["probabilita"]
                        imp = risk_score_data["impatto"]

                        # Scegli colore e icona in base al livello
                        if livello == "Basso":
                            color = "green"
                            icon = "✅"
                        elif livello == "Medio":
                            color = "orange"
                        elif livello == "Alto":
                            color = "red"
                        elif livello == "Critico":
                            color = "darkred"
                        else:
                            color = "gray"

                        st.markdown(f"### Livello Rischio: <span style='color:{color};'>{livello}</span>", unsafe_allow_html=True)
                        st.markdown(f"_(Probabilità Stimata: **{prob}**, Impatto Stimato: **{imp}**)_")

                        # Opzionale: Visualizzare la matrice 3x3
                        st.write("**Matrice di Rischio (Esempio):**")
                        matrix_df = pd.DataFrame({
                            'Impatto Basso': ['Basso', 'Basso', 'Medio'],
                            'Impatto Medio': ['Basso', 'Medio', 'Alto'],
                            'Impatto Alto': ['Medio', 'Alto', 'Critico']
                        }, index=['Probabilità Bassa', 'Probabilità Media', 'Probabilità Alta'])

                        # Evidenzia la cella calcolata (funzione helper)
                        def highlight_cell(x):
                            df = x.copy()
                            # Imposta tutto su sfondo bianco di default
                            df.loc[:,:] = 'background-color: white; color: black'
                            # Trova riga e colonna corrispondenti
                            row_map = {'Bassa': 'Probabilità Bassa', 'Media': 'Probabilità Media', 'Alta': 'Probabilità Alta'}
                            col_map = {'Basso': 'Impatto Basso', 'Medio': 'Impatto Medio', 'Alto': 'Impatto Alto'}
                            row_idx = row_map.get(prob)
                            col_idx = col_map.get(imp)
                            # Se trovati, evidenzia la cella
                            if row_idx and col_idx:
                                df.loc[row_idx, col_idx] = f'background-color: {color}; color: white; font-weight: bold;'
                            return df

                        st.dataframe(matrix_df.style.apply(highlight_cell, axis=None))


                        with st.expander("Dettagli Indicatori (KRI)"):
                            st.json(risk_score_data.get("dettaglio_kri", {}))
                    else:
                        st.warning("Punteggio di rischio non disponibile.")

                    st.subheader("📈 Brief di Mercato (RAG)")
                    st.markdown(market_brief_data)
                    
                st.divider()
                
                st.subheader("📄 Report Strategico Finale")
                with st.spinner("Agente 4: Scrittura e revisione report in corso..."):
                    report_stream = agent4.report(context_data, market_brief_data, charts_data_object, risk_score_data)
                    final_report_data = st.write_stream(report_stream) 
                
                st.success("Fase 4 completata: Report Finale Generato!")
                st.balloons()

            else:
                st.error("Pipeline interrotta: l'agente Dati Grafici non ha prodotto dati validi.")
        else:
            st.error("Pipeline interrotta: l'agente Deconstructor non ha prodotto dati validi.")