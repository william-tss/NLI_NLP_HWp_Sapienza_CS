# Guida didattica al progetto HWp — NLI su dataset adversariali

Guida completa al progetto, pensata per chi parte da zero con Hugging Face, PyTorch e
fine-tuning. Ogni termine tecnico è spiegato la prima volta che compare. Fonti: regole
ufficiali (`info/NLP Homework and Exam.pdf`), proposta approvata (`info/proposal.txt`),
template ACL (`info/NLP_24_25_Homework_or_Homeworkp_template/`).

## Indice

1. [Cosa devi consegnare e come verrai valutato](#1-cosa-devi-consegnare)
2. [Il task: Natural Language Inference](#2-il-task-nli)
3. [Il problema: euristiche superficiali e bias nei dataset](#3-il-problema)
4. [I dataset: MNLI e ANLI](#4-i-dataset)
5. [I modelli: Transformer, BERT, DistilBERT, DeBERTa-v3](#5-i-modelli)
6. [Il fine-tuning: teoria e iperparametri](#6-il-fine-tuning)
7. [Gli strumenti: PyTorch, Hugging Face, metriche](#7-gli-strumenti)
8. [La pipeline sperimentale del progetto](#8-la-pipeline)
9. [Setup operativo: Colab, GitHub, Mac](#9-setup-operativo)
10. [Analisi dei risultati e failure mode](#10-analisi-risultati)
11. [Il report finale (template ACL)](#11-il-report)
12. [Bibliografia commentata](#12-bibliografia)
13. [Roadmap e checklist](#13-roadmap)
14. [Glossario rapido](#14-glossario)

---

## 1. Cosa devi consegnare e come verrai valutato {#1-cosa-devi-consegnare}

Dal PDF ufficiale (regole 25/26):

- Il voto finale è `ROUND[written_test + (HW o HWp)]`. Lo scritto vale max 28; l'HWp dà
  **fino a 4 punti bonus** (l'HW semplice solo 2). L'HWp è opzionale ma tu l'hai già
  proposto e la proposta è **approvata**: ti attieni a quella.
- L'HWp = HW (rassegna sullo stato dell'arte del task) **+ una TUA valutazione comparativa**
  di almeno Y metodologie esistenti contro almeno una baseline, su almeno Y dataset, con
  risultati sperimentali e discussione scientifica. Y = numero di membri del team. Tu lavori
  **da solo → Y = 1**: bastano 1 metodologia vs 1 baseline su 1 dataset. Il tuo piano
  (DistilBERT baseline vs DeBERTa-v3, su MNLI + ANLI R1/R2/R3) **supera ampiamente il minimo**:
  ottimo per la valutazione.
- Report di **almeno Y×2 = 2 pagine** (riferimenti esclusi), scritto con il template LaTeX
  ACL fornito, in inglese. Consegna: **PDF finale + ZIP con il codice**, prima della
  deadline dell'appello in cui vuoi verbalizzare.
- **Obbligatorio: statement sull'uso di GenAI/LLM** nel report — devi dichiarare in modo
  specifico quali strumenti AI hai usato e per cosa (incluso questo assistente). Omettere
  l'uso può penalizzare il voto. Le citazioni bibliografiche vanno verificate a mano: mai
  citare paper senza averne controllato l'esistenza.
- Niente contribution statement (serve solo per team da 2).

Il template LaTeX (`acl2015.tex`) prescrive già le sezioni del report: le vediamo al §11.

---

## 2. Il task: Natural Language Inference {#2-il-task-nli}

**Natural Language Inference (NLI)**, detto anche **Recognizing Textual Entailment (RTE)**,
è un task di classificazione di coppie di frasi. Dato:

- una **premise** (premessa): una frase assunta come vera;
- una **hypothesis** (ipotesi): una seconda frase;

il modello deve decidere la relazione logico-semantica tra le due, scegliendo tra tre classi:

- **entailment** (implicazione): se la premessa è vera, l'ipotesi è necessariamente vera;
- **neutral**: l'ipotesi potrebbe essere vera o falsa; la premessa non basta a deciderlo;
- **contradiction**: se la premessa è vera, l'ipotesi è necessariamente falsa.

**Mappatura etichette su Hugging Face (identica per MNLI e ANLI):**
`0 = entailment, 1 = neutral, 2 = contradiction`. Questo è un punto critico: se le etichette
si scambiano da qualche parte nel codice, il modello "impara" mapping sbagliati e l'accuracy
resta ~33% (caso). Tienilo a mente per il debug.

### Esempi

| Premise | Hypothesis | Label |
|---|---|---|
| A soccer game with multiple males playing. | Some men are playing a sport. | entailment (0) |
| An older and younger man smiling. | Two men are smiling at cats playing on the floor. | neutral (1) |
| A man inspects the uniform of a figure in some East Asian country. | The man is sleeping. | contradiction (2) |

(Esempi classici in stile SNLI; nel report userai esempi reali estratti dai dataset.)

### Perché è un task importante

L'NLI è considerato un "banco di prova" della comprensione semantica: per risolverlo
davvero servono conoscenza lessicale, sintassi, coreferenza, ragionamento di senso comune.
Applicazioni reali (utile per la sezione "Real-world applications" del report): verifica
di coerenza fattuale nei riassunti automatici, fact-checking, question answering,
valutazione di allucinazioni degli LLM (frameworks che verificano se una risposta è
"entailed" dalle fonti), retrieval semantico, dialogo.

---

## 3. Il problema: euristiche superficiali e bias nei dataset {#3-il-problema}

Questa è la motivazione scientifica di tutto il progetto, e il cuore della sezione
Related Work del report.

I modelli Transformer raggiungono accuracy "sovrumane" su benchmark standard (SNLI ~91%,
MNLI ~90%), ma una serie di lavori ha mostrato che parte di questa performance deriva da
**scorciatoie statistiche** presenti nei dataset, non da vera comprensione:

- **Annotation artifacts** (Gururangan et al., 2018): i crowdworker che scrivono le ipotesi
  seguono strategie prevedibili. Es.: per creare una contraddizione aggiungono negazioni
  ("not", "nobody"); per un entailment usano parole generiche ("animal" per "dog"). Risultato:
  la sola ipotesi contiene indizi statistici sull'etichetta.
- **Hypothesis-only baselines** (Poliak et al., 2018): un modello che vede SOLO l'ipotesi
  (mai la premessa!) supera nettamente la baseline di maggioranza su SNLI/MNLI. Se il task
  fosse "pulito", dovrebbe stare al 33%. Prova diretta dei bias.
- **HANS** (McCoy et al., 2019): test set diagnostico che mostra come i modelli addestrati
  su MNLI adottino tre euristiche sintattiche fallaci: (a) **lexical overlap** — se tutte le
  parole dell'ipotesi compaiono nella premessa, predici entailment; (b) **subsequence** —
  se l'ipotesi è una sottosequenza contigua della premessa, entailment; (c) **constituent** —
  se l'ipotesi è un costituente sintattico della premessa, entailment. Sui casi in cui queste
  euristiche portano alla risposta sbagliata, BERT crolla quasi a 0%.

**ANLI** (Nie et al., 2020) nasce come risposta: invece di un test diagnostico costruito con
template, usa **umani in loop adversariale contro il modello** per raccogliere esempi che i
modelli sbagliano sistematicamente (dettagli al §4.2). La tua domanda di ricerca: *un modello
più moderno e potente (DeBERTa-v3) è davvero più robusto di una baseline leggera (DistilBERT)
a questi attacchi, o entrambi si affidano a euristiche superficiali?*

---

## 4. I dataset: MNLI e ANLI {#4-i-dataset}

### 4.1 MNLI (Multi-Genre NLI) — Williams et al., 2018

- Identificatore Hugging Face: **`nyu-mll/multi_nli`**.
- ~393.000 coppie di training, raccolte via crowdsourcing su **10 generi testuali**
  (fiction, telefonate trascritte, lettere, report governativi, ecc.), a differenza di SNLI
  che usava solo didascalie di foto.
- Due set di validazione da ~9.800 esempi ciascuno:
  - **`validation_matched`**: stessi generi visti in training;
  - **`validation_mismatched`**: generi mai visti in training (misura la generalizzazione
    di dominio).
- Le etichette del test ufficiale sono **nascoste** (era una competizione Kaggle): quindi
  **si valuta sulle validation**, prassi standard in letteratura. Nel report lo dichiarerai
  esplicitamente.
- Ruolo nel progetto: **corpus di training** per entrambi i modelli e **baseline di
  valutazione** (le performance "in-distribution").

### 4.2 ANLI (Adversarial NLI) — Nie et al., 2020

- Identificatore Hugging Face: **`facebook/anli`**. Split: `train_r1/dev_r1/test_r1`, e
  analoghi per `r2`, `r3`.
- Raccolto con procedura **HAMLET (Human-And-Model-in-the-Loop Enabled Training)**, iterativa
  su tre round: un annotatore umano vede un contesto (premessa, per lo più da Wikipedia) e
  un'etichetta target, e deve **scrivere un'ipotesi che inganni il modello avversario**
  (cioè che il modello classifichi male ma che un altro umano classifichi correttamente).
  Solo gli esempi verificati da umani entrano nel dataset.
- I tre round usano avversari sempre più forti, quindi la difficoltà cresce:
  - **R1**: avversario BERT-large addestrato su SNLI+MNLI (~17k train, 1.000 dev, 1.000 test);
  - **R2**: avversario RoBERTa (ensemble) addestrato anche sui dati di R1 (~45k train, 1.000/1.000);
  - **R3**: avversario ancora più forte, contesti da fonti più varie e più lunghi
    (~100k train, 1.200 dev, 1.200 test).
  (Le dimensioni sono approssimative: verifica i numeri esatti quando carichi i dati e
  riportali in una tabella nel report.)
- Ruolo nel progetto: **solo valutazione** (test_r1, test_r2, test_r3). NON addestrerai su
  ANLI: vuoi misurare quanto i modelli addestrati su MNLI resistono ad attacchi pensati per
  modelli come loro. Questo va detto chiaramente nel report, perché in letteratura ANLI si
  usa in entrambi i modi (Nie et al. addestrano anche su ANLI; tu misuri la robustezza
  *out-of-distribution*).

### 4.3 Risultati attesi (per capire se stai sbagliando)

- Dopo fine-tuning su MNLI: **~82%** (DistilBERT) e **~88–90%** (DeBERTa-v3-base) su
  MNLI validation_matched; mismatched simile o di poco inferiore.
- Su ANLI l'accuracy **crolla** (tipicamente 25–45%, a volte sotto il caso del 33%,
  specie su R1 per modelli deboli: gli esempi sono selezionati apposta per ingannare
  modelli MNLI-like). DeBERTa-v3 dovrebbe reggere meglio di DistilBERT ma comunque
  degradare molto: è esattamente il fenomeno che discuterai.
- **Attenzione**: ~33% su MNLI validation DOPO il training = bug (etichette scambiate,
  learning rate sbagliato, ecc.), non "robustezza bassa". Il crollo legittimo si vede solo
  su ANLI, con MNLI alto.

---

## 5. I modelli: Transformer, BERT, DistilBERT, DeBERTa-v3 {#5-i-modelli}

### 5.1 Dal testo ai numeri: la tokenizzazione

Una rete neurale lavora solo su numeri, quindi il primo passo è convertire il testo.
Il **tokenizer** spezza il testo in **token**: unità sub-lessicali che possono essere
parole intere o pezzi di parola. Perché pezzi? Perché un vocabolario di parole intere
sarebbe enorme e non coprirebbe mai parole rare o nuove. Con la tokenizzazione
**subword**, una parola frequente resta intera, una rara viene scomposta:

```
"The referee was unhappy."
→ ["the", "referee", "was", "un", "##happy", "."]     (WordPiece, stile BERT)
```

Il prefisso `##` indica "continua la parola precedente". Ogni token del vocabolario ha un
**ID** intero (es. `"the"` → 1996); la frase diventa una lista di ID. Algoritmi diversi:
**WordPiece** (BERT/DistilBERT, vocab ~30k) e **SentencePiece/Unigram** (DeBERTa-v3,
vocab 128k, lavora anche sugli spazi). Per questo i token ID di due modelli diversi sono
incompatibili: ogni modello va usato col SUO tokenizer.

Per una coppia NLI il tokenizer produce tre cose:

- `input_ids`: gli ID di `[CLS] premise [SEP] hypothesis [SEP]`;
- `attention_mask`: 1 per i token reali, 0 per il padding (vedi §6);
- (per alcuni modelli) `token_type_ids`: 0 per la premessa, 1 per l'ipotesi, così il
  modello sa a quale frase appartiene ogni token.

Ogni ID viene poi mappato al suo **embedding**: un vettore di numeri reali (768 dimensioni
nei modelli "base") appreso durante il training, che rappresenta il token in uno spazio
continuo dove parole simili hanno vettori vicini. All'embedding del token si somma (in
BERT) un **positional embedding** che codifica la posizione nella frase — senza di esso il
modello vedrebbe un insieme non ordinato di parole.

### 5.2 La self-attention, passo passo

Il **Transformer** (Vaswani et al., 2017) è una pila di **layer** identici; ogni layer ha
due blocchi: la self-attention e una piccola rete feed-forward. La **self-attention** è il
meccanismo che rende contestuali le rappresentazioni: ogni token aggiorna il proprio
vettore "guardando" tutti gli altri token della sequenza. Come:

1. Da ogni vettore di token si calcolano tre proiezioni lineari: **query** (Q, "cosa sto
   cercando"), **key** (K, "cosa offro come chiave di ricerca"), **value** (V, "il contenuto
   che passo agli altri").
2. Per il token *i*, si fa il prodotto scalare tra la sua query e le key di TUTTI i token:
   punteggi di affinità (scalati per √d per stabilità numerica).
3. Una **softmax** trasforma i punteggi in pesi che sommano a 1: quanto il token *i* deve
   "ascoltare" ciascun altro token.
4. Il nuovo vettore del token *i* è la media pesata dei value di tutti i token.

In formula: `Attention(Q,K,V) = softmax(QKᵀ/√d)·V`. Questo avviene in parallelo su più
**teste** (multi-head attention, 12 teste nei modelli base): ogni testa impara a guardare
relazioni diverse (accordo sintattico, coreferenza, ecc.). Seguono connessioni residue,
layer normalization e il blocco feed-forward; il tutto ripetuto per 12 layer (6 in
DistilBERT). Più si sale nei layer, più le rappresentazioni diventano astratte.

Conseguenza importante per NLI: siccome premise e hypothesis stanno nella STESSA sequenza,
la self-attention confronta direttamente ogni parola dell'ipotesi con ogni parola della
premessa a ogni layer. È qui che il modello può (in teoria) fare vero matching semantico —
o (in pratica) imparare la scorciatoia "tanto overlap lessicale → entailment".

### 5.3 Pre-training e fine-tuning (il paradigma)

- **Pre-training**: il modello viene addestrato su enormi quantità di testo generico con un
  obiettivo auto-supervisionato (senza etichette umane). Per BERT è il **Masked Language
  Modeling (MLM)**: si nascondono ~15% dei token e il modello impara a indovinarli. Così
  apprende grammatica, lessico, conoscenza del mondo.
- **Fine-tuning**: si prende il modello pre-addestrato e lo si ri-addestra brevemente su un
  task specifico con dati etichettati (per te: MNLI), aggiungendo una piccola **testa di
  classificazione** (classification head: un layer lineare che mappa la rappresentazione
  della frase sui 3 logit delle classi). Tutti i pesi si aggiornano, ma di poco (learning
  rate basso). È quello che farai tu; dettagli al §6.

### 5.4 BERT (Devlin et al., 2019) — il capostipite

**BERT** (Bidirectional Encoder Representations from Transformers) è un Transformer
*encoder-only* (legge tutto il testo in entrambe le direzioni; non genera testo, produce
rappresentazioni). BERT-base: 12 layer, hidden size 768, ~110M parametri.

Per i task su coppie di frasi come NLI, l'input si costruisce concatenando le due frasi con
token speciali: `[CLS] premise [SEP] hypothesis [SEP]`. Il token **`[CLS]`** è un segnaposto
la cui rappresentazione finale riassume l'intera coppia: la testa di classificazione legge
quella. Questa codifica congiunta permette alla self-attention di confrontare direttamente
parole della premessa e dell'ipotesi (cross-attention tra le due frasi).

**Il forward pass completo per un esempio NLI**, dall'inizio alla fine:

1. Tokenizer: `(premise, hypothesis)` → `input_ids` + `attention_mask` (lunghezza 128 dopo
   padding/troncamento).
2. Embedding layer: ogni ID → vettore 768-dim (+ positional embedding).
3. 12 layer Transformer: i 128 vettori vengono trasformati layer dopo layer via
   self-attention + feed-forward.
4. Si prende il vettore finale del token `[CLS]` (768 numeri che riassumono la coppia).
5. Testa di classificazione: layer lineare 768→3 → tre **logit**, es. `[2.1, -0.3, 0.5]`.
6. Softmax → probabilità, es. `[0.79, 0.07, 0.14]` → predizione: classe 0 (entailment).

In inferenza ci si ferma qui (si prende l'**argmax**, la classe col logit più alto). In
training si prosegue con la loss (§6).

### 5.5 DistilBERT (Sanh et al., 2019) — la tua baseline

**DistilBERT** è una versione compressa di BERT-base ottenuta per **knowledge distillation**
(distillazione): un modello piccolo ("student") viene addestrato a imitare le distribuzioni
di output di un modello grande ("teacher", BERT-base), oltre che sul task MLM. Risultato:
**6 layer invece di 12, ~66M parametri (−40%), ~60% più veloce, mantiene ~97%** delle
performance di BERT su benchmark standard.

- Checkpoint Hugging Face: **`distilbert-base-uncased`** ("uncased" = testo minuscolizzato,
  il tokenizer non distingue maiuscole).
- Ruolo nel progetto: **baseline leggera**. Ipotesi: essendo meno capiente, si affida di più
  a euristiche superficiali e crollerà di più su ANLI.

### 5.6 DeBERTa-v3-base (He et al., 2023) — il modello avanzato

**DeBERTa** (Decoding-enhanced BERT with disentangled attention) migliora BERT su due fronti;
la **v3** ne aggiunge un terzo:

1. **Disentangled attention**: in BERT contenuto della parola e sua posizione sono sommati
   in un unico vettore; DeBERTa li tiene **separati** (due vettori: contenuto + posizione
   relativa) e calcola l'attention con termini incrociati contenuto-contenuto,
   contenuto-posizione, posizione-contenuto. Modella meglio le relazioni sintattiche
   dipendenti dall'ordine delle parole.
2. **Enhanced mask decoder**: usa anche la posizione assoluta nella predizione dei token
   mascherati.
3. **(v3) Pre-training in stile ELECTRA**: invece del MLM, usa **Replaced Token Detection
   (RTD)** — un piccolo generatore sostituisce alcuni token con alternative plausibili e il
   modello principale (discriminatore) deve dire, per OGNI token, se è originale o
   sostituito. È un segnale di apprendimento più denso ed efficiente del MLM. La v3
   introduce anche il **gradient-disentangled embedding sharing** per stabilizzare questo
   addestramento.

- DeBERTa-v3-base: 12 layer, hidden 768, ~86M parametri di backbone (ma vocabolario molto
  più grande, 128k token, quindi embedding pesanti). Checkpoint: **`microsoft/deberta-v3-base`**.
- Su MNLI è tra i migliori modelli della sua taglia (~90% matched). Ruolo nel progetto:
  **metodologia avanzata** da confrontare con la baseline.

Nota pratica: il tokenizer di DeBERTa-v3 è un **SentencePiece** (diverso dal WordPiece di
BERT/DistilBERT). Con Hugging Face è trasparente: `AutoTokenizer` carica quello giusto per
ciascun modello. Mai usare il tokenizer di un modello con i pesi di un altro.

---

## 6. Il fine-tuning: teoria e iperparametri {#6-il-fine-tuning}

### 6.1 Come funziona l'addestramento (concetti base)

- **Loss (funzione di perdita)**: misura quanto le predizioni sono sbagliate. Per la
  classificazione si usa la **cross-entropy**: penalizza il modello quando assegna bassa
  probabilità alla classe corretta. Il modello produce 3 **logit** (punteggi grezzi, uno per
  classe), la **softmax** li converte in probabilità, la cross-entropy le confronta con
  l'etichetta vera.
- **Gradiente e backpropagation**: si calcola la derivata della loss rispetto a ogni peso
  (il gradiente) e si aggiornano i pesi nella direzione che riduce la loss.
- **Ottimizzatore**: l'algoritmo che decide come usare i gradienti. Lo standard per i
  Transformer è **AdamW** (Adam con weight decay corretto: il **weight decay** è una
  regolarizzazione che spinge i pesi verso valori piccoli per ridurre l'overfitting).
- **Learning rate (LR)**: quanto grande è ogni passo di aggiornamento. Troppo alto → il
  modello "distrugge" ciò che sapeva dal pre-training (training instabile, accuracy al
  caso); troppo basso → non impara. Per il fine-tuning si usano valori piccoli (~2e-5).
- **Warmup**: nei primi passi il LR cresce linearmente da 0 al valore target (es. per il
  primo ~6% degli step), poi decresce linearmente (**linear schedule with warmup**). Evita
  aggiornamenti violenti all'inizio, quando la testa di classificazione è ancora casuale.
- **Epoca**: un passaggio completo sul training set. **Batch**: il gruppetto di esempi
  (es. 32) processati insieme per ogni aggiornamento. **Step**: un aggiornamento dei pesi
  (con 393k esempi e batch 32 → ~12.300 step/epoca).
- **fp16 (mixed precision)**: i calcoli usano numeri in virgola mobile a 16 bit invece di
  32 → ~metà memoria e quasi doppia velocità su GPU, con perdita di precisione trascurabile.
  Da attivare sempre su Colab.
- **Seed**: il seme del generatore casuale (inizializzazione della testa, ordine dei batch).
  Fissarlo (es. 42) rende gli esperimenti **riproducibili** — requisito di serietà
  scientifica da menzionare nel report.
- **Overfitting**: il modello memorizza il training set e peggiora sui dati nuovi. Con 2-3
  epoche su 393k esempi il rischio è basso, ma monitorerai la loss/accuracy di validazione.

### 6.2 Anatomia di uno step di training

Cosa succede, concretamente, a ogni step (il `Trainer` fa tutto questo per te, ma devi
sapere cosa sta succedendo per interpretare log e problemi):

1. **Forward**: un batch di 32 esempi attraversa il modello → 32×3 logit → cross-entropy
   media sul batch (un numero: la loss, es. 0.85).
2. **Backward (backpropagation)**: PyTorch calcola il gradiente della loss rispetto a
   TUTTI i pesi del modello (autograd).
3. **Optimizer step**: AdamW aggiorna ogni peso usando il suo gradiente, il learning rate
   corrente (dato dallo scheduler warmup+decay) e il weight decay.
4. **Azzeramento gradienti** e passaggio al batch successivo.

Cosa vedrai nei log: la **training loss** che parte da ~1.1 (ln 3 ≈ 1.099, il valore della
cross-entropy quando il modello tira a caso su 3 classi — utile sanity check!) e scende
progressivamente verso ~0.3-0.5. Se resta inchiodata a ~1.1, il modello non sta imparando
(LR sbagliato o etichette rotte). A intervalli regolari il Trainer valuta su un set di
validazione e logga `eval_loss` e `eval_accuracy`: è la curva da guardare.

**Gradient accumulation** (se serve): con poca memoria GPU puoi usare batch 16 e
accumulare i gradienti di 2 batch prima dell'optimizer step → batch "effettivo" 32.
Parametro `gradient_accumulation_steps=2` nei TrainingArguments.

### 6.3 Iperparametri di riferimento per questo progetto

| Iperparametro | Valore | Note |
|---|---|---|
| Learning rate | 2e-5 | AdamW; eventualmente 3e-5 per DistilBERT |
| Epoche | 2–3 | MNLI è grande: bastano |
| Batch size | 16–32 | 32 se la memoria GPU regge, altrimenti 16 |
| Max sequence length | 128 | token; taglia le coppie più lunghe (rare in MNLI) |
| Warmup | ~6% degli step | linear schedule |
| Weight decay | 0.01 | standard |
| Precisione | fp16 | su GPU Colab |
| Seed | fisso (es. 42) | riproducibilità |

**Max sequence length**: le coppie vengono troncate/paddate a 128 token. Il **padding**
aggiunge token fittizi per rendere tutte le sequenze del batch della stessa lunghezza (le
GPU lavorano su tensori rettangolari); una **attention mask** dice al modello di ignorarli.
Nota per ANLI R3: i contesti sono più lunghi, alcuni verranno troncati; è un dettaglio
onesto da riportare nella discussione (o da mitigare valutando con max length maggiore).

Stessa configurazione per entrambi i modelli = confronto equo (**controlled comparison**).
Se cambi qualcosa tra i due, dichiaralo e motivalo nel report.

---

## 7. Gli strumenti: PyTorch, Hugging Face, metriche {#7-gli-strumenti}

### 7.1 PyTorch

**PyTorch** è la libreria di deep learning su cui gira tutto. Concetti minimi che ti
serviranno (Hugging Face nasconde quasi tutto il resto):

- **Tensore**: array multidimensionale (come un array NumPy) che può vivere su GPU.
- **`.to("cuda")` / `.to("mps")`**: sposta modello/dati sulla GPU. Su Colab la GPU NVIDIA
  si chiama `cuda`; sul tuo Mac Apple Silicon il backend GPU si chiama `mps` (utile per
  test veloci, non per il training completo).
- **Autograd**: PyTorch calcola i gradienti automaticamente; non scriverai mai una derivata.
- **Modalità train/eval**: `model.train()` attiva il dropout (regolarizzazione che spegne
  neuroni a caso), `model.eval()` lo disattiva per la valutazione. Il `Trainer` di Hugging
  Face lo gestisce da solo.

### 7.2 Hugging Face `transformers`

La libreria che fornisce modelli pre-addestrati e l'infrastruttura di training. Componenti
che userai:

- **Hub**: repository online di modelli/dataset; i checkpoint si scaricano per nome
  (`distilbert-base-uncased`, `microsoft/deberta-v3-base`).
- **`AutoTokenizer`**: carica il tokenizer giusto per il checkpoint indicato e costruisce
  l'input `[CLS] premise [SEP] hypothesis [SEP]` quando gli passi due frasi:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
enc = tok("A man is sleeping.", "The man is awake.",
          truncation=True, max_length=128)
# enc = {"input_ids": [101, 1037, ...], "attention_mask": [1, 1, ...]}
```

- **`AutoModelForSequenceClassification`**: carica il modello pre-addestrato e ci aggiunge
  sopra la testa di classificazione con `num_labels=3` (inizializzata a caso — il warning
  "some weights were newly initialized" è normale e atteso):

```python
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=3)
```

- **`TrainingArguments` + `Trainer`**: il `Trainer` è il ciclo di addestramento già scritto
  (batching, ottimizzatore, scheduler, fp16, valutazione periodica, salvataggio
  checkpoint, logging). Ti evita di scrivere il training loop PyTorch a mano:

```python
from transformers import TrainingArguments, Trainer
args = TrainingArguments(output_dir="out", learning_rate=2e-5,
    num_train_epochs=2, per_device_train_batch_size=32,
    warmup_ratio=0.06, weight_decay=0.01, fp16=True,
    eval_strategy="steps", eval_steps=2000, save_steps=2000, seed=42)
trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                  eval_dataset=val_ds, compute_metrics=accuracy_fn)
trainer.train()
```

### 7.3 Hugging Face `datasets`

- **`load_dataset`** scarica e cacha i dataset già strutturati (campi `premise`,
  `hypothesis`, `label`):

```python
from datasets import load_dataset
mnli = load_dataset("nyu-mll/multi_nli")   # train, validation_matched, validation_mismatched
anli = load_dataset("facebook/anli")       # train_r1, dev_r1, test_r1, ..., r2, r3
print(mnli["train"][0])   # {"premise": "...", "hypothesis": "...", "label": 1, ...}
```

- **`.map(funzione, batched=True)`**: applica la tokenizzazione a tutto il dataset in modo
  efficiente. È il passo di **preprocessing** che trasforma testo in tensori:

```python
def preprocess(batch):
    return tok(batch["premise"], batch["hypothesis"],
               truncation=True, max_length=128)
mnli_tok = mnli.map(preprocess, batched=True)
```

### 7.4 Metriche e librerie di supporto

- **Accuracy** = frazione di predizioni corrette. È LA metrica standard per NLI (classi
  quasi bilanciate nei dataset, quindi è informativa). Libreria: `evaluate` di Hugging Face
  o `scikit-learn`.
- **Confusion matrix** (matrice di confusione): tabella 3×3 che mostra, per ogni classe
  vera, come il modello ha classificato. Fondamentale per l'analisi dei failure mode
  (`scikit-learn` + `matplotlib`/`seaborn` per i grafici).
- Versioni delle librerie: annotale (nel report/README) per riproducibilità. Nota:
  il tokenizer di DeBERTa-v3 richiede i pacchetti `sentencepiece` e `tiktoken`-like
  extra (`pip install sentencepiece`); su Colab si installa al volo.

---

## 8. La pipeline sperimentale del progetto {#8-la-pipeline}

Il disegno sperimentale completo, fase per fase, in formato operativo. Ogni fase indica:
obiettivo, prerequisiti, passi, output prodotti e controlli di correttezza da fare PRIMA
di passare alla fase successiva. (Quando mi dirai di scrivere codice, seguiremo questa
struttura alla lettera.)

### 8.0 Struttura della repository (da creare all'inizio)

```
HWp_NLP_Tossici/
├── src/
│   ├── data.py        # caricamento + tokenizzazione dataset (MNLI, ANLI)
│   ├── train.py       # fine-tuning di UN modello (nome modello via argomento)
│   ├── evaluate.py    # predizioni + accuracy su tutti i set di valutazione
│   └── analysis.py    # confusion matrix, grafici, estrazione errori
├── notebooks/
│   ├── 00_explore.ipynb   # Fase 0 (gira sul Mac)
│   └── colab_run.ipynb    # notebook sottile per Colab: clone, install, lancio script
├── results/           # predizioni (CSV/JSON) e metriche committate
├── report/            # il LaTeX del report (copia del template)
├── requirements.txt   # dipendenze con versioni bloccate
└── README.md          # istruzioni per riprodurre (in inglese, va nello ZIP)
```

Principio: **gli script sono la fonte di verità** (versionati, riproducibili); il notebook
Colab si limita a `git clone` + `pip install -r requirements.txt` + lancio script. Ogni
script accetta argomenti da riga di comando (es. `python src/train.py --model
microsoft/deberta-v3-base`) così lo stesso codice serve per entrambi i modelli — zero
duplicazione, confronto equo garantito.

### Fase 0 — Esplorazione dati (Mac, ~mezza giornata)

- **Obiettivo**: conoscere i dati prima di toccarli; produrre i numeri per il report.
- **Passi**:
  1. `load_dataset` di MNLI e ANLI; stampare 5-10 esempi per split e leggerli davvero.
  2. Tabella dimensioni esatte di ogni split (sostituisce i "circa" del §4).
  3. Distribuzione delle etichette per split (attesa ~33/33/33; verificala).
  4. Distribuzione delle lunghezze in token (con entrambi i tokenizer!): percentile 95/99
     e % di coppie oltre 128 token, per MNLI e per ANLI R1/R2/R3 separatamente.
  5. Scegliere 2-3 esempi belli (uno per classe) da usare nella sezione Examples del report.
- **Output**: `notebooks/00_explore.ipynb` + numeri/tabelle salvati in `results/`.
- **Controlli**: etichette ∈ {0,1,2} con mapping atteso (leggi la feature `label` del
  dataset: Hugging Face espone i nomi delle classi con `features["label"].names` —
  verifica che sia `["entailment", "neutral", "contradiction"]`); MNLI ha anche esempi
  con label -1? (nel train no, ma controlla: -1 = senza gold label, da scartare se presenti).

### Fase 1 — Smoke test end-to-end (Mac o Colab, ~1 ora)

- **Obiettivo**: verificare che TUTTA la pipeline giri prima di spendere ore di GPU.
  Passo che i principianti saltano e poi pagano caro.
- **Passi**: lanciare `train.py` su un campione minuscolo (es. 2.000 esempi di train, 500
  di validation, 1 epoca, DistilBERT) e a seguire `evaluate.py` sui set ridotti.
- **Controlli**: la loss parte da ~1.1 e scende; nessun crash; i file di output hanno il
  formato giusto; l'accuracy sul campione supera il 33% (con 2k esempi arriverà magari a
  50-60%: basta che si muova).

### Fase 2 — Fine-tuning DistilBERT su MNLI (Colab, ~1-3 ore GPU)

- **Obiettivo**: la baseline addestrata.
- **Passi**:
  1. Runtime GPU T4, clone repo, install, login HF Hub (§9).
  2. `python src/train.py --model distilbert-base-uncased` con gli iperparametri del §6.3.
     Valutazione periodica su un sottoinsieme di validation_matched (es. 2.000 esempi, per
     velocità); salvataggio checkpoint su Drive ogni ~2.000 step.
  3. A fine training: salvare il modello finale su Drive E sull'Hub (doppia copia).
  4. Annotare: tempo/epoca, GPU usata, versioni librerie (per il report).
- **Output**: checkpoint finale `distilbert-mnli`.
- **Controlli**: eval_accuracy finale **~80-83%** su matched. Se ~33% → bug (§4.3): NON
  procedere, torna al debug. Se ~70% → sospetto (LR? epoche? dati?): indagare.

### Fase 3 — Fine-tuning DeBERTa-v3-base su MNLI (Colab, ~4-8 ore GPU)

- **Obiettivo**: il modello avanzato, stessi identici iperparametri e stesso script.
- **Passi**: come Fase 2 con `--model microsoft/deberta-v3-base`. È ~2-3× più lento: se la
  sessione Colab cade, riprendere con `resume_from_checkpoint` dall'ultimo checkpoint su
  Drive (§9.3). Valuta di spezzare in 2 sessioni (1 epoca + resume).
- **Output**: checkpoint finale `deberta-v3-mnli`.
- **Controlli**: eval_accuracy **~88-90%** su matched, e > DistilBERT. Se ≈ DistilBERT o
  33% → problema (per DeBERTa-v3 un LR leggermente più basso, es. 1.5-2e-5, è talvolta
  necessario per la stabilità: se il training diverge, è la prima cosa da provare).

### Fase 4 — Valutazione completa (Colab o Mac, ~1 ora)

- **Obiettivo**: tutti i numeri del confronto, in un formato che consenta l'analisi.
- **Passi**: per ciascuno dei 2 checkpoint, `evaluate.py` su 5 set: MNLI
  validation_matched, validation_mismatched, ANLI test_r1, test_r2, test_r3.
  Per ogni (modello, set) salvare un CSV con: indice esempio, premise, hypothesis, label
  vera, predizione, e i 3 logit/probabilità. **Le predizioni grezze sono l'output
  principale**: l'accuracy si ricalcola sempre da lì.
- **Output**: 10 CSV in `results/` + tabella riassuntiva accuracy (2 modelli × 5 set).
- **Controlli**: numero di righe di ogni CSV = dimensione del set; accuracy MNLI coerente
  con quella vista in training; nessun set con predizioni tutte uguali.

### Fase 5 — Analisi dei failure mode (Mac, ~1-2 giorni di analisi)

- **Obiettivo**: la discussione scientifica. Dettagli operativi al §10.
- **Output**: figure (grafico per round, confusion matrix) in `report/`, tabella esempi
  qualitativi, note scritte che diventeranno la sezione Discussion.

### Fase 6 — (Opzionale) Hypothesis-only baseline

Ri-addestrare DistilBERT su MNLI passando al tokenizer SOLO l'ipotesi (stringa vuota o
omissione della premessa), poi valutare sugli stessi 5 set. Quantifica i bias di MNLI sul
tuo stesso setup (atteso: ben sopra 33% su MNLI, ~33% su ANLI) e arricchisce molto la
discussione. Costa un training in più di DistilBERT (~1-3 ore): fallo solo se le fasi 1-5
sono chiuse.

### Cosa NON fare (scope creep da evitare)

Sei da solo e i punti sono max 4: niente training su ANLI, niente modelli aggiuntivi,
niente hyperparameter search estensivo, niente HANS come esperimento (citalo solo nella
related work), niente multi-seed (un seed fisso; dichiara il limite nel report).

---

## 9. Setup operativo: Colab, GitHub, Mac {#9-setup-operativo}

### 9.1 Ciclo di lavoro quotidiano

1. **Sul Mac**: scrivi/modifichi gli script in `src/`, test veloci su piccoli campioni
   (1.000 esempi, 50 step) per verificare che tutto giri. Il backend `mps` di PyTorch usa
   la GPU del chip M: sufficiente per debug, non per il training completo.
2. **Push su GitHub**: ogni modifica funzionante → commit + push. La repo è questa cartella.
3. **Su Colab**: apri `notebooks/colab_run.ipynb`, che fa clone + install + lancio script.
4. **Risultati**: i CSV/metriche prodotti su Colab li scarichi (o li salvi su Drive) e li
   committi in `results/` — i numeri del report devono essere rigenerabili.

### 9.2 Procedura Colab, passo per passo

1. **Attivare la GPU**: menu Runtime → Change runtime type → Hardware accelerator: **T4
   GPU**. Verifica con `!nvidia-smi` (deve mostrare la T4 e ~15-16 GB di VRAM).
   Il `!` in un notebook esegue un comando di shell, non Python.
2. **Clonare la repo e installare**:

```
!git clone https://github.com/<tuo-user>/<tua-repo>.git
%cd <tua-repo>
!pip install -r requirements.txt   # transformers, datasets, evaluate, sentencepiece, ...
```

   (Repo privata? Serve un token GitHub nell'URL o la repo va resa pubblica — più semplice.)
3. **Montare Google Drive** (per i checkpoint persistenti):

```python
from google.colab import drive
drive.mount("/content/drive")     # chiede autorizzazione una tantum
```

   Poi si punta `output_dir` dei TrainingArguments a una cartella di Drive, es.
   `/content/drive/MyDrive/hwp_nlp/distilbert-mnli`. Tutto ciò che NON è su Drive
   muore con la sessione.
4. **Login Hugging Face Hub** (opzionale ma consigliato, per `push_to_hub`): crea un
   token su huggingface.co → Settings → Access Tokens, poi
   `from huggingface_hub import login; login()` e incolli il token. Su Colab puoi salvarlo
   nei "Secrets" (icona chiave nel pannello sinistro) per non incollarlo ogni volta.
5. **Lanciare il training**: `!python src/train.py --model distilbert-base-uncased
   --output_dir /content/drive/MyDrive/hwp_nlp/distilbert-mnli`.

### 9.3 Sopravvivere ai limiti di Colab free

- **Disconnessioni**: le sessioni free durano poche ore e cadono per inattività o limiti
  d'uso GPU; a volte la GPU non è disponibile (riprova più tardi o accetta una GPU
  diversa). Pianifica i training in sessioni che puoi sorvegliare.
- **Checkpoint frequenti**: `save_steps=2000` circa (≈ ogni 10-15 min) con `output_dir` su
  Drive e `save_total_limit=2` (tiene solo gli ultimi 2 checkpoint: un checkpoint
  base-size pesa centinaia di MB e Drive free ha 15 GB).
- **Ripresa**: se la sessione cade, nuova sessione → stessi passi 1-4 →
  `trainer.train(resume_from_checkpoint=True)` (o flag `--resume` nello script): riparte
  dall'ultimo checkpoint in `output_dir`, con scheduler e ottimizzatore ripristinati.
- **Cache dataset**: MNLI+ANLI si riscaricano a ogni sessione (~qualche minuto: accettabile;
  in alternativa si cacha su Drive, ma complica — non ne vale la pena).
- **Misura i tempi**: dopo ~100 step il log mostra i secondi/step → stima l'ora di fine
  epoca prima di impegnarti in una sessione troppo corta.

### 9.4 requirements.txt e riproducibilità

Blocca le versioni principali (esempio: `transformers==4.x`, `datasets`, `evaluate`,
`accelerate`, `sentencepiece`, `scikit-learn`, `matplotlib`). Le versioni ESATTE le
fissiamo quando scriveremo il codice (dipendono da cosa c'è preinstallato su Colab in quel
momento); l'importante è che finiscano nel file e nel report. `sentencepiece` è
indispensabile per il tokenizer di DeBERTa-v3.

---

## 10. Analisi dei risultati e failure mode {#10-analisi-risultati}

La differenza tra un HWp mediocre e uno ottimo è qui: non basta la tabella delle accuracy,
serve la **discussione scientifica** (richiesta esplicitamente dalle regole). Tutte le
analisi partono dai 10 CSV di predizioni della Fase 4 — niente va ricalcolato coi modelli.
In ordine di importanza:

### 10.1 Tabella principale

Accuracy dei 2 modelli sui 5 set, più due colonne derivate utili: il **drop assoluto**
(accuracy MNLI-matched − accuracy media ANLI) e la media ANLI. Domande guida da
rispondere per iscritto: quanto perde ciascun modello passando da MNLI ad ANLI, in punti
assoluti e relativi? Il gap DeBERTa−DistilBERT si mantiene, si allarga o si chiude sui
round adversariali? L'accuracy scende sotto il 33% da qualche parte (= il modello è
*sistematicamente* ingannato, peggio del caso)? Matched vs mismatched: il degrado di
dominio "morbido" è piccolo rispetto al degrado adversariale? (Bel contrasto da notare.)

### 10.2 Grafico per round

Un grafico a linee: asse x = {MNLI-m, R1, R2, R3}, asse y = accuracy, due linee (una per
modello) + linea orizzontale tratteggiata al 33% (caso). È LA figura del report. Lettura
attesa: R1 fu costruito contro un avversario BERT-like → dovrebbe colpire DistilBERT
(famiglia BERT) più duramente di DeBERTa; R2/R3 furono costruiti contro RoBERTa, più
forte → anche DeBERTa dovrebbe soffrire. Verifica se i dati confermano questa lettura e
discuti in ogni caso.

### 10.3 Confusion matrix

Per ogni (modello, set): matrice 3×3 con `sklearn.metrics.confusion_matrix` (righe = label
vera, colonne = predizione), visualizzata come heatmap; normalizzala per riga per leggerla
in percentuale. Cosa cercare:

- Su ANLI il modello sbaglia "uniformemente" o **collassa su una classe**? Confronta la
  distribuzione delle predizioni (somme di colonna) con quella vera (~33/33/33).
- Pattern tipici delle euristiche: eccesso di predizioni *entailment* (scorciatoia
  dell'overlap lessicale, McCoy et al.) o eccesso di *contradiction* (scorciatoia della
  negazione, Gururangan et al.).
- Quale coppia di classi viene confusa di più? Classico: neutral ↔ entailment (il confine
  più sottile anche per gli umani).

Test quantitativo semplice e d'effetto: calcola l'overlap lessicale (parole dell'ipotesi
presenti nella premessa / parole dell'ipotesi) per ogni esempio ANLI e confronta il tasso
di predizioni "entailment" negli esempi ad alto vs basso overlap. Se il modello predice
entailment molto più spesso ad alto overlap *anche quando la label vera non è entailment*,
hai una prova diretta dell'euristica sul TUO modello.

### 10.4 Analisi qualitativa

Procedura di selezione (per non scegliere ad hoc): dagli errori su ANLI, estrai (a) 3-4
esempi sbagliati da ENTRAMBI i modelli (debolezze condivise), (b) 2-3 sbagliati solo da
DistilBERT (cosa compra la capacità in più), (c) 1-2 sbagliati solo da DeBERTa se
interessanti. Per ciascuno annota: round, label vera vs predette, e il tipo di ragionamento
richiesto (numerico/quantità, temporale, coreferenza, conoscenza del mondo, negazione,
plausibilità). Nie et al. (2020) forniscono una tassonomia degli esempi ANLI a cui
ispirarti per i nomi delle categorie. I migliori 3-5 finiscono in una tabella nella
Discussion; guarda anche le probabilità: errori "convinti" (p>0.9 sulla classe sbagliata)
sono più interessanti di errori marginali.

### 10.5 (Se fatta la Fase 6) Hypothesis-only

Quanto sopra il 33% arriva su MNLI? (In letteratura ~52-55% per SNLI, ~53% per MNLI: se ci
sei vicino, i bias ci sono anche nel tuo setup.) E su ANLI? Se ~33%, conferma che ANLI ha
meno artifacts sfruttabili — chiude il cerchio dell'argomentazione.

Collega sempre i numeri alla domanda di ricerca: *più capacità e miglior pre-training
(DeBERTa-v3) comprano robustezza vera o solo accuracy in-distribution?* Ogni figura e
tabella deve contribuire a rispondere.

---

## 11. Il report finale (template ACL) {#11-il-report}

In inglese, nel template `acl2015.tex`, ≥2 pagine escluse le references. Struttura già
fissata dal template — mappa dei contenuti:

| Sezione template | Cosa ci metti |
|---|---|
| Abstract | 4-6 frasi: task, problema (robustezza), setup (2 modelli, MNLI→ANLI), risultato chiave |
| 1. Task description | Definizione NLI, le 3 classi (§2 di questa guida) |
| 1.1 Examples | Tabella con esempi reali premise/hypothesis/label da MNLI e ANLI |
| 1.2 Real-world applications | Fact-checking, verifica di summary, QA, hallucination detection (§2) |
| 2. Related work | SNLI→MNLI→artifacts (Gururangan, Poliak)→HANS→ANLI; BERT→DistilBERT→DeBERTa (§3, §5) |
| 3. Datasets and benchmarks | MNLI e ANLI con numeri esatti, tabella degli split (§4) |
| 4. Tools, libraries, papers with code | transformers, datasets, PyTorch, checkpoint usati, repo ANLI, Papers with Code (§7) |
| 5. State-of-the-art evaluation | Come si valuta NLI in letteratura: accuracy su matched/mismatched, leaderboard ANLI, risultati SOTA riportati nei paper |
| 6. Comparative evaluation | Il TUO setup: dataset, sistemi (DeBERTa-v3) vs baseline (DistilBERT), protocollo (fine-tuning su MNLI, iperparametri, valutazione zero-shot su ANLI), metriche |
| 6.1 Results | Tabella principale + grafico per round |
| 6.2 Discussion | Failure mode, confusion matrix, esempi qualitativi, limiti (un solo seed, modelli base-size, troncamento a 128, valutazione su validation MNLI) (§10) |
| 7. Conclusions | Sintesi + future work (training su ANLI, HANS, modelli large, LLM zero-shot) |
| GenAI statement | Obbligatorio: strumenti usati e per cosa, in modo specifico |

Consigli: cita con `\cite{}` da `refs.bib` (aggiungi le entry BibTeX ufficiali prese da
ACL Anthology, non inventate); ogni claim sullo stato dell'arte deve avere una citazione;
tabelle e figure con caption; scrivi il report DOPO aver congelato i risultati, ma
appunta osservazioni man mano che sperimenti.

---

## 12. Bibliografia commentata {#12-bibliografia}

I riferimenti chiave e il loro ruolo nel report (verifica sempre le entry su ACL
Anthology / arXiv prima di citarle):

- **Bowman et al., 2015 (SNLI, EMNLP)** — primo grande corpus NLI; origine del filone.
- **Williams et al., 2018 (MNLI, NAACL)** — il tuo training set; multi-genere,
  matched/mismatched.
- **Nie et al., 2020 (ANLI, ACL)** — il tuo benchmark adversariale; procedura HAMLET,
  paper centrale del progetto.
- **Gururangan et al., 2018 (Annotation Artifacts, NAACL)** — bias negli hypothesis;
  motivazione.
- **Poliak et al., 2018 (Hypothesis-Only Baselines, \*SEM)** — prova quantitativa dei bias.
- **McCoy et al., 2019 (HANS, ACL)** — le tre euristiche sintattiche; chiave per
  interpretare gli errori.
- **Devlin et al., 2019 (BERT, NAACL)** — architettura e paradigma pre-training/fine-tuning.
- **Sanh et al., 2019 (DistilBERT, arXiv/NeurIPS workshop)** — la baseline.
- **Liu et al., 2019 (RoBERTa, arXiv)** — avversario dei round ANLI R2/R3; pre-training
  migliorato.
- **He et al., 2023 (DeBERTaV3, ICLR)** — il modello avanzato. (Per la disentangled
  attention citare anche He et al., 2021, DeBERTa, ICLR.)
- **Vaswani et al., 2017 (Attention Is All You Need, NeurIPS)** — il Transformer.

---

## 13. Roadmap e checklist {#13-roadmap}

Ordine di lavoro suggerito:

1. ☐ Studiare questa guida; chiarire ogni dubbio (chiedimi pure).
2. ☐ Creare la struttura della repo (§8.0).
3. ☐ Fase 0: esplorazione dati sul Mac (primo codice che scriveremo).
4. ☐ Fase 1: smoke test end-to-end su campione piccolo; setup Colab + Drive/Hub (§9.2).
5. ☐ Fase 2: fine-tuning DistilBERT su MNLI; verifica ~80-83%.
6. ☐ Fase 3: fine-tuning DeBERTa-v3; verifica ~88-90%.
7. ☐ Fase 4: valutazione completa; salvare i 10 CSV di predizioni; tabella risultati.
8. ☐ Fase 5: confusion matrix, grafico per round, esempi qualitativi (§10).
9. ☐ (Opz.) Fase 6: hypothesis-only.
10. ☐ Report LaTeX in inglese + refs.bib verificato + GenAI statement.
11. ☐ ZIP con codice + README; PDF; consegna entro la deadline dell'appello.

Checklist qualità prima della consegna: etichette 0/1/2 coerenti ovunque; seed fisso;
numeri del report rigenerabili dai file salvati; ≥2 pagine escluse references; GenAI
statement presente; citazioni verificate; codice e commenti in inglese.

---

## 14. Glossario rapido {#14-glossario}

- **Accuracy**: % di predizioni corrette.
- **Attention mask**: vettore che indica quali token sono reali e quali padding.
- **Batch**: gruppo di esempi processati insieme in un passo di training.
- **Checkpoint**: salvataggio dei pesi del modello a un certo punto del training.
- **Cross-entropy**: loss standard per la classificazione.
- **Embedding**: vettore numerico che rappresenta un token.
- **Epoca**: un passaggio completo sul training set.
- **Fine-tuning**: ri-addestramento breve di un modello pre-addestrato su un task specifico.
- **Logit**: punteggio grezzo per classe, prima della softmax.
- **Loss**: misura dell'errore che il training minimizza.
- **MLM**: Masked Language Modeling, obiettivo di pre-training di BERT.
- **Overfitting**: memorizzazione del training set a scapito della generalizzazione.
- **Padding**: token fittizi per uniformare le lunghezze nel batch.
- **Pre-training**: addestramento auto-supervisionato su testo generico.
- **RTD**: Replaced Token Detection, obiettivo di pre-training di DeBERTa-v3/ELECTRA.
- **Self-attention**: meccanismo con cui ogni token pesa gli altri token del contesto.
- **Softmax**: funzione che trasforma logit in probabilità.
- **Token / tokenizer**: unità sub-lessicale / componente che converte testo in token ID.
- **Zero-shot (qui)**: valutare su un dataset (ANLI) senza averci mai addestrato.

---

*Prossimo passo quando vorrai: dimmi "scrivi il codice della Fase 0" e partiamo
dall'esplorazione dei dati.*
