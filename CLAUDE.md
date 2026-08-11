# Contesto del progetto — HWp NLP (Sapienza)

Questo file contiene il contesto permanente del progetto. Leggilo a ogni sessione
prima di lavorare.

## Come comunichiamo
- Parlami SEMPRE in italiano.
- La GUIDA didattica che ti chiedo va scritta in ITALIANO.
- Il REPORT finale e il CODICE (commenti inclusi) saranno in INGLESE (standard
  accademico; il template LaTeX è in inglese).

## Fase attuale del lavoro
- In questa fase NON si scrive codice. Serve prima una guida che mi spieghi tutto.
- Scriverò io esplicitamente quando vorrò che tu produca del codice.
- Il compito puntuale (es. "scrivimi la guida") te lo do io in chat.

## Dove trovare le informazioni (cartella "info")
Non dare NULLA per scontato: basati su ciò che leggi davvero in questi file.
- `info/NLP Homework and Exam.pdf` → regole ufficiali dell'esame e requisiti
  dell'homework/progetto (cosa consegnare, valutazione, vincoli formali).
- `info/proposal.txt` → la MIA proposta, GIÀ APPROVATA dai docenti. Definisce task,
  dataset e modelli. Attieniti a questa.
- `info/NLP_24_25_Homework_or_Homeworkp_template/` → template LaTeX ACL per il report
  finale (.tex, .sty, .bib).

## Contesto del progetto
Titolo: "Natural Language Inference: A Comparative Evaluation of Transformer Models on
Adversarial Datasets".
Obiettivo: valutazione comparativa sperimentale di come diverse architetture Transformer
resistono ad attacchi adversariali nel task di Natural Language Inference (NLI),
analizzandone i failure mode e capendo se i modelli moderni si basano su euristiche
linguistiche superficiali invece che su vera comprensione semantica.
- Task: NLI (Natural Language Inference / Recognizing Textual Entailment).
- Dataset: MNLI (training/baseline) e ANLI adversarial, round R1, R2, R3.
- Modelli confrontati: DistilBERT (baseline leggera) vs DeBERTa-v3-base (modello avanzato).
- Lavoro da singolo studente.

## Setup operativo (dallo per assodato)
- Macchina principale per l'addestramento: Google Colab (GPU gratuita, tipicamente T4).
- Sincronizzazione codice: GitHub (questa cartella è già una mia repo). Flusso previsto:
  scrivo/aggiorno il codice → push su GitHub → `git clone` su Colab.
- Macchina secondaria: Mac con Apple Silicon (chip M), utile per sviluppare il codice,
  esplorare i dati e fare test veloci; il training pesante gira su Colab.

## Il mio livello
Non ho MAI usato Hugging Face (transformers, datasets) né PyTorch, e non ho esperienza di
fine-tuning. Devo capire teoria e strumenti PARTENDO DA ZERO. Non dare per scontato nessun
concetto teorico né come si usa uno strumento. Spiega ogni termine tecnico la prima volta
che lo introduci.

## Fatti verificati da cui attingere (per non inventare)
Usa questi come base, ma verificali/espandili leggendo le fonti; non contraddirli senza motivo.
- Mappatura etichette (identica per MNLI e ANLI su Hugging Face):
  0 = entailment, 1 = neutral, 2 = contradiction.
- Identificatori dataset su Hugging Face: MNLI = `nyu-mll/multi_nli`; ANLI = `facebook/anli`
  (split train_r1/dev_r1/test_r1, ... r2, r3). MNLI ha `validation_matched` e
  `validation_mismatched`; le label del test ufficiale MNLI sono nascoste, quindi si valuta
  sulle validation.
- Ordini di grandezza attesi: dopo fine-tuning su MNLI, accuracy ~82% (DistilBERT) e
  ~88-90% (DeBERTa-v3-base) su MNLI matched; su ANLI l'accuracy crolla molto (spesso
  25-45%), a volte sotto il caso (33%), perché gli esempi sono scelti per ingannare modelli
  addestrati su MNLI. Un'accuracy ~33% dopo il training è invece sintomo di un BUG
  (etichette o learning rate sbagliati).
- Iperparametri tipici per fine-tuning NLI: learning rate 2e-5 (AdamW), 2-3 epoche,
  batch 16-32, max length 128, warmup ~6%, weight decay 0.01, fp16 su GPU.
- Riferimenti chiave (reali, da verificare prima di citarli): Bowman et al. 2015 (SNLI,
  EMNLP); Williams et al. 2018 (MNLI, NAACL); Nie et al. 2020 (ANLI, ACL); McCoy et al.
  2019 (HANS, ACL); Gururangan et al. 2018 (Annotation Artifacts, NAACL); Poliak et al.
  2018 (Hypothesis-Only Baselines, *SEM); Devlin et al. 2019 (BERT, NAACL); Sanh et al.
  2019 (DistilBERT); Liu et al. 2019 (RoBERTa); He et al. 2023 (DeBERTaV3, ICLR).

## Modalità di lavoro
- Lavora con la massima attenzione e rileggi/verifica ciò che scrivi quando serve, ma NON
  sprecare token inutilmente: non ne ho infiniti.
- Se qualcosa non ti è chiaro, FAMMI DOMANDE prima di procedere: non inventare e non tirare
  a indovinare.
