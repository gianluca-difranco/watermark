# Watermark Project

Il seguente repository ha come obiettivo quello di confrontare alcuni algoritmi di inserimento di watermarking invisibile in un'immagine.


Il watermark invisibile ha lo scopo di garantire la protezione del copyright e l'autenticazione dei contenuti digitali, una volta estratto dal file.

##  Scopo del Progetto

L'obiettivo principale di questo lavoro è sviluppare un sistema di watermarking robusto che sia in grado di:

1.  **Embedding (Inserimento Watermark):** Applicare un marchio testuale (watermark) digitale all'interno di un'immagine sorgente in modo impercettibile all'occhio umano, preservando la qualità visiva originale.
2.  **Decoding (Estrazione Watermark):** Rilevare ed estrarre il marchio dall'immagine, permettendo di verificare la proprietà intellettuale o l'autenticità del file, anche in presenza di manipolazioni o compressioni.
3.  **Comparing (Confronto):** Verificare come il watermark alteri la qualità della foto originale, confrontando le prestazioni dei diversi algoritmi di inserimento.
4.  ** Strength (Resistenza agli attacchi):** Verificare quanto il watermark sia robusto, ovvero quanto persista dopo l'applicazione di attacchi comuni.
##  Metodologia

Il progetto utilizza librerie di elaborazione immagini standard (come `OpenCV` e `NumPy`) per manipolare i dati dei pixel o le frequenze dell'immagine.

Per il dominio spaziale si è scelto di inserire l'informazione del watermark sui LSB (least significant bit).
Per quanto riguarda il dominio delle frequenze

##  Installazione

Per eseguire il codice, bisogna seguire questi passaggi:

1.  **Clona il repository:**
    ```bash
    git clone https://github.com/gianluca-difranco/watermark-project
    cd watermark-project
    ```

2.  **Installa le dipendenze:**
    Assicurati di avere Python installato, quindi esegui:
    ```bash
    pip install -r requirements.txt
    ```
## Utilizzo

### 1. Inserimento del Watermark
Esegui lo script per applicare il watermark all'immagine originale:

```bash
python main.py --type space --input files/input.png --watermark files/watermark.png --output files/space_domain_watermark
```

In console verranno mostrati i risultati dell'operazione.
