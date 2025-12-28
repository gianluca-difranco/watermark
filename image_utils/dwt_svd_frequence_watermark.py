import os

import numpy as np
import pywt
import cv2
from pathlib import Path

from classes.dataclass import SaveAttackContext
from image_utils.utils import apply_attacks, save_and_compare, calculate_psnr, calculate_ssim,show_watermark


def resize_watermark(wm, target_shape):
    """Ridimensiona la filigrana per adattarla alle dimensioni della sottobanda."""
    # cv2.resize accetta (width, height), mentre shape è (rows, cols)
    return cv2.resize(wm, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)


def save_key(path, embedding_data):
    """Salva i dati necessari per l'estrazione in un file compresso .npz"""
    np.savez_compressed(
        path,
        U_wm=embedding_data['U_wm'],
        V_wm=embedding_data['V_wm'],
        S_LL_orig=embedding_data['S_LL_orig'],
        alpha=embedding_data['alpha'],
        shape_wm=embedding_data['shape_wm']
    )
    print(f"[INFO] Chiave di watermarking salvata in: {path}")


def load_key(path: Path):
    """Carica la chiave per l'estrazione"""
    if not path.exists():
        raise FileNotFoundError(f"Impossibile trovare il file chiave: {path}")

    data = np.load(str(path))
    embedding_data = {
        'U_wm': data['U_wm'],
        'V_wm': data['V_wm'],
        'S_LL_orig': data['S_LL_orig'],
        'alpha': float(data['alpha']),
        'shape_wm': tuple(data['shape_wm'])
    }
    return embedding_data

def apply_watermark_color_svd(input_image_path: Path, output_dir_path: Path, watermark_image_path: Path) -> Path:
    """
    Incorpora la filigrana modificando i valori singolari (SVD) della sottobanda LL (DWT)
    su TUTTI i canali (RGB).
    """
    alpha = 0.1  # Fattore di forza del watermark

    # 1. Caricamento Immagini
    host_img = cv2.imread(str(input_image_path))

    watermark_img = cv2.imread(str(watermark_image_path), cv2.IMREAD_GRAYSCALE)

    if host_img is None or watermark_img is None:
        print("ERRORE: Immagini non trovate.")
        return None

    # Pre-elaborazione: Convertiamo in float32 per i calcoli matematici
    host_img = host_img.astype(np.float32) / 255.0
    watermark_img = watermark_img.astype(np.float32) / 255.0

    # 2. Separazione dei canali (Blue, Green, Red)
    channels = cv2.split(host_img)

    watermarked_channels = []
    S_LL_orig_list = []  # Qui salveremo i valori S originali per ogni canale

    U_wm_save, V_wm_save = None, None
    wm_shape_save = None

    # 3. Elaborazione per ogni canale
    for channel in channels:
        # DWT (Discrete Wavelet Transform)
        coeffs = pywt.dwt2(channel, 'haar')
        LL, (LH, HL, HH) = coeffs

        # Preparazione Watermark
        # Ridimensiona il watermark per matchare la dimensione di LL
        wm_resized = resize_watermark(watermark_img, LL.shape)
        if wm_shape_save is None:
            wm_shape_save = wm_resized.shape

        # -SVD sull'Host
        U_LL, S_LL, V_LL = np.linalg.svd(LL, full_matrices=False)

        S_LL_orig_list.append(S_LL)

        # SVD sul Watermark
        U_wm, S_wm, V_wm = np.linalg.svd(wm_resized, full_matrices=False)
        if U_wm_save is None:
            U_wm_save, V_wm_save = U_wm, V_wm

        # Embedding
        S_LL_new = S_LL + (alpha * S_wm)

        # Ricostruzione (Inverse SVD)
        Sigma_LL_new = np.diag(S_LL_new)
        LL_w = np.dot(U_LL, np.dot(Sigma_LL_new, V_LL))

        # IDWT (Inverse DWT)
        coeffs_w = LL_w, (LH, HL, HH)
        channel_w = pywt.idwt2(coeffs_w, 'haar')

        watermarked_channels.append(channel_w)

    # 4. Unione dei canali
    img_w_merged = cv2.merge(watermarked_channels)

    # Clipping finale per assicurarsi di essere nel range 0-255 e conversione a uint8
    img_w_final = np.clip(img_w_merged * 255, 0, 255).astype(np.uint8)

    # 5. Salvataggio
    os.makedirs(output_dir_path, exist_ok=True)

    output_png = output_dir_path / 'watermarked_img.png'
    output_key = output_dir_path / 'watermarked_img.npz'

    cv2.imwrite(str(output_png), img_w_final)

    # 6. Creazione della Chiave di Estrazione
    embedding_data = {
        'U_wm': U_wm_save,
        'V_wm': V_wm_save,
        'S_LL_orig': np.array(S_LL_orig_list),
        'alpha': alpha,
        'shape_wm': wm_shape_save
    }

    save_key(output_key, embedding_data)

    return output_png


def extract_watermark_color_dwt_svd(watermarked_image, **kwargs):
    """
    Estrae la filigrana dai 3 canali di un'immagine a colori e ne fa la media.
    """

    # Recupero della chiave
    # Per prima cosa, controllo se mi sono stati passati i dati della chiave direttamente
    # o se devo caricarli da un file
    if 'embedding_data' in kwargs.keys():
        embedding_data = kwargs['embedding_data']
    elif 'key_path' in kwargs.keys():
        embedding_data = load_key(kwargs['key_path'])
    else:
        raise ValueError("Devi fornire il percorso del file chiave (.npz) o i dati embedding_data.")

    # recupero i dati necessari dalla chiave: le matrici U e V per ricostruire
    # l'immagine e il fattore 'alpha' che indica quanto era forte la filigrana.
    U_wm = embedding_data['U_wm']
    V_wm = embedding_data['V_wm']
    alpha = embedding_data['alpha']
    shape_wm = embedding_data['shape_wm']  # (H, W) originale della filigrana ridimensionata

    # Recupero i valori dell'immagine originale
    # Questa matrice contiene i valori per tutti e 3 i canali di colore.
    S_LL_orig_matrix = embedding_data['S_LL_orig']

    #  Preparazione dell'immagine
    # Converto l'immagine in numeri decimali (tra 0 e 1) per poter fare calcoli precisi.
    watermarked_image = watermarked_image.astype(np.float32) / 255.0

    # Separo i canali (Blu, Verde, Rosso)
    channels = cv2.split(watermarked_image)

    extracted_candidates = []

    # Elaborazione Canale per Canale
    for i, channel in enumerate(channels):

        # Analisi Wavelet (DWT)
        coeffs_att = pywt.dwt2(channel, 'haar')
        LL_att, _ = coeffs_att

        # Controllo Dimensioni (Sicurezza)
        # Se l'immagine è stata ridimensionata o attaccata, le dimensioni potrebbero non coincidere.
        # forzo i dati ad avere la grandezza che mi aspetto, per evitare errori.
        target_h, target_w = shape_wm[0], shape_wm[1]
        if LL_att.shape != (target_h, target_w):
            LL_att = cv2.resize(LL_att, (target_w, target_h))


        # Decomposizione SVD
        # Scompongo questa parte dell'immagine nei suoi componenti matematici fondamentali (SVD).
        # Ottengo 'S_LL_att', che sono i valori singolari dell'immagine
        _, S_LL_att, _ = np.linalg.svd(LL_att, full_matrices=False)

        # Selezione dati originali corretti
        # Prendo i valori singolari originali corrispondenti allo stesso colore (canale) che sto elaborando adesso.
        if S_LL_orig_matrix.ndim == 2:
            S_LL_orig_channel = S_LL_orig_matrix[i]
        else:
            # Caso di sicurezza per vecchie chiavi in bianco e nero.
            S_LL_orig_channel = S_LL_orig_matrix


        # Formula Inversa: sottraggo i valori dell'immagine originale da quelli dell'immagine attuale.
        # La differenza, divisa per la forza 'alpha', mi restituisce i valori del watermark.
        min_len = min(len(S_LL_att), len(S_LL_orig_channel))
        S_wm_extracted = (S_LL_att[:min_len] - S_LL_orig_channel[:min_len]) / alpha

        # Ricostruzione
        # Metto i valori estratti in una matrice diagonale.
        Sigma_wm_extracted = np.diag(S_wm_extracted)

        # Uso le matrici U e V (che avevo salvato nella chiave) per ricostruire visivamente la filigrana di questo canale.
        wm_layer = np.dot(U_wm[:, :min_len], np.dot(Sigma_wm_extracted, V_wm[:min_len, :]))

        extracted_candidates.append(wm_layer)

    # Unione dei Risultati
    # Faccio la media matematica tra le versioni estratte
    wm_average = np.mean(extracted_candidates, axis=0)

    # --- 5. Post-processing ---
    # Normalizzazione e conversione in [0-255]
    wm_final = np.clip(wm_average * 255, 0, 255).astype(np.uint8)

    return wm_final

def frequence_wm_attack_and_compare(host_path : Path, watermark_path : Path, output_dir_path : Path) -> None:

    if not output_dir_path.exists():
        os.makedirs(output_dir_path)

    watermarked_img_path: Path = apply_watermark_color_svd(host_path, output_dir_path, watermark_path)

    # verifico quanto differisce l'immagine originale da quella con watermark
    calculate_ssim(host_path, watermarked_img_path)
    calculate_psnr(host_path, watermarked_img_path)


    print("-------------------------------------------------------------")
    attacks = apply_attacks(watermarked_img_path)


    key_path: Path = output_dir_path / 'watermarked_img.npz'
    embedding_data = load_key(key_path)


    wm_original_resized = cv2.resize(cv2.imread(str(watermark_path), cv2.IMREAD_GRAYSCALE),
                                     embedding_data['shape_wm'])

    cv2.imwrite(str(watermark_path.with_name("watermark_resize.png")), wm_original_resized)
    extract_parameters = {'embedding_data':embedding_data,'key_path': key_path}
    context = SaveAttackContext(attacks, output_dir_path, extract_watermark_color_dwt_svd, extract_parameters)
    output_file_dict: dict[str,Path] = save_and_compare(context)

    print("-------------------------------------------------------------")
    attacked_images = [watermarked_img_path]
    attacked_watermarks = [watermark_path]
    for key, value in output_file_dict.items():
        if key.startswith('extracted'):
            attacked_watermarks.append(value)
            calculate_ssim(watermark_path, value)
            calculate_psnr(watermark_path, value)
            print("-------------------------------------------------------------")
        else:
            attacked_images.append(value)


    show_watermark(attacked_images)
    show_watermark(attacked_watermarks,grayscale=True)