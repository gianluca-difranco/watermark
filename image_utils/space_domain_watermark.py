import cv2
import numpy as np
import os
from pathlib import Path

from classes.dataclass import SaveAttackContext
from image_utils.utils import apply_attacks, save_and_compare, calculate_psnr, calculate_ssim,show_watermark


def apply_watermark_color(input_image_path: Path, output_dir_path: Path, watermark_image_path: Path) -> Path:
    """
    Applica un watermark nei LSB di tutti i canali (RGB) di un'immagine a colori.
    """

    # Immagine caricata a colori
    original_img = cv2.imread(str(input_image_path))
    # necessario recuperare h,w in questo modo perché viene restituita una tripla essendo a colori
    h, w, _ = original_img.shape

    # watermark caricato in scala di grigi
    watermark_img = cv2.imread(str(watermark_image_path), cv2.IMREAD_GRAYSCALE)

    # Verifico che immagine e watermark abbiano la stessa dimensione
    if watermark_img.shape != (h, w):
        watermark_img = cv2.resize(watermark_img, (w, h))

    # Recupero il bit del watermark da applicare
    watermark_bit = (watermark_img >> 7) & 1


    # Creo una matrice a 3 canali dove ogni canale contiene gli stessi bit del watermark
    watermark_3ch = np.dstack([watermark_bit, watermark_bit, watermark_bit])

    # Applico il watermark
    # Metto a 0 l'ultimo bit di tutti i canali dell'immagine originale
    # ed inserisco il bit del watermark (0 o 1) nella posizione LSB
    watermarked_img = (original_img & 254) | watermark_3ch


    os.makedirs(output_dir_path, exist_ok=True)

    output_path = output_dir_path / 'watermarked_img.png'

    # Salvo l'immagine finale a colori
    cv2.imwrite(str(output_path), watermarked_img)


    return output_path

def extract_watermark(image: np.ndarray) -> np.ndarray:
    trans_lsb_original = (image & 1) * 255
    return trans_lsb_original.astype(np.uint8)

def space_wm_attack_and_compare(host_path: Path, watermark_path:Path, output_dir_path : Path) -> None:
    if not output_dir_path.exists():
        os.makedirs(output_dir_path)

    watermarked_img_path: Path = apply_watermark_color(input_image_path=host_path,output_dir_path=output_dir_path, watermark_image_path=watermark_path)

    # verifico quanto differisce l'immagine originale da quella con watermark
    calculate_ssim(host_path, watermarked_img_path)
    calculate_psnr(host_path, watermarked_img_path)

    print("-------------------------------------------------------------")
    attacks = apply_attacks(watermarked_img_path)


    context = SaveAttackContext(attacks, output_dir_path, extract_watermark)
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