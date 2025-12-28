from image_utils.dwt_svd_frequence_watermark import frequence_wm_attack_and_compare
from image_utils.space_domain_watermark import space_wm_attack_and_compare
from pathlib import Path
import argparse


def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Watermark Project CLI: Tool per inserire ed estrarre watermark dalle immagini.")
    parser.add_argument('--type', type=str, choices=['space', 'frequence'], required=True,
                        help="Tipo di watermark da applicare: 'space' per il watermark sul dominio spaziale, 'frequence' per il watermark sul dominio delle frequenze.")
    parser.add_argument('--input', '-i', type=str, required=True,
                        help="Path dell'immagine di input (originale o watermarked).")
    parser.add_argument('--watermark', '-w', type=str,
                        help="Path dell'immagine del watermark.")
    parser.add_argument('--output', '-o', type=str, required=True,
                        help="Path dove salvare il risultato.")
    # Parsing degli argomenti
    return parser.parse_args()

def main():
    # --type space --input files/input.png --watermark files/watermark.png --output files/space_domain_watermark
    args = get_args()

    # Percorsi file (modifica se necessario)
    input_image_path = Path(args.input)
    watermark_path = Path(args.watermark)
    output_dir_path = Path(args.output)

    if args.type == 'space':
        #test sul dominio spaziale
        space_wm_attack_and_compare(input_image_path, watermark_path, output_dir_path)
    elif args.type == 'frequence':
        #test sul dominio delle frequenze
        frequence_wm_attack_and_compare(input_image_path, watermark_path, output_dir_path)
    else:
        raise ValueError("Tipo di watermark non valido.")

if __name__ == "__main__":
    main()


