from PIL import Image
from numpy import ndarray
from pathlib import Path
import matplotlib.pyplot as plt
import math
from classes.dataclass import SaveAttackContext
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def apply_attacks(watermarked_img_path: Path) -> dict[str,cv2.typing.MatLike]:
    """Applica una serie di attacchi per testare la robustezza."""
    attacks = {}
    watermark_img = cv2.imread(str(watermarked_img_path))

    # 1. Compressione JPEG
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # Qualità 50 (media)
    _, encimg = cv2.imencode('.jpg', watermark_img, encode_param)

    img_jpeg = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    attacks['JPEG (Q=50)'] = img_jpeg

    # 2. Rumore Sale e Pepe
    rows, cols = watermark_img.shape[:2]
    noise = np.zeros((rows, cols), np.uint8)

    cv2.randu(noise, 0, 255)
    img_noise = watermark_img.copy()
    img_noise[noise < 10] = 0  # Pepper
    img_noise[noise > 245] = 255  # Salt
    attacks['Salt & Pepper'] = img_noise

    # 3. Rotazione
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 2, 1)  # 2 gradi
    img_rot = cv2.warpAffine(watermark_img, M, (cols, rows))
    attacks['Rotation (2 deg)'] = img_rot

    return attacks


def save_and_compare(context : SaveAttackContext) -> dict[str,Path]:

    output_file_dict : dict[str,Path] = {}
    for attack_name, attacked_img in context.attacks.items():
        print(f"\nTestando attacco: {attack_name}")
        safe_name = attack_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        attacked_path: Path = Path(f'{context.output_dir}/attacked_{safe_name}.png')
        extracted_wm_path: Path = Path(f'{context.output_dir}/extracted_from_{safe_name}.png')


        cv2.imwrite(str(attacked_path), attacked_img)
        output_file_dict[f'attacked_{safe_name}.png'] = attacked_path

        extracted_wm = context.extract_function(attacked_img, **context.extract_parameters)
        cv2.imwrite(str(extracted_wm_path), extracted_wm)

        output_file_dict[f'extracted_from_{safe_name}.png'] = extracted_wm_path


    return output_file_dict


def calculate_ssim(img1: Path, img2: Path):


    image_1 = cv2.imread(str(img1))
    image_2 = cv2.imread(str(img2))

    image_1, image_2 = _image_resize(image_1, image_2)


    img1_float = image_1.astype(np.float64) / 255.0
    img2_float = image_2.astype(np.float64) / 255.0

    img1_float32 = img1_float.astype(np.float32)
    img2_float32 = img2_float.astype(np.float32)

    img1_rgb = cv2.cvtColor(img1_float32, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2_float32, cv2.COLOR_BGR2RGB)


    ssim_value = ssim(
        img1_rgb,
        img2_rgb,
        data_range=1.0,
        channel_axis=-1,
        multichannel=True

    )
    print(f"SSIM: {ssim_value:.4f} | tra {img1.name} e {img2.name}")
    return ssim_value


def calculate_mse(image_a:Path, image_b:Path):
    """
    Mean Squared Error: Calcola la media degli errori quadrati tra i pixel corrispondenti.
    Un valore più basso è migliore.
    """
    image_1 = cv2.imread(str(image_a))
    image_2 = cv2.imread(str(image_b))


    image_1, image_2 = _image_resize(image_1, image_2)

    # Assicurarsi che le immagini siano float per evitare overflow
    err = np.sum((image_1.astype("float") - image_2.astype("float")) ** 2)
    err /= float(image_1.shape[0] * image_1.shape[1])
    print(f"MSE: {err:.4f} | tra {image_a.name} e {image_b.name}")
    return err


def calculate_psnr(image_a:Path, image_b:Path):
    """
    Peak Signal-to-Noise Ratio: Misura il rapporto tra la potenza massima del segnale e il rumore.
    Valori sopra i 30dB indicano solitamente un'ottima qualità invisibile.
    """

    mse = calculate_mse(image_a, image_b)

    if mse == 0:
        return 100

    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    print(f"PSNR: {psnr:.4f} | tra {image_a.name} e {image_b.name}")
    return psnr


def _image_resize(image_a: ndarray, image_b: ndarray):
    h1, w1 = image_a.shape[:2]
    h2, w2 = image_b.shape[:2]

    if (h1, w1) != (h2, w2):
        # Scegliamo di ridimensionare l'immagine con più pixel per adattarla a quella più piccola
        # (Questo preserva meglio la qualità rispetto all'upscaling)
        if (h1 * w1) > (h2 * w2):
            image_a = cv2.resize(image_a, (w2, h2), interpolation=cv2.INTER_AREA)
        else:
            image_b = cv2.resize(image_b, (w1, h1), interpolation=cv2.INTER_AREA)
    return image_a, image_b


def show_watermark(kwargs, grayscale=False):

    images = [ Image.open(path) for path in kwargs]

    plt.figure(figsize=(8, 4))
    for i, image in enumerate(images):

        plt.subplot(1, len(images), i+1)
        if grayscale:
            plt.imshow(image, cmap="gray")
        else:
            plt.imshow(image)
        plt.axis("off")

    plt.axis("off")
    plt.tight_layout()
    plt.show()