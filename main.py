

import cv2
import numpy as np
import pytesseract
import re

# ***************************************************************
# ⚠ Configuração Importante: Caminho do Tesseract ⚠
# ***************************************************************
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'



def preprocessar_para_ocr(roi):
    """
    Deixa a região da placa melhor para o Tesseract:
    - converte pra cinza (se precisar)
    - aumenta o tamanho
    - aplica blur
    - aplica limiarização (binarização)
    """
    if len(roi.shape) == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Aumenta a imagem para o Tesseract enxergar melhor
    roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Remove ruído
    roi = cv2.GaussianBlur(roi, (5, 5), 0)

    # Binarização adaptativa
    roi = cv2.adaptiveThreshold(
        roi, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    return roi


def extrair_texto_placa(cropped_placa):
    """
    Recebe a região da placa, pré-processa e aplica OCR.
    Retorna string com letras e números.
    """
    if cropped_placa is None or cropped_placa.size == 0:
        return ""

    placa_bin = preprocessar_para_ocr(cropped_placa)

    # Mostra a imagem tratada da placa pra você ver se está boa
    cv2.imshow("Placa binarizada para OCR", placa_bin)

    config_ocr = (
        '--psm 7 --oem 3 '
        '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    )

    texto_bruto = pytesseract.image_to_string(placa_bin, config=config_ocr)
    texto_bruto = texto_bruto.upper().strip()

    # Limpa tudo que não for letra ou número
    texto_limpo = ''.join(ch for ch in texto_bruto if ch.isalnum())

    # Opcional: tenta achar algo entre 6 e 8 caracteres (placas comuns)
    padroes = re.findall(r'[A-Z0-9]{5,8}', texto_limpo)

    if padroes:
        return padroes[0]
    else:
        return texto_limpo


def processar_frame_e_detectar(frame):
    """
    Processa o frame e tenta localizar a placa.
    Retorna:
      - frame com contorno desenhado (se achar placa)
      - contorno (location)
      - recorte da placa em escala de cinza
    """
    frame_draw = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)

    # Encontrar contornos
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = keypoints[0]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    cropped_placa = None

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * peri, True)

        if len(approx) == 4:  # quadrilátero
            location = approx

            mask = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(mask, [location], 0, 255, -1)

            (xs, ys) = np.where(mask == 255)
            if xs.size == 0 or ys.size == 0:
                continue

            (topx, topy) = (np.min(xs), np.min(ys))
            (bottomx, bottomy) = (np.max(xs), np.max(ys))

            cropped_placa = gray[topx:bottomx + 1, topy:bottomy + 1]
            break

    if location is not None:
        cv2.drawContours(frame_draw, [location], 0, (0, 255, 0), 3)

    return frame_draw, location, cropped_placa


def iniciar_deteccao_em_tempo_real():
    """
    Liga a webcam, detecta, faz OCR e mostra o resultado na tela.
    """
    # Em alguns PCs no Windows ajuda usar cv2.CAP_DSHOW:
    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERRO: Não foi possível abrir a câmara.")
        return

    print("Câmara iniciada. Pressione 's' para escanear a placa ou 'q' para sair.")

    ultima_placa_lida = ""  # vai guardar o último texto reconhecido

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Não foi possível receber o frame. Saindo...")
            break

        frame_com_deteccao, location, cropped_placa = processar_frame_e_detectar(frame)

        # Se já tivermos lido alguma placa, escrevemos na tela
        if ultima_placa_lida:
            cv2.putText(
                frame_com_deteccao,
                f"Placa: {ultima_placa_lida}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

        cv2.imshow('Deteccao de Placa em Tempo Real', frame_com_deteccao)

        key = cv2.waitKey(1) & 0xFF

        # Sair
        if key == ord('q'):
            break

        # Tentar ler a placa (OCR)
        elif key == ord('s'):
            if cropped_placa is not None:
                texto_placa = extrair_texto_placa(cropped_placa)

                if texto_placa:
                    ultima_placa_lida = texto_placa
                    print(f"\n✅ PLACA IDENTIFICADA: {texto_placa}")
                    cv2.imshow('Placa Recortada', cropped_placa)
                else:
                    print("\n⚠ OCR sem sucesso. Tente posicionar a placa mais claramente.")
            else:
                print("\n⚠ Nenhuma placa detectada no momento do 's'.")

    cap.release()
    cv2.destroyAllWindows()


# --- Execução ---
if __name__ == "__main__":
    iniciar_deteccao_em_tempo_real()
