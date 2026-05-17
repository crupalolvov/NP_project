import cv2
import mediapipe as mp
import numpy as np
import torch
import os

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    IMAGE_PATH = os.path.join(BASE_DIR, "mano_estesa.jpg")
    OUTPUT_PATH = os.path.join(BASE_DIR, "hand_calibration.pt")

    if not os.path.exists(IMAGE_PATH):
        print(f"ERRORE: Inserisci una foto chiamata 'mano_estesa.jpg' nella cartella:\n{BASE_DIR}")
        return

    print("Caricamento MediaPipe e analisi foto...")
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
    if not os.path.exists(MODEL_PATH):
        print(f"ERRORE: File del modello '{MODEL_PATH}' non trovato.")
        return

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5)
    
    with HandLandmarker.create_from_options(options) as landmarker:
        img = cv2.imread(IMAGE_PATH)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = landmarker.detect(mp_image)

        if not results.hand_world_landmarks:
            print("ERRORE: Nessuna mano rilevata nella foto. Prova con una foto in cui il palmo è ben visibile.")
            return

        print("Mano rilevata! Estrazione delle coordinate 3D metriche...")
        # Usiamo i world_landmarks (in metri, con polso nell'origine) per avere le proporzioni reali
        hand_lms = results.hand_world_landmarks[0]
        
        lms = np.zeros((21, 3))
        for i, lm in enumerate(hand_lms):
            lms[i] = [lm.x, lm.y, lm.z]

        # Dizionario delle ossa e dei giunti
        bones = {
            'Thumb_Proximal': (1, 2), 'Thumb_Intermediate': (2, 3), 'Thumb_Distal': (3, 4),
            'Index_Proximal': (5, 6), 'Index_Intermediate': (6, 7), 'Index_Distal': (7, 8),
            'Middle_Proximal': (9, 10), 'Middle_Intermediate': (10, 11), 'Middle_Distal': (11, 12),
            'Ring_Proximal': (13, 14), 'Ring_Intermediate': (14, 15), 'Ring_Distal': (15, 16),
            'Pinky_Proximal': (17, 18), 'Pinky_Intermediate': (18, 19), 'Pinky_Distal': (19, 20)
        }

        calib_data = {'lengths': {}, 'meta_vectors': {}}

        # 1. Calcolo lunghezze falangi
        for bone_name, (idx1, idx2) in bones.items():
            calib_data['lengths'][bone_name] = float(np.linalg.norm(lms[idx2] - lms[idx1]))

        # 2. Calcolo vettori metacarpi dal polso (LM 0)
        meta_bases = {'Thumb': 1, 'Index': 5, 'Middle': 9, 'Ring': 13, 'Pinky': 17}
        for finger, idx in meta_bases.items():
            calib_data['meta_vectors'][finger] = np.array(lms[idx] - lms[0])

        # Salvataggio Universale
        torch.save(calib_data, OUTPUT_PATH)
        print(f"SUCCESSO! Calibrazione statica salvata in:\n{OUTPUT_PATH}")
        print("Ora IKA.py e animate_hand.py useranno sempre questa misura perfetta.")
        
        # --- VISUALIZZAZIONE RISULTATI SULLA FOTO ---
        print("\nGenerazione anteprima visiva in corso...")
        image_landmarks = results.hand_landmarks[0] # Coordinate per il disegno sull'immagine (normalizzate 0-1)
        h, w, _ = img.shape
        
        # Disegna i metacarpi (dal polso LM_0 alla base di ogni dito)
        for idx in [1, 5, 9, 13, 17]:
            p1 = (int(image_landmarks[0].x * w), int(image_landmarks[0].y * h))
            p2 = (int(image_landmarks[idx].x * w), int(image_landmarks[idx].y * h))
            cv2.line(img, p1, p2, (0, 255, 0), 3)
            
        # Disegna le falangi usando il dizionario 'bones' esistente
        for bone_name, (idx1, idx2) in bones.items():
            p1 = (int(image_landmarks[idx1].x * w), int(image_landmarks[idx1].y * h))
            p2 = (int(image_landmarks[idx2].x * w), int(image_landmarks[idx2].y * h))
            cv2.line(img, p1, p2, (0, 255, 0), 3)
            
        # Disegna i landmark (articolazioni) come punti rossi
        for lm in image_landmarks:
            cv2.circle(img, (int(lm.x * w), int(lm.y * h)), 6, (0, 0, 255), -1)
            
        # Ridimensiona l'immagine se è troppo grande per lo schermo (es. foto da smartphone)
        scale = min(800 / h, 800 / w)
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            
        cv2.imshow("Calibrazione (Premi un tasto qualsiasi sulla foto per chiudere)", img)
        cv2.waitKey(0) # Mette in pausa lo script finché non premi un tasto
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()