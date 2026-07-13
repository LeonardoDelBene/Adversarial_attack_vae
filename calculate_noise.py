import os
import numpy as np
from PIL import Image
import pandas as pd
from pathlib import Path
import json

# Percorso base della cartella DiffVax
base_path = "/equilibrium/ldelbene/Immunization/output/SD_Inpainting/full_dataset/DiffVax"

# Lista per salvare i risultati
results = []

# Itero attraverso le 200 coppie di immagini
for i in range(200):
    img_folder = os.path.join(base_path, f"img_{i}")
    
    if not os.path.exists(img_folder):
        print(f"Cartella {img_folder} non trovata!")
        continue
    
    immunized_path = os.path.join(img_folder, "immunized_image.png")
    original_path = os.path.join(img_folder, "original_image.png")
    
    # Verifico che entrambi i file esistono
    if not os.path.exists(immunized_path) or not os.path.exists(original_path):
        print(f"File mancanti in img_{i}")
        continue
    
    try:
        # Carico le immagini
        immunized_img = Image.open(immunized_path).convert('RGB')
        original_img = Image.open(original_path).convert('RGB')
        
        # Converto a numpy array (valori 0-255)
        immunized_array = np.array(immunized_img, dtype=np.float32)
        original_array = np.array(original_img, dtype=np.float32)
        
        # Calcolo la differenza: immunized - original
        noise = immunized_array - original_array
        
        # Calcolo il massimo del valore assoluto
        max_noise = np.max(np.abs(noise))
        
        # Salvo i risultati
        results.append({
            'image_id': i,
            'max_absolute_noise': max_noise,
            'mean_absolute_noise': np.mean(np.abs(noise)),
            'min_absolute_noise': np.min(np.abs(noise)),
            'std_absolute_noise': np.std(np.abs(noise))
        })
        
        if (i + 1) % 20 == 0:
            print(f"Processate {i + 1} immagini...")
    
    except Exception as e:
        print(f"Errore durante il processamento di img_{i}: {e}")

# Salvo i risultati in un DataFrame
df_results = pd.DataFrame(results)

# Stampo le statistiche globali
print("\n" + "="*60)
print("STATISTICHE DEL RUMORE MASSIMO AGGIUNTO")
print("="*60)
print(f"Numero di coppie processate: {len(results)}")
print(f"\nMassimo rumore massimo: {df_results['max_absolute_noise'].max():.4f}")
print(f"Minimo rumore massimo: {df_results['max_absolute_noise'].min():.4f}")
print(f"Media del rumore massimo: {df_results['max_absolute_noise'].mean():.4f}")
print(f"Deviazione standard: {df_results['max_absolute_noise'].std():.4f}")
print(f"Mediana: {df_results['max_absolute_noise'].median():.4f}")
print(f"Quartile 25%: {df_results['max_absolute_noise'].quantile(0.25):.4f}")
print(f"Quartile 75%: {df_results['max_absolute_noise'].quantile(0.75):.4f}")

# Salvo i risultati in CSV
output_csv = "/equilibrium/ldelbene/Immunization/noise_analysis.csv"
df_results.to_csv(output_csv, index=False)
print(f"\nRisultati salvati in: {output_csv}")

# Salvo anche un riassunto JSON
summary = {
    'total_images': len(results),
    'max_absolute_noise': {
        'max': float(df_results['max_absolute_noise'].max()),
        'min': float(df_results['max_absolute_noise'].min()),
        'mean': float(df_results['max_absolute_noise'].mean()),
        'std': float(df_results['max_absolute_noise'].std()),
        'median': float(df_results['max_absolute_noise'].median()),
        'q25': float(df_results['max_absolute_noise'].quantile(0.25)),
        'q75': float(df_results['max_absolute_noise'].quantile(0.75))
    }
}

output_json = "/equilibrium/ldelbene/Immunization/noise_summary.json"
with open(output_json, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Riassunto salvato in: {output_json}")
