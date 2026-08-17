import os
import json
import pandas as pd


# =====================================================================
# CONFIGURAZIONE: elenco delle cartelle "root" da analizzare.
# Per ciascuna root lo script cerca al suo interno:
#   - global_summary.txt  -> se presente, contribuisce al CSV delle metriche globali
#   - summary.json         -> se presente, contribuisce al CSV di robustness
# Una root può avere uno solo dei due file, entrambi, o nessuno: lo script
# genera automaticamente solo i CSV per cui ha trovato almeno un dato.
# =====================================================================
ROOTS = [
    "./robustness_results/InstructPix2Pix/DiffVax",
    "./robustness_results/SD_Inpainting/DiffVax",
    "./robustness_results/SD_Img2Img/DiffVax",

    "./robustness_results/InstructPix2Pix/MagicBrush_gray_FT",
    "./robustness_results/SD_Inpainting/MagicBrush_gray_FT",
    "./robustness_results/SD_Img2Img/MagicBrush_gray_FT",

    "./robustness_results/InstructPix2Pix/MagicBrush_photoguard",
    "./robustness_results/SD_Inpainting/MagicBrush_photoguard",
    "./robustness_results/SD_Img2Img/MagicBrush_photoguard",

    "./robustness_results/InstructPix2Pix/MagicBrush_gray_FT",
    "./robustness_results/SD_Inpainting/MagicBrush_gray_FT",
    "./robustness_results/SD_Img2Img/MagicBrush_gray_FT",

    "./robustness_results/InstructPix2Pix/MagicBrush_target_opt",
    "./robustness_results/SD_Inpainting/MagicBrush_target_opt",
    "./robustness_results/SD_Img2Img/MagicBrush_target_opt",

    "./robustness_results/InstructPix2Pix/PhotoGuard",
    "./robustness_results/SD_Inpainting/PhotoGuard",
    "./robustness_results/SD_Img2Img/PhotoGuard",

    "./robustness_results/InstructPix2Pix/VAE_MSE_FT_2_STAGE",
    "./robustness_results/SD_Inpainting/VAE_MSE_FT_2_STAGE",
    "./robustness_results/SD_Img2Img/VAE_MSE_FT_2_STAGE",

    
    "./robustness_results/InstructPix2Pix/VAE_MSE_TARGET_OPT",
    "./robustness_results/SD_Inpainting/VAE_MSE_TARGET_OPT",
    "./robustness_results/SD_Img2Img/VAE_MSE_TARGET_OPT",
    
]


def exp_name_from_root(root):
    """Ricava il nome esperimento da una root, es. output/SD_Inpainting/.../TedBench_x"""
    return root.replace("./output/", "").replace("output/", "").rstrip("/")
 
 
def parse_global_summary(text, exp_name):
    """Parsa il contenuto di un file global_summary.txt"""
    data = {
        'name': exp_name,
        'psnr_quality': None,
        'ssim_quality': None,
        'fsim_quality': None,
        'psnr_protection': None,
        'ssim_protection': None,
        'fsim_protection': None,
        'subject_lpips_orig': None,
        'global_lpips_orig': None,
        'subject_lpips_edited': None,
        'global_lpips_edited': None,
        'attack_success_score': None,
        'attack_successes_count': None,
        'attack_success_rate': None,
        'optimistic_miou_orig': None,
        'pessimistic_miou_orig': None,
        'optimistic_miou_edited': None,
        'pessimistic_miou_edited': None,
    }
 
    lines = text.split('\n')
    in_orig_lpips = False
    in_edited_lpips = False
    in_orig_miou = False
    in_edited_miou = False
    fsim_short_count = 0
 
    for line in lines:
        line = line.strip()
 
        # Parse PSNR
        if 'Average original vs immunized PSNR:' in line:
            val = line.split(':', 1)[1].strip() if ':' in line else None
            data['psnr_quality'] = float(val) if val and val.lower() != 'nan' else None
        if 'Average edited_original vs edited_immunized PSNR:' in line:
            val = line.split(':', 1)[1].strip() if ':' in line else None
            data['psnr_protection'] = float(val) if val and val.lower() != 'nan' else None
 
        # Parse SSIM
        if 'Average original vs immunized SSIM:' in line:
            val = line.split(':', 1)[1].strip() if ':' in line else None
            data['ssim_quality'] = float(val) if val and val.lower() != 'nan' else None
        if 'Average edited_original vs edited_immunized SSIM:' in line:
            val = line.split(':', 1)[1].strip() if ':' in line else None
            data['ssim_protection'] = float(val) if val and val.lower() != 'nan' else None
 
        # Parse FSIM long form
        if 'Average original vs immunized FSIM:' in line or 'Average original vs immunized FSIM (valid=' in line:
            parts = line.split(':', 1)
            val = parts[1].strip() if len(parts) > 1 else None
            data['fsim_quality'] = float(val) if val and val.lower() != 'nan' else None
        if 'Average edited_original vs edited_immunized FSIM:' in line or 'Average edited_original vs edited_immunized FSIM (valid=' in line:
            parts = line.split(':', 1)
            val = parts[1].strip() if len(parts) > 1 else None
            data['fsim_protection'] = float(val) if val and val.lower() != 'nan' else None
 
        # Parse FSIM compact form
        if line.startswith('FSIM:'):
            val = line.split(':', 1)[1].strip() if ':' in line else None
            fval = float(val) if val and val.lower() != 'nan' else None
            if fsim_short_count == 0:
                data['fsim_quality'] = fval
            elif fsim_short_count == 1:
                data['fsim_protection'] = fval
            fsim_short_count += 1
 
        # Parse LPIPS sections
        if 'Original vs Immunized LPIPS' in line:
            in_orig_lpips = True
            in_edited_lpips = False
            continue
        if 'Edited vs Adversarial LPIPS' in line:
            in_orig_lpips = False
            in_edited_lpips = True
            continue
 
        # Parse Segmentation mIoU sections (match esatto per non confondersi con gli header LPIPS)
        if line == '---- Original vs Immunized ----':
            in_orig_miou = True
            in_edited_miou = False
            continue
        if line == '---- Edited vs Adversarial ----':
            in_orig_miou = False
            in_edited_miou = True
            continue
 
        # Reset flags LPIPS
        if (line.startswith('===') or line.startswith('----')) and 'LPIPS' not in line:
            in_orig_lpips = False
            in_edited_lpips = False
 
        # Parse LPIPS Original vs Immunized
        if in_orig_lpips and line.startswith('Subject LPIPS:'):
            val = line.split(':', 1)[1].strip() if ':' in line else None
            data['subject_lpips_orig'] = float(val) if val and val.lower() != 'nan' else None
        if in_orig_lpips and line.startswith('Global LPIPS:'):
            val = line.split(':', 1)[1].strip() if ':' in line else None
            data['global_lpips_orig'] = float(val) if val and val.lower() != 'nan' else None
 
        # Parse LPIPS Edited vs Adversarial
        if in_edited_lpips and line.startswith('Subject LPIPS:'):
            val = line.split(':', 1)[1].strip() if ':' in line else None
            data['subject_lpips_edited'] = float(val) if val and val.lower() != 'nan' else None
        if in_edited_lpips and line.startswith('Global LPIPS:'):
            val = line.split(':', 1)[1].strip() if ':' in line else None
            data['global_lpips_edited'] = float(val) if val and val.lower() != 'nan' else None
 
        # Parse Segmentation mIoU Original vs Immunized
        if in_orig_miou and line.startswith('Optimistic mIoU:'):
            val = line.split(':', 1)[1].strip() if ':' in line else None
            data['optimistic_miou_orig'] = float(val) if val and val.lower() != 'nan' else None
        if in_orig_miou and line.startswith('Pessimistic mIoU:'):
            val = line.split(':', 1)[1].strip() if ':' in line else None
            data['pessimistic_miou_orig'] = float(val) if val and val.lower() != 'nan' else None
 
        # Parse Segmentation mIoU Edited vs Adversarial
        if in_edited_miou and line.startswith('Optimistic mIoU:'):
            val = line.split(':', 1)[1].strip() if ':' in line else None
            data['optimistic_miou_edited'] = float(val) if val and val.lower() != 'nan' else None
        if in_edited_miou and line.startswith('Pessimistic mIoU:'):
            val = line.split(':', 1)[1].strip() if ':' in line else None
            data['pessimistic_miou_edited'] = float(val) if val and val.lower() != 'nan' else None
 
        # Parse Qwen Attack Evaluation Summary
        if line.startswith('Average attack success score'):
            val = line.split(':', 1)[1].strip() if ':' in line else None
            data['attack_success_score'] = float(val) if val and val.lower() != 'nan' else None
        if line.startswith('Successful attacks'):
            val = line.split(':', 1)[1].strip() if ':' in line else None
            data['attack_successes_count'] = int(val) if val else None
        if line.startswith('Attack success rate:'):
            val = line.split(':', 1)[1].strip() if ':' in line else None
            data['attack_success_rate'] = float(val) if val and val.lower() != 'nan' else None
 
    return data
 
 
def parse_robustness_json(json_path, exp_name):
    """
    Parsa un file summary.json con struttura del tipo:
    {
        "clean@None": {"attack_success_score": ..., "optimistic_iou": ..., "pessimistic_iou": ...},
        "jpeg@85": {...},
        ...
    }
    Restituisce una lista di righe in formato long (una riga per ogni tipo di attacco).
    """
    with open(json_path, 'r') as f:
        raw = json.load(f)
 
    rows = []
    for attack_key, metrics in raw.items():
        # Salta eventuali chiavi top-level che non sono dizionari di metriche
        # (es. metadati tipo "n_samples": 30 accanto alle chiavi degli attacchi)
        if not isinstance(metrics, dict):
            print(f"  ⚠ Chiave '{attack_key}' ignorata (valore non è un dizionario): {metrics!r}")
            continue
 
        if '@' in attack_key:
            attack_type, attack_param = attack_key.split('@', 1)
        else:
            attack_type, attack_param = attack_key, None
 
        rows.append({
            'Esperimento': exp_name,
            'attack_key': attack_key,
            'attack_type': attack_type,
            'attack_param': attack_param,
            'attack_success_score': metrics.get('attack_success_score'),
            'optimistic_iou': metrics.get('optimistic_iou'),
            'pessimistic_iou': metrics.get('pessimistic_iou'),
        })
 
    return rows
 
 
def build_global_summary_csv(entries):
    """entries: lista di (exp_name, global_summary_path) per cui il file esiste."""
    print(f"\n--- global_summary.txt trovati: {len(entries)} ---")
 
    all_data = []
    for exp_name, path in entries:
        with open(path, 'r') as f:
            text = f.read()
        all_data.append(parse_global_summary(text, exp_name))
        print(f"✓ Caricato: {path}")
 
    df = pd.DataFrame(all_data)
 
    numeric_cols = df.select_dtypes(include=['float64']).columns
    for col in numeric_cols:
        df[col] = df[col].apply(lambda x: f'{x:.4f}' if pd.notna(x) else '-')
 
    df = df.rename(columns={
        'name': 'Esperimento',
        'psnr_quality': 'PSNR Qualità',
        'ssim_quality': 'SSIM Qualità',
        'fsim_quality': 'FSIM Qualità',
        'psnr_protection': 'PSNR Protezione',
        'ssim_protection': 'SSIM Protezione',
        'fsim_protection': 'FSIM Protezione',
        'subject_lpips_orig': 'LPIPS Sogg. Orig',
        'global_lpips_orig': 'LPIPS Glob. Orig',
        'subject_lpips_edited': 'LPIPS Sogg. Edit',
        'global_lpips_edited': 'LPIPS Glob. Edit',
        'attack_success_rate': 'Tasso Attacco',
        'optimistic_miou_orig': 'mIoU Ottim. Orig',
        'pessimistic_miou_orig': 'mIoU Pessim. Orig',
        'optimistic_miou_edited': 'mIoU Ottim. Edit',
        'pessimistic_miou_edited': 'mIoU Pessim. Edit',
    })
 
    output_file = 'metriche_globali_TedBench.csv'
    df.to_csv(output_file, index=False)
    print(f"✓ Esportato: {output_file} ({len(df)} righe)")
 
 
def build_robustness_csv(entries):
    """entries: lista di (exp_name, summary_json_path) per cui il file esiste."""
    print(f"\n--- summary.json trovati: {len(entries)} ---")
 
    all_rows = []
    for exp_name, path in entries:
        all_rows.extend(parse_robustness_json(path, exp_name))
        print(f"✓ Caricato: {path}")
 
    df = pd.DataFrame(all_rows)
 
    numeric_cols = ['attack_success_score', 'optimistic_iou', 'pessimistic_iou']
    for col in numeric_cols:
        df[col] = df[col].apply(lambda x: f'{x:.4f}' if pd.notna(x) else '-')
 
    df_renamed = df.rename(columns={
        'attack_key': 'Attacco',
        'attack_type': 'Tipo Attacco',
        'attack_param': 'Parametro',
        'attack_success_score': 'Punteggio Successo Attacco',
        'optimistic_iou': 'IoU Ottimistico',
        'pessimistic_iou': 'IoU Pessimistico',
    })
 
    output_file = 'metriche_robustness.csv'
    df_renamed.to_csv(output_file, index=False)
    print(f"✓ Esportato: {output_file} ({len(df_renamed)} righe)")
 
    # Versione wide: una riga per esperimento, colonne per ogni combinazione attacco/metrica
    df_wide = df.pivot_table(
        index='Esperimento',
        columns='attack_key',
        values=numeric_cols,
        aggfunc='first'
    )
    df_wide.columns = [f'{metric}__{attack}' for metric, attack in df_wide.columns]
    df_wide = df_wide.reset_index()
 
    output_file_wide = 'metriche_robustness_wide.csv'
    df_wide.to_csv(output_file_wide, index=False)
    print(f"✓ Esportato: {output_file_wide} ({len(df_wide)} righe)")
 
 
def main():
    print(f"Cercando dati in {len(ROOTS)} cartelle root...")
 
    global_entries = []       # (exp_name, path) per global_summary.txt trovati
    robustness_entries = []   # (exp_name, path) per summary.json trovati
 
    for root in ROOTS:
        exp_name = exp_name_from_root(root)
 
        gs_path = os.path.join(root, 'global_summary.txt')
        rj_path = os.path.join(root, 'summary.json')
 
        found_any = False
 
        if os.path.exists(gs_path):
            global_entries.append((exp_name, gs_path))
            found_any = True
        if os.path.exists(rj_path):
            robustness_entries.append((exp_name, rj_path))
            found_any = True
 
        if not found_any:
            print(f"✗ Nessun file trovato in: {root}")
 
    # Genera solo i CSV per cui esistono dati
    if global_entries:
        build_global_summary_csv(global_entries)
    else:
        print("\nNessun global_summary.txt trovato: CSV metriche globali non generato.")
 
    if robustness_entries:
        build_robustness_csv(robustness_entries)
    else:
        print("\nNessun summary.json trovato: CSV robustness non generato.")
 
 
if __name__ == "__main__":
    main()
 