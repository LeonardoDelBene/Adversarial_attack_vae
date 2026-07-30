import os
import pandas as pd


def parse_global_summary(text, filename):
    """Parsa il contenuto di un file global_summary.txt"""
    data = {
        'name': filename.replace('output/', '').replace('/global_summary.txt', ''),
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

    for i, line in enumerate(lines):
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

def main():
    summary_files = [

        "./output/InstructionPix2Pix/full_dataset/TedBench_diff_noise_all/global_summary.txt",
        "./output/SD_Inpainting/full_dataset/TedBench_diff_noise_all/global_summary.txt",
        "./output/SD_Img2Img/full_dataset/TedBench_diff_noise_all/global_summary.txt",

    
        "./output/InstructionPix2Pix/full_dataset/TedBench_diff_noise_mask/global_summary.txt",
        "./output/SD_Inpainting/full_dataset/TedBench_diff_noise_mask/global_summary.txt",
        "./output/SD_Img2Img/full_dataset/TedBench_diff_noise_mask/global_summary.txt",

        "./output/InstructionPix2Pix/full_dataset/TedBench_photoguard/global_summary.txt",
        "./output/SD_Inpainting/full_dataset/TedBench_photoguard/global_summary.txt",
        "./output/SD_Img2Img/full_dataset/TedBench_photoguard/global_summary.txt",

        "./output/InstructionPix2Pix/full_dataset/TedBench_diff_noise_mask_invert/global_summary.txt",
        "./output/SD_Inpainting/full_dataset/TedBench_diff_noise_mask_invert/global_summary.txt",
        "./output/SD_Img2Img/full_dataset/TedBench_diff_noise_mask_invert/global_summary.txt",

    ]

    print(f"Cercando {len(summary_files)} file di riepilogo...")

    all_data = []
    loaded_files = 0

    for file_path in summary_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                text = f.read()
            parsed_data = parse_global_summary(text, file_path)
            all_data.append(parsed_data)
            loaded_files += 1
            print(f"✓ Caricato: {file_path}")
        else:
            print(f"✗ File non trovato: {file_path}")

    print(f"\nTotale file caricati: {loaded_files}/{len(summary_files)}")

    df = pd.DataFrame(all_data)

    # Formatta le colonne numeriche a 4 decimali
    numeric_cols = df.select_dtypes(include=['float64']).columns
    for col in numeric_cols:
        df[col] = df[col].apply(lambda x: f'{x:.4f}' if pd.notna(x) else '-')

    print(f"\nDataFrame creato con {len(df)} righe")

    df_renamed = df.rename(columns={
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

    print("Colonne rinominate")

    output_file = 'metriche_globali_TedBench.csv'
    df.to_csv(output_file, index=False)
    print(f"✓ Esportato: {output_file}")

if __name__ == "__main__":
    main()

