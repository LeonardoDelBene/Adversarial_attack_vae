import json
import re
from pathlib import Path

# Imposta qui le directory del full_dataset contenenti le cartelle img_0, img_1, ..., img_199
# Ogni directory deve contenere le sottocartelle img_* con i file prompt_and_metrics.txt
BASE_DIRS = [
    Path("/equilibrium/ldelbene/Immunization/output/SD_Inpainting/full_dataset/VAE_MSE"),
    Path("/equilibrium/ldelbene/Immunization/output/SD_Inpainting/full_dataset/VAE_MSE_FT"),
    Path("/equilibrium/ldelbene/Immunization/output/SD_Inpainting/full_dataset/VAE_MSE_FT_2_STAGE"),

    Path("/equilibrium/ldelbene/Immunization/output/SD_Img2Img/full_dataset/VAE_MSE_FT_2_STAGE"),
    Path("/equilibrium/ldelbene/Immunization/output/SD_Img2Img/full_dataset/VAE_MSE_FT"),
    Path("/equilibrium/ldelbene/Immunization/output/SD_Img2Img/full_dataset/VAE_MSE"),
    Path("/equilibrium/ldelbene/Immunization/output/SD_Img2Img/full_dataset/DiffVax"),
    
    Path("/equilibrium/ldelbene/Immunization/output/InstructionPix2Pix/full_dataset/VAE_MSE_FT"),
    Path("/equilibrium/ldelbene/Immunization/output/InstructionPix2Pix/full_dataset/VAE_MSE_FT_2_STAGE"),
    Path("/equilibrium/ldelbene/Immunization/output/InstructionPix2Pix/full_dataset/VAE_MSE"),
    Path("/equilibrium/ldelbene/Immunization/output/InstructionPix2Pix/full_dataset/DiffVax"),
]

FACTOR_KEYS = [
    "background_infiltration",
    "global_coherence_ruin",
    "identity_erasure",
    "scale_distortion",
    "spatial_chaos",
    "texture_degradation",
    "distortion_effectiveness",
    "light_color_mismatch",
    "edge_visibility",
    "alignment_sabotage",
    "editing_incompleteness",
    "total_implausibility",
]


def extract_json_block(text: str):
    text = text.strip()
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()

    # try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # attempt to find any JSON object blocks and parse them
    blocks = []
    stack = []
    in_string = False
    escape = False
    start = None

    for index, char in enumerate(text):
        if char == "\\" and in_string:
            escape = not escape
            continue
        if char == '"' and not escape:
            in_string = not in_string
        escape = False

        if not in_string:
            if char == '{':
                if not stack:
                    start = index
                stack.append('{')
            elif char == '}' and stack:
                stack.pop()
                if not stack and start is not None:
                    blocks.append(text[start:index + 1])
                    start = None

    parsed_blocks = []
    for block in blocks:
        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict):
                parsed_blocks.append(parsed)
        except json.JSONDecodeError:
            continue

    for parsed in parsed_blocks:
        if "factor_scores" in parsed:
            return parsed

    return parsed_blocks[0] if parsed_blocks else None


def parse_factor_scores(file_path: Path):
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError:
        return None

    parsed = extract_json_block(text)
    if not parsed or "factor_scores" not in parsed:
        return None

    factor_scores = parsed["factor_scores"]
    if not isinstance(factor_scores, dict):
        return None

    parsed_scores = {}
    for key in FACTOR_KEYS:
        value = factor_scores.get(key)
        if value is None:
            continue
        try:
            parsed_scores[key] = float(value)
        except (TypeError, ValueError):
            continue

    return parsed_scores if parsed_scores else None


def collect_scores(base_dir: Path):
    if not base_dir.exists() or not base_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    scores_sum = {key: 0.0 for key in FACTOR_KEYS}
    counts = {key: 0 for key in FACTOR_KEYS}
    processed_files = 0

    for img_dir in sorted(base_dir.iterdir()):
        if not img_dir.is_dir() or not img_dir.name.startswith("img_"):
            continue

        metrics_file = img_dir / "prompt_and_metrics.txt"
        if not metrics_file.exists():
            continue

        result = parse_factor_scores(metrics_file)
        if not result:
            continue

        processed_files += 1
        for key, score in result.items():
            scores_sum[key] += score
            counts[key] += 1

    return scores_sum, counts, processed_files


def write_summary(base_dir: Path, lines: list[str]):
    summary_path = base_dir / "global_summary.txt"
    summary_text = "\n".join(lines) + "\n"
    with summary_path.open("a", encoding="utf-8") as f:
        f.write(summary_text)


def process_directory(base_dir: Path):
    scores_sum, counts, processed_files = collect_scores(base_dir)

    if processed_files == 0:
        message = f"No valid prompt_and_metrics.txt files were found under the given directory: {base_dir}"
        print(message)
        write_summary(base_dir, [message])
        return

    lines = [
        f"Directory: {base_dir}",
        f"Processed {processed_files} img_* folders.",
        "Average factor scores:",
    ]

    for key in FACTOR_KEYS:
        if counts[key] > 0:
            avg = scores_sum[key] / counts[key]
            lines.append(f"  {key}: {avg:.4f} ({counts[key]} values)")
        else:
            lines.append(f"  {key}: no valid values found")

    for line in lines:
        print(line)

    write_summary(base_dir, lines)


def main():
    for base_dir in BASE_DIRS:
        process_directory(base_dir)


if __name__ == "__main__":
    main()
