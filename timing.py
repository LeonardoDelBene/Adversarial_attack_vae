"""
timing.py

Misura il tempo di inferenza della pipeline di immunizzazione (funzione
`immunize()` del progetto) per una singola immagine, su GPU, ripetendo
la misura per diversi valori di n_steps (es. 50, 100, 200, 300, 400...).

Nota sul modulo da importare: questo script assume che il tuo file
principale (quello con get_config, load_models, load_sample, immunize)
sia importabile come `main`. Se il file si chiama diversamente, aggiorna
solo la riga di import qui sotto con il nome corretto del modulo — il
resto dello script non va toccato.

Perché non basta un semplice time.perf_counter() prima/dopo la chiamata:
- Le operazioni CUDA sono ASINCRONE: la chiamata Python ritorna subito,
  mentre il lavoro sulla GPU può ancora essere in corso. Senza
  torch.cuda.synchronize() prima di leggere l'orologio, si misura solo
  il tempo di "lancio" dei kernel, non il tempo di calcolo reale.
- La primissima chiamata su GPU è quasi sempre più lenta (cuDNN
  autotuning, allocazioni lazy della cache CUDA, compilazione dei
  kernel): va scartata con qualche run di warm-up, altrimenti la misura
  è pessimistica e non rappresentativa del regime "a caldo". Il warm-up
  viene fatto una sola volta all'inizio (non per ogni n_steps), tanto
  serve solo a "riscaldare" CUDA/cuDNN una volta per tutte.
- Il tempo di una singola immunizzazione può oscillare di qualche decina
  di ms da una chiamata all'altra: per ogni n_steps si ripete la misura
  più volte e si riportano media, deviazione standard, min/max.

NB: `immunize()` richiede i gradienti attivi (calcola la perturbazione
via backprop attraverso n_steps di ottimizzazione), quindi qui NON si
usa torch.no_grad().
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ADATTARE: nome del modulo che contiene get_config / load_models /
# load_sample / immunize (il file che hai incollato sopra).
import main as pl


@dataclass
class TimingConfig:
    n_steps_list: list[int] = field(default_factory=lambda: [50, 100, 200, 300, 400])
    n_warmup: int = 1          # run di riscaldamento CUDA, fatti una sola volta a inizio script
    n_runs: int = 5            # run misurati per ciascun valore di n_steps (costoso: tenerlo basso)
    output_path: Optional[Path] = Path("timing_results/immunize_vs_nsteps.json")


def _sync_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponibile su questo sistema.")
    torch.cuda.synchronize()


def _time_n_runs(image, image_mask, immunization_mdl, config, n_runs: int) -> np.ndarray:
    """Esegue n_runs chiamate a pl.immunize(...) con la config corrente
    e restituisce i tempi misurati (in secondi), con sync CUDA prima/dopo
    ogni chiamata."""
    times_s: list[float] = []
    for i in range(n_runs):
        _sync_cuda()
        start = time.perf_counter()
        _ = pl.immunize(image, image_mask, immunization_mdl, config)
        _sync_cuda()
        end = time.perf_counter()
        elapsed = end - start
        times_s.append(elapsed)
        print(f"    run {i + 1}/{n_runs}: {elapsed:.4f} s")
    return np.array(times_s)


def time_immunize_vs_nsteps(
    image, image_mask, immunization_mdl, config, timing_config: TimingConfig
) -> dict:
    """Misura il tempo di `pl.immunize(...)` su una singola immagine per
    ciascun valore di n_steps in timing_config.n_steps_list.

    `config` viene modificato in-place impostando `config["n_steps"]` a
    ogni iterazione: usa una copia se ti serve preservare l'originale.
    """

    # --- Warm-up: una volta sola, con il primo valore di n_steps ---
    config["n_steps"] = timing_config.n_steps_list[0]
    print(f"Warm-up ({timing_config.n_warmup} run, n_steps={config['n_steps']})...")
    for _ in range(timing_config.n_warmup):
        _ = pl.immunize(image, image_mask, immunization_mdl, config)
        _sync_cuda()

    # --- Misura per ciascun n_steps ---
    per_nsteps: dict[str, dict] = {}
    for n_steps in timing_config.n_steps_list:
        config["n_steps"] = n_steps
        print(f"\nn_steps = {n_steps}")
        times = _time_n_runs(image, image_mask, immunization_mdl, config, timing_config.n_runs)

        per_nsteps[str(n_steps)] = {
            "n_steps": n_steps,
            "n_runs": timing_config.n_runs,
            "mean_s": float(times.mean()),
            "std_s": float(times.std()),
            "min_s": float(times.min()),
            "max_s": float(times.max()),
            "median_s": float(np.median(times)),
            "all_runs_s": times.tolist(),
        }

    result = {
        "model_attack": config["model_attack"],
        "n_warmup": timing_config.n_warmup,
        "per_nsteps": per_nsteps,
    }

    if timing_config.output_path is not None:
        timing_config.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(timing_config.output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    return result


def print_report(result: dict) -> None:
    print("\n=== Tempo di inferenza - immunize() su singola immagine, per n_steps ===")
    print(f"Modello di attacco: {result['model_attack']}\n")
    print(f"{'n_steps':>8} | {'media (s)':>10} | {'std (s)':>8} | {'min (s)':>8} | {'max (s)':>8}")
    print("-" * 55)
    for n_steps_str, stats in result["per_nsteps"].items():
        print(
            f"{stats['n_steps']:>8} | {stats['mean_s']:>10.4f} | {stats['std_s']:>8.4f} "
            f"| {stats['min_s']:>8.4f} | {stats['max_s']:>8.4f}"
        )


if __name__ == "__main__":
    config = pl.get_config()
    # Forziamo il caricamento di un solo sample (non l'intero dataset).
    config["run_full_dataset"] = False

    print("Caricamento modelli e checkpoint...")
    attack_model, immunization_mdl = pl.load_models(config)

    print("Caricamento sample...")
    image, image_mask, _ = pl.load_sample(config)

    timing_config = TimingConfig(
        n_steps_list=[50, 100, 200, 300, 400, 500],
        n_warmup=1,
        n_runs=3,
    )

    result = time_immunize_vs_nsteps(image, image_mask, immunization_mdl, config, timing_config)
    print_report(result)