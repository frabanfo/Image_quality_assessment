"""
Visualizza le curve di training restituite da train() / train_weighted().

Uso da codice:
    from src.plot_training import plot_training_curves, save_history
    h1, h2a, h2b = train(...)
    plot_training_curves([h1, h2a, h2b], save_path="results/model_a/curves.png")
    save_history([h1, h2a, h2b], "results/model_a/history.json")

Uso da CLI (su history salvata):
    python -m src.plot_training --history results/model_a/history.json \
                                --output  results/model_a/curves.png \
                                --title   "Model A — MSE"
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Serializzazione history
# ---------------------------------------------------------------------------

def save_history(histories, path: str) -> None:
    """Salva una lista di Keras History (o dict) come JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = [h.history if hasattr(h, "history") else h for h in histories]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"History salvata in: {path}")


def load_history(path: str) -> list[dict]:
    """Carica una history salvata con save_history()."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

_DEFAULT_LABELS = ["Phase 1", "Phase 2a", "Phase 2b"]


def plot_training_curves(
    histories,
    save_path: str,
    title: str = "",
    phase_labels: list[str] | None = None,
) -> None:
    """
    Due pannelli verticali:
      - in alto : loss (train + val) per tutte le epoche
      - in basso : val_srcc per tutte le epoche

    Linee verticali tratteggiate marcano i confini tra le fasi.

    histories : lista di Keras History o dict (output di load_history()).
    save_path : path dove salvare il PNG.
    """
    if phase_labels is None:
        phase_labels = _DEFAULT_LABELS[:len(histories)]

    # Estrai dict se sono oggetti History Keras
    dicts = [h.history if hasattr(h, "history") else h for h in histories]

    def _concat(key):
        return [v for d in dicts for v in d.get(key, [])]

    loss       = _concat("loss")
    val_loss   = _concat("val_loss")
    val_srcc   = _concat("val_srcc")
    epochs_end = []  # epoche di fine fase (per le linee verticali)
    cursor = 0
    for d in dicts[:-1]:
        cursor += len(d.get("loss", []))
        epochs_end.append(cursor)

    total = len(loss)
    xs = list(range(1, total + 1))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    fig.suptitle(title, fontsize=12, fontweight="bold")

    # --- Loss ---
    ax1.plot(xs, loss,     label="train loss", color="steelblue",  lw=1.5)
    if val_loss:
        ax1.plot(xs, val_loss, label="val loss",   color="darkorange", lw=1.5, ls="--")
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # --- SRCC ---
    if val_srcc:
        ax2.plot(xs[:len(val_srcc)], val_srcc, color="mediumseagreen", lw=1.5, label="val SRCC")
        ax2.set_ylabel("Val SRCC", fontsize=10)
        ax2.set_ylim(max(0, min(val_srcc) - 0.05), min(1, max(val_srcc) + 0.05))
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)

    ax2.set_xlabel("Epoch", fontsize=10)

    # Linee di separazione fasi
    for i, ep in enumerate(epochs_end):
        label = f"end {phase_labels[i]}" if i < len(phase_labels) else ""
        for ax in (ax1, ax2):
            ax.axvline(ep + 0.5, color="gray", lw=1.0, ls=":", alpha=0.7, label=label)

    # Annotazioni fasi sull'asse superiore
    starts = [0] + [e for e in epochs_end]
    ends   = epochs_end + [total]
    for i, (s, e) in enumerate(zip(starts, ends)):
        mid = (s + e) / 2 + 0.5
        lbl = phase_labels[i] if i < len(phase_labels) else f"Ph {i+1}"
        ax1.text(mid, ax1.get_ylim()[1], lbl, ha="center", va="bottom",
                 fontsize=8, color="gray")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Curve di training salvate in: {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Plot training curves from saved history.")
    parser.add_argument("--history", required=True,
                        help="Path al file history.json (salvato con save_history())")
    parser.add_argument("--output", required=True,
                        help="Path di output per il PNG")
    parser.add_argument("--title", default="Training curves",
                        help="Titolo del grafico")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    histories = load_history(args.history)
    plot_training_curves(histories, save_path=args.output, title=args.title)
