"""Libreria di forme candidate per lo scostamento EFS(%) in funzione dell'EFD
accumulato (%), condivisa da decadimento per bucket e confronto race plan."""
import numpy as np

# Soglie minime di file per abilitare modelli con più parametri (evita overfitting)
MIN_FILES_FOR_QUADRATIC = 10
MIN_FILES_FOR_CUBIC = 15


# ---------------------------------------------------------------------------
# Model fitting: try several candidate shapes, pick the best by adjusted R²
# ---------------------------------------------------------------------------
def fit_candidates(x: np.ndarray, y: np.ndarray, n_files: int) -> dict:
    """
    Fit several candidate forms of deviation_pct(x) where x = EFD% accumulata
    nel file (0-100), relativa all'EFD totale del file stesso.

    n_files = numero di file indipendenti a supporto (non righe/bin): usato
    per limitare i gradi di libertà del modello quando il campione è piccolo,
    dato che i bin di uno stesso file NON sono osservazioni indipendenti.
    """
    n = len(x)
    ss_tot = np.sum((y - y.mean()) ** 2)

    def r2_of(pred):
        ss_res = np.sum((y - pred) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    def adj_r2_of(r2, k):
        if n - k - 1 <= 0 or np.isnan(r2):
            return np.nan
        return 1 - (1 - r2) * (n - 1) / (n - k - 1)

    candidates = {}

    # Linear: y = a + b*x  (k=2, sempre incluso)
    c = np.polyfit(x, y, 1)
    pred = np.polyval(c, x)
    r2 = r2_of(pred)
    candidates["Lineare"] = {
        "coefs": c, "k": 2, "r2": r2, "adj_r2": adj_r2_of(r2, 2),
        "predict": lambda xx, c=c: np.polyval(c, xx),
        "equation": rf"\Delta EFS(\%) = {c[1]:.3f} + {c[0]:.4f} \cdot p",
    }

    # Logaritmica: y = a + b*ln(x+1)  (k=2, sempre incluso)
    x_log = np.log1p(x)
    c = np.polyfit(x_log, y, 1)
    pred = np.polyval(c, x_log)
    r2 = r2_of(pred)
    candidates["Logaritmica"] = {
        "coefs": c, "k": 2, "r2": r2, "adj_r2": adj_r2_of(r2, 2),
        "predict": lambda xx, c=c: np.polyval(c, np.log1p(xx)),
        "equation": rf"\Delta EFS(\%) = {c[1]:.3f} + {c[0]:.4f} \cdot \ln(p+1)",
    }

    # Radice quadrata: y = a + b*sqrt(x)  (k=2, sempre incluso)
    x_sqrt = np.sqrt(x)
    c = np.polyfit(x_sqrt, y, 1)
    pred = np.polyval(c, x_sqrt)
    r2 = r2_of(pred)
    candidates["Radice quadrata"] = {
        "coefs": c, "k": 2, "r2": r2, "adj_r2": adj_r2_of(r2, 2),
        "predict": lambda xx, c=c: np.polyval(c, np.sqrt(xx)),
        "equation": rf"\Delta EFS(\%) = {c[1]:.3f} + {c[0]:.4f} \cdot \sqrt{{p}}",
    }

    # Quadratica: y = a + b*x + c*x^2  (k=3, richiede abbastanza file)
    if n_files >= MIN_FILES_FOR_QUADRATIC:
        c = np.polyfit(x, y, 2)
        pred = np.polyval(c, x)
        r2 = r2_of(pred)
        candidates["Quadratica"] = {
            "coefs": c, "k": 3, "r2": r2, "adj_r2": adj_r2_of(r2, 3),
            "predict": lambda xx, c=c: np.polyval(c, xx),
            "equation": rf"\Delta EFS(\%) = {c[2]:.3f} + {c[1]:.4f} \cdot p + {c[0]:.5f} \cdot p^2",
        }

    # Cubic: y = a + b*x + c*x^2 + d*x^3  (k=4, richiede ancora più file)
    if n_files >= MIN_FILES_FOR_CUBIC:
        c = np.polyfit(x, y, 3)
        pred = np.polyval(c, x)
        r2 = r2_of(pred)
        candidates["Cubica"] = {
            "coefs": c, "k": 4, "r2": r2, "adj_r2": adj_r2_of(r2, 4),
            "predict": lambda xx, c=c: np.polyval(c, xx),
            "equation": (rf"\Delta EFS(\%) = {c[3]:.3f} + {c[2]:.4f} \cdot p + "
                         rf"{c[1]:.5f} \cdot p^2 + {c[0]:.6f} \cdot p^3"),
        }

    return candidates


def best_candidate(candidates: dict):
    return max(candidates, key=lambda k: (candidates[k]["adj_r2"]
                                          if not np.isnan(candidates[k]["adj_r2"]) else -np.inf))