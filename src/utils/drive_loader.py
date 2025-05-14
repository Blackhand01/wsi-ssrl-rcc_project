# src/utils/drive_loader.py

from pathlib import Path
import openslide
from typing import List

# Sostituisci con il tuo path montato
DEFAULT_DRIVE_ROOT = (
    Path.home()
    / "Library"
    / "CloudStorage"
    / "Google Drive-stefano2001roy@gmail.com"
    / "Il mio Drive"
    / "Colab_Notebooks"
    / "RCC_WSIs"
)

def list_slides(
    root: Path = DEFAULT_DRIVE_ROOT, patterns: List[str] = ("*.scn", "*.svs")
) -> List[Path]:
    """Return all WSI files (scn, svs) under the given root directory."""
    slides = []
    for pat in patterns:
        slides.extend(root.rglob(pat))
    return slides

def open_slide(path: Path) -> openslide.OpenSlide:
    """Open and return an OpenSlide object for a given slide path."""
    return openslide.OpenSlide(str(path))
