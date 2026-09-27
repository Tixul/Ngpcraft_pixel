"""Image -> NGPC sprite pipeline (v1).

Chaîne : crop optionnel -> ajustements -> détourage (GrabCut) -> downscale
-> quantize 3 couleurs + transparent (snap RGB444).
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


try:
    import mediapipe as mp  # type: ignore
    _HAS_MEDIAPIPE = True
except ImportError:
    _HAS_MEDIAPIPE = False


# rembg : matting neural SOTA, remplace GrabCut par une segmentation de qualité
# nettement supérieure (cheveux fins, vêtements translucides, fonds complexes).
# ~170 MB au premier run (téléchargement auto du modèle BRIA-RMBG-1.4 Apache 2.0).
try:
    from rembg import remove as _rembg_remove, new_session as _rembg_new_session  # type: ignore
    _HAS_REMBG = True
    _REMBG_SESSION = None
except ImportError:
    _HAS_REMBG = False
    _REMBG_SESSION = None


# ONNX Runtime (déjà requis par rembg). Utilisé aussi pour Real-ESRGAN super-res.
try:
    import onnxruntime as _ort  # type: ignore
    _HAS_ONNXRUNTIME = True
except ImportError:
    _HAS_ONNXRUNTIME = False


def _get_rembg_session():
    """Lazy-load la session rembg au 1er usage (évite le startup cost + download)."""
    global _REMBG_SESSION
    if not _HAS_REMBG:
        return None
    if _REMBG_SESSION is None:
        try:
            _REMBG_SESSION = _rembg_new_session("u2net")  # modèle par défaut
        except Exception:
            return None
    return _REMBG_SESSION


# Seuil post-downscale : quand le downscale Lanczos moyenne les pixels de bord
# (ex : mèches de cheveux fines 1-2 px), l'alpha résultant peut descendre à 50-100.
# Un seuil trop strict (128) efface ces détails. 96 est un compromis qui préserve
# les cheveux fins sans introduire de ghost pixels post-downscale.
ALPHA_OPAQUE_THRESHOLD = 96


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGBA")


def apply_adjustments(img: Image.Image, contrast=0, brightness=0, saturation=0) -> Image.Image:
    if contrast == 0 and brightness == 0 and saturation == 0:
        return img
    rgb = img.convert("RGB")
    if brightness != 0:
        rgb = ImageEnhance.Brightness(rgb).enhance(1.0 + brightness / 100.0)
    if contrast != 0:
        rgb = ImageEnhance.Contrast(rgb).enhance(1.0 + contrast / 100.0)
    if saturation != 0:
        rgb = ImageEnhance.Color(rgb).enhance(1.0 + saturation / 100.0)
    r, g, b = rgb.split()
    a = img.split()[3]
    return Image.merge("RGBA", (r, g, b, a))


# ---------------------------------------------------------------------------
# Face detection (MediaPipe) — 4 MB model, Apache 2.0, 100% local après install.
# Optionnel : si mediapipe absent, les options face_crop retournent None silencieusement.
# ---------------------------------------------------------------------------


def detect_face_bbox(img_pil: Image.Image,
                     expand_sides: float = 1.6,
                     expand_above: float = 0.6,
                     expand_below: float = 2.5,
                     min_confidence: float = 0.5) -> tuple[int, int, int, int] | None:
    """Détecte le visage principal via MediaPipe Face Detection (modèle "full range"),
    élargit la bbox pour inclure cheveux (above) et buste (below) et la marge latérale.

    Retourne (l, t, r, b) en coordonnées pixel de img_pil, ou None si :
    - mediapipe absent
    - aucun visage détecté
    - erreur d'inférence

    Paramètres typiques pour un portrait bust-shot :
      expand_sides=1.6 (6/10 de face_w de chaque côté)
      expand_above=0.6 (60% de face_h au-dessus, pour les cheveux)
      expand_below=2.5 (250% de face_h en dessous, pour le torse)
    """
    if not _HAS_MEDIAPIPE:
        return None
    arr = np.array(img_pil.convert("RGB"))
    h, w = arr.shape[:2]
    if w < 20 or h < 20:
        return None
    try:
        with mp.solutions.face_detection.FaceDetection(
            model_selection=1,  # 1 = full-range (faces jusqu'à ~5m), 0 = short-range
            min_detection_confidence=min_confidence,
        ) as detector:
            results = detector.process(arr)
    except Exception:
        return None
    if not getattr(results, "detections", None):
        return None
    # Prend la détection de plus haute confiance
    best = max(
        results.detections,
        key=lambda d: (d.score[0] if d.score else 0.0),
    )
    rel = best.location_data.relative_bounding_box
    fx = rel.xmin * w
    fy = rel.ymin * h
    fw = rel.width * w
    fh = rel.height * h
    cx = fx + fw / 2
    new_w = fw * expand_sides
    l = max(0, int(round(cx - new_w / 2)))
    r = min(w, int(round(cx + new_w / 2)))
    t = max(0, int(round(fy - fh * expand_above)))
    b = min(h, int(round(fy + fh + fh * expand_below)))
    if r - l < 8 or b - t < 8:
        return None
    return (l, t, r, b)


def detect_eye_positions_rel(img_pil: Image.Image,
                             min_confidence: float = 0.5) -> list[tuple[float, float]]:
    """Détecte les centres des iris gauche/droit via MediaPipe Face Mesh (avec
    `refine_landmarks=True` pour l'iris sub-millimétrique). Retourne une liste de
    tuples (x_rel, y_rel) en [0,1], coords relatives de l'image source.
    Liste vide si mediapipe absent, image trop petite, ou pas de visage."""
    if not _HAS_MEDIAPIPE:
        return []
    arr = np.array(img_pil.convert("RGB"))
    h, w = arr.shape[:2]
    if w < 64 or h < 64:
        return []
    try:
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=min_confidence,
        ) as mesh:
            results = mesh.process(arr)
    except Exception:
        return []
    if not getattr(results, "multi_face_landmarks", None):
        return []
    lm = results.multi_face_landmarks[0].landmark
    # Indices : 468 = iris droit, 473 = iris gauche (avec refine_landmarks=True)
    eyes: list[tuple[float, float]] = []
    for idx in (468, 473):
        try:
            p = lm[idx]
            eyes.append((float(p.x), float(p.y)))
        except IndexError:
            pass
    return eyes


def apply_eye_preservation(output_rgba: Image.Image,
                           source_img: Image.Image,
                           effective_crop: tuple[int, int, int, int] | None = None
                           ) -> Image.Image:
    """Post-quantize : force la couleur la plus sombre de la palette aux coords
    estimées des yeux (détectés via MediaPipe Face Mesh dans la source).

    Gère le mapping source → output en prenant en compte `effective_crop` (la bbox
    utilisée pour le crop avant downscale). Ne gère PAS encore la distorsion rig
    (tighten bands, rescale non-uniforme) — dans ces cas l'approximation reste
    acceptable car la tête dans le output est généralement centrée en haut.
    """
    eyes_rel = detect_eye_positions_rel(source_img)
    if not eyes_rel:
        return output_rgba
    W, H = source_img.size
    # Re-normaliser dans les coords du crop si présent
    if effective_crop is not None:
        l, t, r, b = effective_crop
        cw, ch = r - l, b - t
        if cw <= 0 or ch <= 0:
            return output_rgba
        remapped = []
        for (ex, ey) in eyes_rel:
            sx, sy = ex * W, ey * H
            if l <= sx < r and t <= sy < b:
                remapped.append(((sx - l) / cw, (sy - t) / ch))
        eyes_rel = remapped
        if not eyes_rel:
            return output_rgba

    out_arr = np.array(output_rgba.convert("RGBA"))
    oh, ow = out_arr.shape[:2]
    alpha = out_arr[..., 3]
    opaque = alpha >= 128
    if not opaque.any():
        return output_rgba

    # Couleur la plus sombre de la palette actuelle
    rgb = out_arr[..., :3]
    op_pixels = rgb[opaque]
    packed = (op_pixels[:, 0].astype(np.uint32) << 16) | \
             (op_pixels[:, 1].astype(np.uint32) << 8) | op_pixels[:, 2]
    uniq = np.unique(packed)
    best_rgb = None
    best_l = float("inf")
    for p in uniq:
        pi = int(p)
        rr = (pi >> 16) & 0xFF
        gg = (pi >> 8) & 0xFF
        bb = pi & 0xFF
        L = 0.299 * rr + 0.587 * gg + 0.114 * bb
        if L < best_l:
            best_l = L
            best_rgb = (rr, gg, bb)
    if best_rgb is None:
        return output_rgba

    for (ex, ey) in eyes_rel:
        px = int(round(ex * ow))
        py = int(round(ey * oh))
        if 0 <= px < ow and 0 <= py < oh and opaque[py, px]:
            out_arr[py, px, 0] = best_rgb[0]
            out_arr[py, px, 1] = best_rgb[1]
            out_arr[py, px, 2] = best_rgb[2]
            out_arr[py, px, 3] = 255
    return Image.fromarray(out_arr, "RGBA")


def remove_background_grabcut(img_pil: Image.Image, margin_ratio: float = 0.08,
                              user_mask: np.ndarray | None = None,
                              iters: int = 5,
                              margin_top_ratio: float | None = None) -> Image.Image:
    """Détourage auto ou semi-auto.

    Si user_mask fourni (np.uint8, même taille que img_pil) :
        0 = pas d'indice, 1 = garder (foreground), 2 = enlever (background)
        Le GrabCut démarre avec ces indices (GC_INIT_WITH_MASK), + une zone
        présomptive au centre pour amorcer la segmentation.
    Sinon : auto avec rect centré.

    `margin_top_ratio` : marge haute spécifique. Si None, = margin_ratio (8% standard).
    Descendre à 0.02 aide à préserver les cheveux qui touchent le bord haut, MAIS
    peut faire classer les zones claires du sujet comme fond si la photo a elle-
    même un fond clair (modèle bg appris sur les bords = blancs → supprime le
    blanc du sujet). À utiliser sélectivement.
    """
    if margin_top_ratio is None:
        margin_top_ratio = margin_ratio
    if not _HAS_CV2:
        return img_pil.convert("RGBA")
    arr = np.array(img_pil.convert("RGB"))
    h, w = arr.shape[:2]
    if w < 20 or h < 20:
        return img_pil.convert("RGBA")

    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    has_hints = user_mask is not None and user_mask.any()

    # Marges : top plus petit (cheveux préservés), sides/bottom = margin_ratio standard
    mx = max(1, int(w * margin_ratio))
    my_bottom = max(1, int(h * margin_ratio))
    my_top = max(1, int(h * margin_top_ratio))

    if has_hints:
        mask[:] = cv2.GC_PR_BGD
        mask[my_top:h - my_bottom, mx:w - mx] = cv2.GC_PR_FGD
        mask[user_mask == 1] = cv2.GC_FGD
        mask[user_mask == 2] = cv2.GC_BGD
        try:
            cv2.grabCut(arr, mask, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            return img_pil.convert("RGBA")
    else:
        rect = (mx, my_top, max(2, w - 2 * mx), max(2, h - my_top - my_bottom))
        try:
            cv2.grabCut(arr, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
        except cv2.error:
            return img_pil.convert("RGBA")

    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    out = np.array(img_pil.convert("RGBA"))
    out[:, :, 3] = fg
    return Image.fromarray(out, "RGBA")


# ---------------------------------------------------------------------------
# Super-resolution : Real-ESRGAN via ONNX Runtime (~65 MB model, Apache 2.0).
# Upscale 4× la source avant le downscale pipeline → gros saut qualité sur
# sources basse résolution (photos moyennes, screenshots, artworks ≤1000 px).
#
# Modèle téléchargé au 1er usage dans ~/.cache/ngpcraft_pixel/.
# Fallback silencieux (retourne l'image inchangée) si onnxruntime absent ou
# téléchargement échoué.
# ---------------------------------------------------------------------------

import os as _os
from pathlib import Path as _Path
import urllib.request as _urllib_request

REALESRGAN_MODELS = {
    # URL communauté HuggingFace vérifiées (HEAD 200, taille ~64 MB).
    # Alt : https://huggingface.co/crj/dl-ws/resolve/main/real_esrgan_x4.onnx (66.2 MB)
    # Alt : https://huggingface.co/tidus2102/Real-ESRGAN/resolve/main/Real-ESRGAN_x2plus.onnx (64.0 MB)
    "x4plus": {
        "url": "https://huggingface.co/AXERA-TECH/Real-ESRGAN/resolve/main/onnx/realesrgan-x4.onnx",
        "scale": 4,
        "desc": "x4 général — photos, scènes, artworks",
    },
    # Note : pas de variante 'anime' ONNX librement dispo. Le modèle x4 général
    # marche aussi bien sur illustrations. Variantes spécialisées à convertir
    # depuis les .pth officiels si besoin.
}

_SUPERRES_SESSION: "_ort.InferenceSession | None" = None
_SUPERRES_CURRENT_MODEL: str | None = None


def _cache_dir() -> _Path:
    override = _os.environ.get("NGPCRAFT_PIXEL_CACHE")
    if override:
        p = _Path(override)
    else:
        p = _Path.home() / ".cache" / "ngpcraft_pixel"
    p.mkdir(parents=True, exist_ok=True)
    return p


def superres_model_path(model_name: str) -> _Path:
    """Chemin local du fichier .onnx pour un modèle Real-ESRGAN donné."""
    return _cache_dir() / f"RealESRGAN_{model_name}.onnx"


def superres_model_is_cached(model_name: str) -> bool:
    p = superres_model_path(model_name)
    return p.exists() and p.stat().st_size > 1_000_000


def rembg_model_dir() -> _Path:
    """Répertoire où rembg stocke ses modèles (typiquement ~/.u2net)."""
    # rembg utilise U2NET_HOME ou ~/.u2net
    override = _os.environ.get("U2NET_HOME")
    if override:
        return _Path(override)
    return _Path.home() / ".u2net"


def rembg_model_is_cached(model_name: str = "u2net") -> bool:
    """True si le .onnx rembg est déjà sur disque."""
    p = rembg_model_dir() / f"{model_name}.onnx"
    return p.exists() and p.stat().st_size > 1_000_000


def collect_ml_status() -> dict:
    """Retourne un dict structurant l'état de toutes les deps / modèles ML.
    Utilisé par l'UI (Gestionnaire de modèles) pour afficher le statut."""
    status = {
        "mediapipe": {
            "lib_name": "mediapipe",
            "has_lib": _HAS_MEDIAPIPE,
            "bundled_models": _HAS_MEDIAPIPE,  # embarqués dans le pip package
            "models": [],  # rien à télécharger, bundled
        },
        "onnxruntime": {
            "lib_name": "onnxruntime",
            "has_lib": _HAS_ONNXRUNTIME,
            "bundled_models": True,
            "models": [],
        },
        "rembg": {
            "lib_name": "rembg",
            "has_lib": _HAS_REMBG,
            "bundled_models": False,
            "models": [
                {
                    "name": "u2net",
                    "desc": "U2Net matting (~170 MB)",
                    "path": str(rembg_model_dir() / "u2net.onnx"),
                    "is_cached": rembg_model_is_cached("u2net"),
                    "url": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
                },
            ],
        },
        "realesrgan": {
            "lib_name": "onnxruntime",
            "has_lib": _HAS_ONNXRUNTIME,
            "bundled_models": False,
            "models": [
                {
                    "name": name,
                    "desc": info["desc"],
                    "path": str(superres_model_path(name)),
                    "is_cached": superres_model_is_cached(name),
                    "url": info["url"],
                }
                for name, info in REALESRGAN_MODELS.items()
            ],
        },
    }
    return status


def clear_model_cache(include_rembg: bool = False) -> int:
    """Supprime tous les modèles en cache (NgpCraft_pixel). Si include_rembg=True,
    supprime aussi ceux de rembg (~/.u2net/). Retourne le nombre de fichiers supprimés."""
    n = 0
    cache = _cache_dir()
    if cache.exists():
        for f in cache.glob("*.onnx"):
            try:
                f.unlink()
                n += 1
            except Exception:
                pass
    if include_rembg:
        rd = rembg_model_dir()
        if rd.exists():
            for f in rd.glob("*.onnx"):
                try:
                    f.unlink()
                    n += 1
                except Exception:
                    pass
    return n


def _ensure_superres_model(model_name: str = "x4plus") -> str | None:
    """Télécharge le modèle Real-ESRGAN ONNX dans le cache s'il n'est pas déjà là.
    Retourne le chemin local ou None si téléchargement échoué / modèle inconnu."""
    if not _HAS_ONNXRUNTIME:
        return None
    info = REALESRGAN_MODELS.get(model_name)
    if info is None:
        return None
    path = _cache_dir() / f"RealESRGAN_{model_name}.onnx"
    if not path.exists():
        try:
            _urllib_request.urlretrieve(info["url"], str(path))
        except Exception:
            return None
    if not path.exists() or path.stat().st_size < 1_000_000:
        # Fichier corrompu / tronqué — supprime et laisse retry
        try:
            path.unlink()
        except Exception:
            pass
        return None
    return str(path)


def _get_superres_session(model_name: str = "x4plus"):
    global _SUPERRES_SESSION, _SUPERRES_CURRENT_MODEL
    if _SUPERRES_SESSION is not None and _SUPERRES_CURRENT_MODEL == model_name:
        return _SUPERRES_SESSION
    path = _ensure_superres_model(model_name)
    if path is None:
        return None
    try:
        _SUPERRES_SESSION = _ort.InferenceSession(
            path, providers=["CPUExecutionProvider"]
        )
        _SUPERRES_CURRENT_MODEL = model_name
    except Exception:
        _SUPERRES_SESSION = None
        _SUPERRES_CURRENT_MODEL = None
        return None
    return _SUPERRES_SESSION


def upscale_superres(img_pil: Image.Image,
                     model_name: str = "x4plus",
                     max_input_side: int = 1000) -> Image.Image:
    """Super-resolution Real-ESRGAN ×4. Ne tourne que si la plus grande dimension
    de l'entrée est ≤ max_input_side (garde-fou mémoire / perf). Fallback silencieux
    = image inchangée."""
    if not _HAS_ONNXRUNTIME:
        return img_pil
    w, h = img_pil.size
    if max(w, h) > max_input_side:
        return img_pil  # pas la peine, source déjà grande
    session = _get_superres_session(model_name)
    if session is None:
        return img_pil
    arr = np.array(img_pil.convert("RGB"))
    tensor = arr.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    try:
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: tensor})[0]
    except Exception:
        return img_pil
    up_arr = (output[0].transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)
    if img_pil.mode == "RGBA":
        # Upscale l'alpha avec Lanczos (pas de perte notable)
        alpha_src = img_pil.split()[3]
        alpha_up = alpha_src.resize(
            (up_arr.shape[1], up_arr.shape[0]), Image.LANCZOS
        )
        full = np.dstack([up_arr, np.array(alpha_up)])
        return Image.fromarray(full, "RGBA")
    return Image.fromarray(up_arr, "RGB")


def remove_background_rembg(img_pil: Image.Image) -> Image.Image:
    """Détourage via rembg (matting neural). Bien meilleur que GrabCut sur cheveux
    fins, vêtements translucides, fonds complexes. Retourne RGBA avec alpha produit
    par le modèle (valeurs 0–255, pas binaire). Si rembg absent ou erreur, fallback
    GrabCut."""
    sess = _get_rembg_session()
    if sess is None:
        return remove_background_grabcut(img_pil)
    try:
        out = _rembg_remove(img_pil, session=sess)
        # rembg renvoie parfois en mode RGB avec alpha dans un canal séparé, on force RGBA
        return out.convert("RGBA")
    except Exception:
        return remove_background_grabcut(img_pil)


def bilateral_preprocess(img_pil: Image.Image, d: int = 7,
                         sigma_color: int = 60, sigma_space: int = 60) -> Image.Image:
    """Bilateral filter : aplatit les zones uniformes tout en gardant les bords.
    Gros gain de lisibilité au downscale (approche inspirée de Pyxelate)."""
    if not _HAS_CV2:
        return img_pil
    arr = np.array(img_pil.convert("RGBA"))
    rgb = np.ascontiguousarray(arr[:, :, :3])
    try:
        filt = cv2.bilateralFilter(rgb, d, sigma_color, sigma_space)
    except cv2.error:
        return img_pil
    arr[:, :, :3] = filt
    return Image.fromarray(arr, "RGBA")


def sharpen_edges(img_pil: Image.Image, radius: float = 1.5,
                  percent: int = 130, threshold: int = 3) -> Image.Image:
    """Unsharp mask — accentue les contours importants avant le downscale."""
    rgb = img_pil.convert("RGB").filter(
        ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold)
    )
    r, g, b = rgb.split()
    a = img_pil.convert("RGBA").split()[3]
    return Image.merge("RGBA", (r, g, b, a))


def downscale(img: Image.Image, target_w: int, target_h: int,
              multi_pass: bool = True) -> Image.Image:
    """Downscale Lanczos. Si multi_pass=True et source >>> target, halving itératif
    avant le resize final pour mieux préserver les détails."""
    if multi_pass:
        w, h = img.size
        while w > target_w * 2 and h > target_h * 2:
            img = img.resize((w // 2, h // 2), Image.LANCZOS)
            w, h = img.size
    return img.resize((target_w, target_h), Image.LANCZOS)


def auto_crop_subject(img_rgba: Image.Image, padding_ratio: float = 0.05) -> Image.Image:
    """Crop sur la bbox opaque + padding (ratio de la plus grande dimension).
    Puis pad transparent pour garder la même aspect ratio que la source originale avant
    l'étape de downscale qui stretche. Retourne image recadrée."""
    arr = np.array(img_rgba)
    if arr.shape[2] < 4:
        return img_rgba
    opaque = arr[..., 3] >= 128
    if not opaque.any():
        return img_rgba
    ys, xs = np.where(opaque)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    max_dim = max(y1 - y0, x1 - x0)
    pad = max(1, int(round(max_dim * padding_ratio)))
    y0 = max(0, y0 - pad); y1 = min(arr.shape[0], y1 + pad)
    x0 = max(0, x0 - pad); x1 = min(arr.shape[1], x1 + pad)
    return img_rgba.crop((x0, y0, x1, y1))


def pad_to_aspect(img_rgba: Image.Image, target_ratio: float) -> Image.Image:
    """Ajoute du transparent pour que l'image ait target_ratio (= target_w / target_h)."""
    w, h = img_rgba.size
    if h == 0 or w == 0:
        return img_rgba
    cur = w / h
    if abs(cur - target_ratio) < 1e-3:
        return img_rgba
    if cur > target_ratio:
        new_h = int(round(w / target_ratio))
        pad = (new_h - h) // 2
        out = Image.new("RGBA", (w, new_h), (0, 0, 0, 0))
        out.paste(img_rgba, (0, pad))
    else:
        new_w = int(round(h * target_ratio))
        pad = (new_w - w) // 2
        out = Image.new("RGBA", (new_w, h), (0, 0, 0, 0))
        out.paste(img_rgba, (pad, 0))
    return out


def erode_alpha(img_rgba: Image.Image, px: int = 1) -> Image.Image:
    """Érode 1px (ou plus) l'alpha opaque — tue le halo GrabCut avant downscale."""
    if px <= 0 or not _HAS_CV2:
        return img_rgba
    arr = np.array(img_rgba.convert("RGBA"))
    alpha = arr[..., 3]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * px + 1, 2 * px + 1))
    arr[..., 3] = cv2.erode(alpha, kernel, iterations=1)
    return Image.fromarray(arr, "RGBA")


# ---------------------------------------------------------------------------
# Espace LAB + k-means — pour une quantize perceptuelle plus juste
# ---------------------------------------------------------------------------

_LAB_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
_SRGB_TO_XYZ = np.array([
    [0.4124, 0.3576, 0.1805],
    [0.2126, 0.7152, 0.0722],
    [0.0193, 0.1192, 0.9505],
], dtype=np.float32)


def rgb_to_lab(rgb_uint8: np.ndarray) -> np.ndarray:
    """(..., 3) uint8 → (..., 3) float32 LAB via sRGB→XYZ→LAB (D65)."""
    rgb_f = rgb_uint8.astype(np.float32) / 255.0
    lin = np.where(rgb_f <= 0.04045, rgb_f / 12.92, ((rgb_f + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _SRGB_TO_XYZ.T
    xyz_n = xyz / _LAB_WHITE
    f = np.where(xyz_n > 0.008856, np.cbrt(xyz_n), 7.787 * xyz_n + 16.0 / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


_RGB444_GRID_RGB = np.array(
    [(r * 17, g * 17, b * 17) for r in range(16) for g in range(16) for b in range(16)],
    dtype=np.uint8,
)
_RGB444_GRID_LAB: np.ndarray | None = None


def _get_rgb444_lab_grid() -> np.ndarray:
    global _RGB444_GRID_LAB
    if _RGB444_GRID_LAB is None:
        _RGB444_GRID_LAB = rgb_to_lab(_RGB444_GRID_RGB)
    return _RGB444_GRID_LAB


def _kmeans(points: np.ndarray, k: int, n_iters: int = 25, seed: int = 42,
            weights: np.ndarray | None = None) -> np.ndarray:
    """k-means++ init, assignations vectorisées. points (N, D) float. Retourne (k, D).

    `weights` : si fourni, k-means++ init pondéré + moyenne pondérée à chaque
    itération. Permet de biaiser les centres vers des pixels importants (typiquement
    la zone tête pour un perso, via un poids 3× sur le top 45%)."""
    rng = np.random.default_rng(seed)
    n = len(points)
    if n <= k:
        return points.astype(np.float32)
    if weights is None:
        weights = np.ones(n, dtype=np.float32)
    else:
        weights = weights.astype(np.float32)
    # k-means++ init pondéré
    first_probs = weights / weights.sum()
    first = int(rng.choice(n, p=first_probs))
    centers = [points[first]]
    for _ in range(k - 1):
        stacked = np.stack(centers)
        d2 = ((points[:, None, :] - stacked[None, :, :]) ** 2).sum(-1)
        dmin = d2.min(axis=1) * weights
        total = float(dmin.sum())
        if total <= 0:
            centers.append(points[int(rng.integers(0, n))])
            continue
        probs = dmin / total
        centers.append(points[int(rng.choice(n, p=probs))])
    centers = np.stack(centers).astype(np.float32)
    for _ in range(n_iters):
        d2 = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        assign = np.argmin(d2, axis=1)
        new = centers.copy()
        for i in range(k):
            m = assign == i
            if m.any():
                pts = points[m]
                w = weights[m]
                w_sum = float(w.sum())
                if w_sum > 0:
                    new[i] = (pts * w[:, None]).sum(axis=0) / w_sum
                else:
                    new[i] = pts.mean(axis=0)
        if np.allclose(centers, new, atol=0.3):
            centers = new
            break
        centers = new
    return centers


def _snap_lab_to_rgb444_unique(centers_lab: np.ndarray) -> list[tuple[int, int, int]]:
    """Pour chaque centre LAB, trouve le RGB444 grid le plus proche en évitant
    les doublons (diversité de palette)."""
    grid_lab = _get_rgb444_lab_grid()
    used: set[tuple[int, int, int]] = set()
    out: list[tuple[int, int, int]] = []
    for c in centers_lab:
        d2 = ((grid_lab - c) ** 2).sum(axis=1)
        order = np.argsort(d2)
        chosen = None
        for idx in order[:64]:
            rgb = tuple(int(v) for v in _RGB444_GRID_RGB[idx])
            if rgb not in used:
                used.add(rgb)
                chosen = rgb
                break
        if chosen is None:
            chosen = tuple(int(v) for v in _RGB444_GRID_RGB[order[0]])
        out.append(chosen)
    return out


# ---------------------------------------------------------------------------
# Choix de palette : median-cut (rapide) ou k-means LAB (meilleur perceptuel)
# ---------------------------------------------------------------------------


def _pick_palette_mediancut(rgb: np.ndarray, opaque: np.ndarray,
                            num_colors: int) -> list[tuple[int, int, int]]:
    has_trans = bool((~opaque).any())
    tmp = np.where(opaque[:, :, None], rgb, 0).astype(np.uint8)
    pil_tmp = Image.fromarray(tmp, "RGB")
    k = num_colors + (1 if has_trans else 0)
    quant = pil_tmp.quantize(colors=k, method=0, dither=0)
    pal = quant.getpalette()[: 3 * k]
    entries = [tuple(pal[i:i + 3]) for i in range(0, len(pal), 3)]
    if has_trans:
        d0 = [r * r + g * g + b * b for (r, g, b) in entries]
        drop = int(np.argmin(d0))
        entries = [e for i, e in enumerate(entries) if i != drop]
    return [_snap_rgb444(e) for e in entries]


def _pick_palette_kmeans(rgb: np.ndarray, opaque: np.ndarray,
                         num_colors: int,
                         weights_mask: np.ndarray | None = None) -> list[tuple[int, int, int]]:
    op_pixels = rgb[opaque]
    if len(op_pixels) == 0:
        return []
    op_weights = None
    if weights_mask is not None:
        op_weights = weights_mask[opaque].astype(np.float32)
    if len(op_pixels) > 20000:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(op_pixels), 20000, replace=False)
        op_pixels = op_pixels[idx]
        if op_weights is not None:
            op_weights = op_weights[idx]
    lab = rgb_to_lab(op_pixels)
    centers = _kmeans(lab, num_colors, weights=op_weights)
    return _snap_lab_to_rgb444_unique(centers)


def _pick_palette(rgb: np.ndarray, opaque: np.ndarray, num_colors: int,
                  method: str,
                  weights_mask: np.ndarray | None = None) -> list[tuple[int, int, int]]:
    if method == "kmeans":
        return _pick_palette_kmeans(rgb, opaque, num_colors, weights_mask=weights_mask)
    return _pick_palette_mediancut(rgb, opaque, num_colors)


# ---------------------------------------------------------------------------
# Application de la palette — flat (nearest LAB) ou dither (FS via PIL)
# ---------------------------------------------------------------------------


def _apply_palette_flat_lab(rgb: np.ndarray, snapped: list[tuple[int, int, int]]) -> np.ndarray:
    """Nearest-color en espace LAB — meilleur rendu des contours et anti-aliasés."""
    if not snapped:
        return np.zeros_like(rgb)
    snapped_arr = np.array(snapped, dtype=np.uint8)
    snapped_lab = rgb_to_lab(snapped_arr)
    rgb_lab = rgb_to_lab(rgb).reshape(-1, 3)
    d2 = ((rgb_lab[:, None, :] - snapped_lab[None, :, :]) ** 2).sum(axis=2)
    nearest = np.argmin(d2, axis=1)
    return snapped_arr[nearest].reshape(rgb.shape)


def _apply_palette_dithered(rgb: np.ndarray, opaque: np.ndarray,
                            snapped: list[tuple[int, int, int]]) -> np.ndarray:
    """Floyd-Steinberg via PIL.Image.quantize(palette=...), avec palette snappée imposée.
    Remplit le hors-opaque avec la moyenne pour éviter le bleed d'erreur."""
    if not snapped:
        return np.zeros_like(rgb)
    pal_img = Image.new("P", (16, 16))
    flat_pal: list[int] = []
    for c in snapped:
        flat_pal.extend(c)
    flat_pal += [0] * (768 - len(flat_pal))
    pal_img.putpalette(flat_pal)
    if (~opaque).any():
        avg = rgb[opaque].mean(axis=0).astype(np.uint8)
        tmp = rgb.copy()
        tmp[~opaque] = avg
    else:
        tmp = rgb
    pil_rgb = Image.fromarray(tmp, "RGB")
    quant = pil_rgb.quantize(palette=pal_img, dither=1)
    idx = np.array(quant)
    k = len(snapped)
    snapped_arr = np.array(snapped, dtype=np.uint8)
    idx = np.clip(idx, 0, k - 1)
    return snapped_arr[idx]


def _snap_rgb444(rgb):
    r, g, b = rgb
    return ((r >> 4) * 17, (g >> 4) * 17, (b >> 4) * 17)


def rgb_to_ngpc444_hex(r: int, g: int, b: int) -> str:
    """Format NGPC : (R>>4) | ((G>>4)<<4) | ((B>>4)<<8), 4 chiffres hex."""
    packed = ((r >> 4) & 0xF) | (((g >> 4) & 0xF) << 4) | (((b >> 4) & 0xF) << 8)
    return f"{packed:04X}"


def quantize_ngpc(img_rgba: Image.Image, num_colors: int = 3,
                  dither: bool = False, method: str = "mediancut",
                  face_priority: bool = False) -> Image.Image:
    """Réduit les pixels opaques à num_colors couleurs, snappées RGB444.
    method : "mediancut" (rapide) ou "kmeans" (LAB, meilleur perceptuel).
    Si dither=True, tramage Floyd-Steinberg.
    Si face_priority=True et method=kmeans, les pixels du top 45% reçoivent 3× le poids
    dans le clustering — la palette privilégie les teintes de peau/cheveux."""
    arr = np.array(img_rgba)
    if arr.shape[2] == 3:
        arr = np.dstack([arr, np.full(arr.shape[:2], 255, np.uint8)])
    alpha = arr[:, :, 3]
    # Seuil abaissé (96) : préserve les cheveux fins et anti-aliased edges dont
    # l'alpha tombe à 50-100 après downscale Lanczos. Un seuil strict (128)
    # effaçait ces détails sur les petits sprites.
    opaque = alpha >= ALPHA_OPAQUE_THRESHOLD

    if not opaque.any():
        out = arr.copy()
        out[:, :, 3] = 0
        return Image.fromarray(out, "RGBA")

    rgb = arr[:, :, :3].copy()
    weights_mask = None
    if face_priority and method == "kmeans":
        h = arr.shape[0]
        head_bottom = max(1, int(round(h * 0.45)))  # aligné sur ratio chibi
        weights_mask = np.ones(arr.shape[:2], dtype=np.float32)
        weights_mask[:head_bottom] = 3.0
    snapped = _pick_palette(rgb, opaque, num_colors, method, weights_mask=weights_mask)
    if not snapped:
        out = arr.copy()
        out[:, :, 3] = 0
        return Image.fromarray(out, "RGBA")

    if dither:
        new_rgb = _apply_palette_dithered(rgb, opaque, snapped)
    else:
        new_rgb = _apply_palette_flat_lab(rgb, snapped)

    out = np.zeros_like(arr)
    out[:, :, :3] = new_rgb
    out[:, :, 3] = np.where(opaque, 255, 0).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def split_two_layers(img_rgba: Image.Image, max_per_layer: int = 3):
    """Sépare un sprite en 2 layers compatibles NGPC (3 couleurs + transparent chacun).
    Stratégie : couleurs les plus utilisées dans layer A, reste dans layer B.
    Compatible avec `ngpc_sprite_export.py --layer2`.
    Retourne (layer_a, layer_b) — tous deux PIL RGBA."""
    arr = np.array(img_rgba)
    if arr.shape[2] == 3:
        arr = np.dstack([arr, np.full(arr.shape[:2], 255, np.uint8)])
    h, w = arr.shape[:2]
    opaque = arr[..., 3] >= 128
    rgb = arr[..., :3]
    blank = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    if not opaque.any():
        return blank, blank
    packed = (rgb[opaque, 0].astype(np.uint32) << 16) | \
             (rgb[opaque, 1].astype(np.uint32) << 8) | rgb[opaque, 2]
    uniq, counts = np.unique(packed, return_counts=True)
    color_list = []
    for p, c in zip(uniq, counts):
        pi = int(p)
        color_list.append((((pi >> 16) & 0xFF, (pi >> 8) & 0xFF, pi & 0xFF), int(c)))
    color_list.sort(key=lambda x: -x[1])
    if len(color_list) <= max_per_layer:
        return Image.fromarray(arr, "RGBA"), blank
    layer_a_cols = {c for c, _ in color_list[:max_per_layer]}
    layer_b_cols = {c for c, _ in color_list[max_per_layer:]}

    def build_layer(cols):
        out = np.zeros_like(arr)
        for c in cols:
            m = opaque & (rgb[..., 0] == c[0]) & (rgb[..., 1] == c[1]) & (rgb[..., 2] == c[2])
            out[m, :3] = c
            out[m, 3] = 255
        return Image.fromarray(out, "RGBA")

    return build_layer(layer_a_cols), build_layer(layer_b_cols)


# ---------------------------------------------------------------------------
# Character mode : 3-band body rig (tête / torse / jambes)
# ---------------------------------------------------------------------------

# Ratios cibles (head, body, legs) sommant à 1.0 — valeurs chibi/hero standard
BODY_RIG_PRESETS: dict[str, tuple[float, float, float]] = {
    "chibi":           (0.45, 0.30, 0.25),
    "hero":            (0.25, 0.40, 0.35),
    "super_deformed":  (0.50, 0.25, 0.25),
    "natural":         (0.33, 0.33, 0.34),
}


def _tighten_horizontal(band_pil: Image.Image) -> Image.Image:
    """Crop horizontal au bbox opaque. Laisse la hauteur intacte. Si aucune pixel
    opaque, renvoie l'image inchangée."""
    arr = np.array(band_pil.convert("RGBA"))
    if arr.shape[2] < 4:
        return band_pil
    alpha = arr[..., 3]
    opaque = alpha >= 128
    if not opaque.any():
        return band_pil
    col_any = opaque.any(axis=0)
    cols_idx = np.where(col_any)[0]
    if len(cols_idx) == 0:
        return band_pil
    x0, x1 = int(cols_idx[0]), int(cols_idx[-1]) + 1
    if x1 - x0 < 2:
        return band_pil
    return band_pil.crop((x0, 0, x1, band_pil.height))


def apply_body_rig_3band(img_pil: Image.Image,
                         neck_pct: float, hip_pct: float,
                         target_ratios: tuple[float, float, float],
                         tighten_bands: bool = False) -> Image.Image:
    """Rescale 3-band non-uniforme. La source est coupée en 3 bandes horizontales
    (tête / torse / jambes) selon `neck_pct` et `hip_pct` (% de la hauteur source),
    chacune redimensionnée pour occuper target_ratios[i] de la hauteur totale.

    Si tighten_bands=True, chaque bande est d'abord recroppée à son bbox opaque
    horizontal puis étirée à la pleine largeur — gros gain de détail par partie
    du corps (surtout la tête), au prix d'une légère distorsion horizontale.
    """
    if neck_pct >= hip_pct:
        return img_pil
    if neck_pct <= 0 or hip_pct >= 100:
        return img_pil
    total = sum(target_ratios)
    if abs(total - 1.0) > 0.01:
        return img_pil
    w, h = img_pil.size
    if h < 6:
        return img_pil
    neck_y = max(1, int(round(h * neck_pct / 100.0)))
    hip_y = min(h - 1, int(round(h * hip_pct / 100.0)))
    if hip_y <= neck_y:
        return img_pil
    head = img_pil.crop((0, 0, w, neck_y))
    body = img_pil.crop((0, neck_y, w, hip_y))
    legs = img_pil.crop((0, hip_y, w, h))

    if tighten_bands:
        head = _tighten_horizontal(head)
        body = _tighten_horizontal(body)
        legs = _tighten_horizontal(legs)

    th_h = max(1, int(round(h * target_ratios[0])))
    tb_h = max(1, int(round(h * target_ratios[1])))
    tl_h = h - th_h - tb_h
    if tl_h < 1:
        tl_h = 1
        if tb_h > th_h:
            tb_h = h - th_h - tl_h
        else:
            th_h = h - tb_h - tl_h

    # Chaque bande : étirée à la pleine largeur w × target height
    head_r = head.resize((w, th_h), Image.LANCZOS)
    body_r = body.resize((w, tb_h), Image.LANCZOS)
    legs_r = legs.resize((w, tl_h), Image.LANCZOS)

    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    out.paste(head_r, (0, 0))
    out.paste(body_r, (0, th_h))
    out.paste(legs_r, (0, th_h + tb_h))
    return out


# ---------------------------------------------------------------------------
# Character mode : contour silhouette + symétrie (phase 1)
# ---------------------------------------------------------------------------


def apply_silhouette_outline(img_rgba: Image.Image) -> Image.Image:
    """Contour 1-pixel interne : les pixels opaques sur la frange (= opaque & NOT eroded)
    sont forcés sur la couleur la plus sombre de la palette courante. Préserve la taille
    du sprite (pas d'extension externe). Ne marche que sur image déjà quantizée."""
    arr = np.array(img_rgba.convert("RGBA"))
    alpha = arr[..., 3]
    opaque = alpha >= 128
    if not opaque.any():
        return img_rgba

    # Couleurs opaques présentes
    rgb = arr[..., :3]
    op_pixels = rgb[opaque]
    packed = (op_pixels[:, 0].astype(np.uint32) << 16) | \
             (op_pixels[:, 1].astype(np.uint32) << 8) | op_pixels[:, 2]
    uniq = np.unique(packed)
    if len(uniq) == 0:
        return img_rgba
    # La plus sombre par luminance approx
    best = None
    best_l = float("inf")
    for p in uniq:
        pi = int(p)
        r = (pi >> 16) & 0xFF
        g = (pi >> 8) & 0xFF
        b = pi & 0xFF
        L = 0.299 * r + 0.587 * g + 0.114 * b
        if L < best_l:
            best_l = L
            best = (r, g, b)

    # Erosion : pixels opaques sur la frange = opaque & NOT eroded
    if _HAS_CV2:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eroded = cv2.erode(opaque.astype(np.uint8) * 255, kernel, iterations=1) >= 128
    else:
        # Erosion numpy fallback : intersection des 4-voisins
        eroded = np.zeros_like(opaque)
        eroded[1:-1, 1:-1] = (
            opaque[1:-1, 1:-1] & opaque[:-2, 1:-1] & opaque[2:, 1:-1] &
            opaque[1:-1, :-2] & opaque[1:-1, 2:]
        )
    ring = opaque & ~eroded

    out = arr.copy()
    out[ring, 0] = best[0]
    out[ring, 1] = best[1]
    out[ring, 2] = best[2]
    out[ring, 3] = 255
    return Image.fromarray(out, "RGBA")


def apply_selout(img_rgba: Image.Image) -> Image.Image:
    """Selective outlining (selout) : chaque pixel de frange prend le 'darker-shade
    partner' de sa propre couleur (la suivante plus sombre dans la palette triée par
    luminance). Donne un contour multi-tons beaucoup plus naturel qu'un liseré noir
    uniforme. Style classique Capcom vs SNK adapté à une palette 3 couleurs.

    Règle sur palette [C0=clair, C1=mid, C2=sombre] triée par luma :
      - Pixel clair (C0) en frange → remplacé par mid (C1)
      - Pixel mid (C1) en frange → remplacé par sombre (C2)
      - Pixel sombre (C2) en frange → reste (déjà aussi sombre que possible)
    """
    arr = np.array(img_rgba.convert("RGBA"))
    alpha = arr[..., 3]
    opaque = alpha >= 128
    if not opaque.any():
        return img_rgba

    rgb = arr[..., :3]
    # Palette triée par luma (ascendant = du plus sombre au plus clair)
    op_pixels = rgb[opaque]
    packed = (op_pixels[:, 0].astype(np.uint32) << 16) | \
             (op_pixels[:, 1].astype(np.uint32) << 8) | op_pixels[:, 2]
    uniq = np.unique(packed)
    colors = []
    for p in uniq:
        pi = int(p)
        colors.append(((pi >> 16) & 0xFF, (pi >> 8) & 0xFF, pi & 0xFF))
    if not colors:
        return img_rgba
    # Tri par luma ascendant — [sombre, mid, clair]
    colors.sort(key=lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2])
    # darker_partner[c] = plus sombre dans la liste (= c precedent, ou c lui-même si déjà sombre)
    darker_partner: dict[tuple, tuple] = {}
    for i, c in enumerate(colors):
        darker_partner[c] = colors[max(0, i - 1)]

    # Frange opaque = opaque & NOT eroded
    if _HAS_CV2:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eroded = cv2.erode(opaque.astype(np.uint8) * 255, kernel, iterations=1) >= 128
    else:
        eroded = np.zeros_like(opaque)
        eroded[1:-1, 1:-1] = (
            opaque[1:-1, 1:-1] & opaque[:-2, 1:-1] & opaque[2:, 1:-1] &
            opaque[1:-1, :-2] & opaque[1:-1, 2:]
        )
    ring = opaque & ~eroded

    out = arr.copy()
    for c, partner in darker_partner.items():
        if partner == c:
            continue  # déjà le plus sombre, rien à faire
        color_mask = (rgb[..., 0] == c[0]) & (rgb[..., 1] == c[1]) & (rgb[..., 2] == c[2])
        target = ring & color_mask
        out[target, 0] = partner[0]
        out[target, 1] = partner[1]
        out[target, 2] = partner[2]
    return Image.fromarray(out, "RGBA")


def apply_island_cleanup(img_rgba: Image.Image) -> Image.Image:
    """Nettoyage post-quantize pour lisibilité au petit format :
      1) Pixels opaques orphelins (0 voisin opaque en 8-connectivité) → rendus transparents
      2) Trous transparents de 1 pixel entourés (≥7 voisins opaques sur 8) → remplis avec
         la couleur majoritaire des voisins

    Les sprites au 16×16/32×32 souffrent énormément de 1-2 pixels parasites qui cassent
    la silhouette. Ce pass les élimine.
    """
    arr = np.array(img_rgba.convert("RGBA"))
    alpha = arr[..., 3]
    opaque = alpha >= 128
    h, w = opaque.shape

    # Compte voisins opaques en 8-connectivité via offsets numpy
    neigh = np.zeros(opaque.shape, dtype=np.int8)
    op_i8 = opaque.astype(np.int8)
    # 4 directions cardinales
    neigh[1:, :] += op_i8[:-1, :]
    neigh[:-1, :] += op_i8[1:, :]
    neigh[:, 1:] += op_i8[:, :-1]
    neigh[:, :-1] += op_i8[:, 1:]
    # 4 diagonales
    neigh[1:, 1:] += op_i8[:-1, :-1]
    neigh[:-1, :-1] += op_i8[1:, 1:]
    neigh[1:, :-1] += op_i8[:-1, 1:]
    neigh[:-1, 1:] += op_i8[1:, :-1]

    orphan = opaque & (neigh == 0)
    hole = (~opaque) & (neigh >= 7)

    out = arr.copy()
    # 1) Orphelins → transparent
    if orphan.any():
        out[orphan, 3] = 0

    # 2) Trous 1px → remplir avec couleur majoritaire du 3×3 voisin
    if hole.any():
        ys_hole, xs_hole = np.where(hole)
        for y, x in zip(ys_hole, xs_hole):
            y0, y1 = max(0, y - 1), min(h, y + 2)
            x0, x1 = max(0, x - 1), min(w, x + 2)
            neigh_block = arr[y0:y1, x0:x1]
            neigh_alpha = neigh_block[..., 3] >= 128
            if not neigh_alpha.any():
                continue
            neigh_rgb = neigh_block[..., :3][neigh_alpha]
            packed = (neigh_rgb[:, 0].astype(np.uint32) << 16) | \
                     (neigh_rgb[:, 1].astype(np.uint32) << 8) | neigh_rgb[:, 2]
            uniq, counts = np.unique(packed, return_counts=True)
            dominant = int(uniq[int(np.argmax(counts))])
            out[y, x, 0] = (dominant >> 16) & 0xFF
            out[y, x, 1] = (dominant >> 8) & 0xFF
            out[y, x, 2] = dominant & 0xFF
            out[y, x, 3] = 255

    return Image.fromarray(out, "RGBA")


def apply_symmetric_mirror(img_rgba: Image.Image) -> Image.Image:
    """Mirror vertical : réplique la moitié dominante (plus d'opaques) sur l'autre.
    Préserve la taille totale."""
    arr = np.array(img_rgba.convert("RGBA"))
    w = arr.shape[1]
    if w < 2:
        return img_rgba
    half = w // 2
    alpha = arr[..., 3]
    left_op = int((alpha[:, :half] >= 128).sum())
    right_op = int((alpha[:, w - half:] >= 128).sum())
    if left_op >= right_op:
        arr[:, w - half:] = arr[:, :half][:, ::-1]
    else:
        arr[:, :half] = arr[:, w - half:][:, ::-1]
    return Image.fromarray(arr, "RGBA")


def extract_palette(img_rgba: Image.Image) -> list[tuple[int, int, int]]:
    """Liste les couleurs opaques uniques, triées par luminance croissante."""
    arr = np.array(img_rgba)
    if arr.shape[2] < 4:
        return []
    op = arr[..., 3] >= 128
    if not op.any():
        return []
    rgb = arr[..., :3][op]
    packed = (rgb[:, 0].astype(np.uint32) << 16) | (rgb[:, 1].astype(np.uint32) << 8) | rgb[:, 2]
    uniq = np.unique(packed)
    colors = []
    for p in uniq:
        pi = int(p)
        colors.append(((pi >> 16) & 0xFF, (pi >> 8) & 0xFF, pi & 0xFF))
    colors.sort(key=lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2])
    return colors


# ---------------------------------------------------------------------------
# Mode BG (tilemap) — NGPC specs : 2 scroll planes, 16 palettes × 3 couleurs,
# tiles 8x8 (2bpp), max 32x32 tiles par plane, ≤ 512 tiles uniques après dedupe.
# Ref : `ngpc_tilemap.py`, TILEMAPS_SCROLL.md, HW_REGISTERS.md
# ---------------------------------------------------------------------------


def _palette_distance_lab_sorted(pal_a_lab: np.ndarray, pal_b_lab: np.ndarray) -> float:
    """Distance heuristique entre 2 palettes (≤3 couleurs) : pair-up par luminance
    croissante (tri sur L*) puis somme des écarts carrés. Bon compromis vitesse/qualité."""
    n = min(len(pal_a_lab), len(pal_b_lab))
    if n == 0:
        return float("inf")
    a = pal_a_lab[np.argsort(pal_a_lab[:, 0])][:n]
    b = pal_b_lab[np.argsort(pal_b_lab[:, 0])][:n]
    return float(((a - b) ** 2).sum())


def _merge_two_palettes(p_a: frozenset, p_b: frozenset) -> frozenset:
    """Fusionne deux palettes (sets de couleurs RGB444) en ≤3 couleurs via k-means LAB."""
    union = list(p_a | p_b)
    if len(union) <= 3:
        return frozenset(union)
    union_rgb = np.array(union, dtype=np.uint8)
    union_lab = rgb_to_lab(union_rgb)
    centers = _kmeans(union_lab, 3)
    return frozenset(_snap_lab_to_rgb444_unique(centers))


def tile_aware_quantize(img_rgba: Image.Image,
                        num_palettes: int = 16,
                        black_is_transparent: bool = False) -> tuple[Image.Image, dict]:
    """Quantize tilemap-aware pour NGPC BG.

    Contraintes satisfaites :
    - Chaque tile 8×8 utilise ≤3 couleurs opaques (+ transparent index 0)
    - Palettes distinctes ≤ num_palettes (max 16)
    - Couleurs finales snappées RGB444

    Pipeline :
      1. Quantize globale à ~num_palettes*3 couleurs candidates (k-means LAB)
      2. Re-mapping source vers ces candidates
      3. Par tile 8×8 : collecter les ≤3 couleurs les plus utilisées (= mini-palette)
      4. Greedy merge des mini-palettes jusqu'à ≤num_palettes (distance Lab sorted-pair)
      5. Pour chaque tile : assignée à sa palette mergée la plus proche, re-quantize à 3 couleurs

    Retour : (image PIL RGBA alignée-tile, dict stats)
    """
    arr = np.array(img_rgba.convert("RGBA"))
    h, w = arr.shape[:2]
    if w % 8 != 0 or h % 8 != 0:
        raise ValueError(f"Dimensions non multiples de 8 : {w}×{h}")

    # Seuil abaissé : préserve les détails fins post-downscale (mêmes raisons
    # que quantize_ngpc — anti-aliasing sur les bords des zones opaques)
    opaque = arr[..., 3] >= ALPHA_OPAQUE_THRESHOLD
    rgb = arr[..., :3]

    if black_is_transparent:
        black_mask = (rgb[..., 0] < 16) & (rgb[..., 1] < 16) & (rgb[..., 2] < 16)
        opaque = opaque & ~black_mask

    nth, ntw = h // 8, w // 8
    total_tiles = nth * ntw

    # Cas : tout transparent
    if not opaque.any():
        out = arr.copy()
        out[..., 3] = 0
        return Image.fromarray(out, "RGBA"), {
            "unique_tiles": 0, "total_tiles": total_tiles,
            "palettes_used": 0, "palettes_max": num_palettes,
            "saturated_tiles": 0,
        }

    # === Étape 1 : pool candidat global ~N*3 couleurs via k-means LAB ===
    op_pixels = rgb[opaque]
    target_cands = min(num_palettes * 3, max(3, len(op_pixels)))
    if len(op_pixels) > 30000:
        rng = np.random.default_rng(0)
        op_pixels = op_pixels[rng.choice(len(op_pixels), 30000, replace=False)]
    op_lab = rgb_to_lab(op_pixels)
    cand_centers_lab = _kmeans(op_lab, target_cands)
    cand_rgb = np.array(_snap_lab_to_rgb444_unique(cand_centers_lab), dtype=np.uint8)
    cand_lab = rgb_to_lab(cand_rgb)

    # === Étape 2 : re-map source vers candidates (nearest LAB) ===
    rgb_lab = rgb_to_lab(rgb).reshape(-1, 3)
    d2_to_cand = ((rgb_lab[:, None, :] - cand_lab[None, :, :]) ** 2).sum(axis=2)
    nearest_cand = np.argmin(d2_to_cand, axis=1)
    remapped = cand_rgb[nearest_cand].reshape(rgb.shape)

    # === Étape 3 : par tile 8×8, extraire top-3 couleurs opaques ===
    tile_palettes: list[frozenset | None] = []  # index = ty*ntw + tx
    for ty in range(nth):
        y0, y1 = ty * 8, (ty + 1) * 8
        for tx in range(ntw):
            x0, x1 = tx * 8, (tx + 1) * 8
            tile_op = opaque[y0:y1, x0:x1]
            if not tile_op.any():
                tile_palettes.append(None)
                continue
            tile_rgb = remapped[y0:y1, x0:x1]
            op_cols = tile_rgb[tile_op]
            packed = (op_cols[:, 0].astype(np.uint32) << 16) | \
                     (op_cols[:, 1].astype(np.uint32) << 8) | op_cols[:, 2]
            uniq, cnts = np.unique(packed, return_counts=True)
            order = np.argsort(-cnts)
            top = uniq[order[:3]]
            colors = frozenset(
                ((int(p) >> 16) & 0xFF, (int(p) >> 8) & 0xFF, int(p) & 0xFF)
                for p in top
            )
            tile_palettes.append(colors)

    # === Étape 4 : merger les mini-palettes jusqu'à ≤ num_palettes ===
    unique_pals: list[frozenset] = list(
        {p for p in tile_palettes if p is not None}
    )
    if len(unique_pals) > num_palettes:
        pal_labs: list[np.ndarray] = [
            rgb_to_lab(np.array(list(p), dtype=np.uint8)) for p in unique_pals
        ]
        # Matrice de distances (triangulaire supérieure, autres = +inf)
        n = len(unique_pals)
        D = np.full((n, n), np.inf, dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = _palette_distance_lab_sorted(pal_labs[i], pal_labs[j])
        while len(unique_pals) > num_palettes:
            flat = int(np.argmin(D))
            i, j = divmod(flat, D.shape[1])
            if D[i, j] == np.inf:
                break
            merged = _merge_two_palettes(unique_pals[i], unique_pals[j])
            a, b = max(i, j), min(i, j)
            unique_pals.pop(a); unique_pals.pop(b)
            pal_labs.pop(a); pal_labs.pop(b)
            D = np.delete(D, [a, b], axis=0)
            D = np.delete(D, [a, b], axis=1)
            unique_pals.append(merged)
            new_lab = rgb_to_lab(np.array(list(merged), dtype=np.uint8))
            pal_labs.append(new_lab)
            # Ajouter ligne+col pour la nouvelle palette
            new_dists = np.array(
                [_palette_distance_lab_sorted(pal_labs[k], new_lab) for k in range(len(pal_labs) - 1)],
                dtype=np.float32,
            )
            D = np.pad(D, ((0, 1), (0, 1)), mode="constant", constant_values=np.inf)
            D[:-1, -1] = new_dists  # upper tri cell pour les anciens vs new
            D[-1, :] = np.inf        # dernière ligne toute inf (jamais lue comme i<j)

    # === Étape 5 : assigner chaque tile à sa palette mergée + re-quantize ===
    pal_arrays = [np.array(list(p), dtype=np.uint8) for p in unique_pals]
    pal_labs_final = [rgb_to_lab(pa) for pa in pal_arrays]

    out = np.zeros_like(arr)
    saturated = 0
    for ty in range(nth):
        y0, y1 = ty * 8, (ty + 1) * 8
        for tx in range(ntw):
            x0, x1 = tx * 8, (tx + 1) * 8
            tile_op = opaque[y0:y1, x0:x1]
            if not tile_op.any():
                out[y0:y1, x0:x1, 3] = 0
                continue
            original_pal = tile_palettes[ty * ntw + tx]
            if original_pal is None or not unique_pals:
                continue
            orig_lab = rgb_to_lab(np.array(list(original_pal), dtype=np.uint8))
            best_idx = 0
            best_d = float("inf")
            for k, pal_lab in enumerate(pal_labs_final):
                d = _palette_distance_lab_sorted(orig_lab, pal_lab)
                if d < best_d:
                    best_d = d
                    best_idx = k
            pal = pal_arrays[best_idx]
            pal_lab = pal_labs_final[best_idx]
            tile_rgb = remapped[y0:y1, x0:x1]
            tile_lab = rgb_to_lab(tile_rgb).reshape(-1, 3)
            d2 = ((tile_lab[:, None, :] - pal_lab[None, :, :]) ** 2).sum(axis=2)
            nearest = np.argmin(d2, axis=1).reshape(8, 8)
            out[y0:y1, x0:x1, :3] = pal[nearest]
            out[y0:y1, x0:x1, 3] = np.where(tile_op, 255, 0)
            if len(original_pal) > 3:
                saturated += 1

    # === Stats : tiles uniques — strict (exact) et flip-aware (hflip/vflip/hvflip) ===
    # NGPC stocke 1 tile physique en VRAM + bits hflip/vflip par cell de tilemap.
    # Donc le vrai coût VRAM = count des tiles après dedupe sous les 4 transformations.
    tile_set_strict: set[bytes] = set()
    tile_set_flipaware: set[bytes] = set()
    for ty in range(nth):
        y0, y1 = ty * 8, (ty + 1) * 8
        for tx in range(ntw):
            x0, x1 = tx * 8, (tx + 1) * 8
            tile = out[y0:y1, x0:x1]
            tile_set_strict.add(tile.tobytes())
            # Canonical form = min des 4 variantes (identity, H, V, HV)
            canon = min(
                tile.tobytes(),
                tile[:, ::-1].tobytes(),
                tile[::-1, :].tobytes(),
                tile[::-1, ::-1].tobytes(),
            )
            tile_set_flipaware.add(canon)

    return Image.fromarray(out, "RGBA"), {
        "unique_tiles": len(tile_set_strict),
        "unique_tiles_with_flips": len(tile_set_flipaware),
        "total_tiles": total_tiles,
        "palettes_used": len(unique_pals),
        "palettes_max": num_palettes,
        "saturated_tiles": saturated,
    }


class StagedPipeline:
    """Pipeline 3-étages avec cache : change de quantize_method ne relance pas GrabCut.

    Étages :
      A : crop + ajustements + bilateral + sharpen + GrabCut + erode       (source → fullsize RGBA)
      B : + auto-crop + pad aspect + downscale multi-pass                  (fullsize → target-size RGBA)
      C : + quantize (method + dither)                                     (target-size → sprite final)
    """

    def __init__(self):
        self._key_a: tuple | None = None
        self._res_a: Image.Image | None = None
        self._key_b: tuple | None = None
        self._res_b: Image.Image | None = None
        self._key_c: tuple | None = None
        self._res_c: Image.Image | None = None
        self._stats_c: dict | None = None  # pour mode BG
        self._effective_crop: tuple | None = None  # bbox utilisé en stage A
        self._last_source_img: Image.Image | None = None  # référence pour stage C

    def reset(self):
        self._key_a = self._key_b = self._key_c = None
        self._res_a = self._res_b = self._res_c = None
        self._stats_c = None

    def last_stats(self) -> dict | None:
        return self._stats_c

    def run(self, source_img: Image.Image, source_version: int,
            crop_rect, contrast: int, brightness: int, saturation: int,
            smooth_edges: bool, sharpen: bool, remove_bg: bool,
            user_mask: np.ndarray | None, user_mask_version: int,
            anti_halo: bool, auto_crop: bool, target_w: int, target_h: int,
            multi_pass_downscale: bool, num_colors: int,
            quantize_method: str, dither: bool,
            mode: str = "sprite", num_palettes: int = 16,
            black_is_transparent: bool = False,
            outline: bool = False,
            selout: bool = False,
            mirror: bool = False,
            body_rig: bool = False,
            rig_neck_pct: float = 33.0,
            rig_hip_pct: float = 66.0,
            rig_preset: str = "chibi",
            tighten_bands: bool = False,
            preserve_top: bool = False,
            face_priority: bool = False,
            island_cleanup: bool = False,
            face_crop: bool = False,
            cutout_method: str = "grabcut",
            eye_preserve: bool = False,
            super_res: bool = False,
            super_res_model: str = "x4plus") -> Image.Image:
        # === Stage A : préprocess + super-res + détourage ===
        key_a = (source_version, tuple(crop_rect) if crop_rect else None,
                 face_crop, super_res, super_res_model,
                 contrast, brightness, saturation,
                 smooth_edges, sharpen, remove_bg,
                 user_mask_version if user_mask is not None else None,
                 anti_halo, preserve_top, cutout_method)
        # Garde la réf source pour le stage C (eye preservation)
        self._last_source_img = source_img
        if key_a != self._key_a or self._res_a is None:
            img = source_img
            mask = user_mask
            # Si face_crop=True et pas de crop_rect manuel, tente la détection MediaPipe.
            # Le crop_rect manuel garde toujours la priorité si défini.
            effective_crop = crop_rect
            if face_crop and crop_rect is None:
                face_bbox = detect_face_bbox(source_img)
                if face_bbox is not None:
                    effective_crop = face_bbox
            # Stocker le crop effectif pour le stage C (eye preservation)
            self._effective_crop = effective_crop
            if effective_crop is not None:
                l, t, r, b = effective_crop
                l = max(0, l); t = max(0, t)
                r = min(img.width, r); b = min(img.height, b)
                if r - l >= 4 and b - t >= 4:
                    img = img.crop((l, t, r, b))
                    if mask is not None:
                        mask = mask[t:b, l:r]
            # Super-resolution AVANT les ajustements : plus de détails captés
            # par les filtres bilateral / sharpen subséquents, et plus de
            # matière pour le downscale Lanczos multi-pass.
            if super_res:
                img = upscale_superres(img, model_name=super_res_model)
                # Mask user (brush hints) : si présent, on upscale aussi
                if mask is not None:
                    if img.width != mask.shape[1] or img.height != mask.shape[0]:
                        mask_pil = Image.fromarray(mask, "L")
                        mask_pil = mask_pil.resize((img.width, img.height), Image.NEAREST)
                        mask = np.array(mask_pil)
            if contrast or brightness or saturation:
                img = apply_adjustments(img, contrast, brightness, saturation)
            if smooth_edges:
                img = bilateral_preprocess(img)
            if sharpen:
                img = sharpen_edges(img)
            if remove_bg:
                # rembg ignore les hints utilisateurs (ML segmentation complète) ;
                # GrabCut les utilise pour affiner. rembg sert aux détourages "one-shot
                # propres" (photos studio / fond uni), GrabCut aux cas semi-manuels.
                if cutout_method == "rembg" and _HAS_REMBG and (mask is None or not mask.any()):
                    img = remove_background_rembg(img)
                else:
                    top_ratio = 0.02 if preserve_top else None
                    img = remove_background_grabcut(
                        img, user_mask=mask, margin_top_ratio=top_ratio
                    )
            else:
                img = img.convert("RGBA")
            if remove_bg and anti_halo:
                img = erode_alpha(img, px=1)
            self._res_a = img
            self._key_a = key_a
            # Invalider étages en aval
            self._key_b = self._key_c = None

        # === Stage B : body rig + auto-crop + pad aspect + downscale target ===
        key_b = (self._key_a, body_rig, rig_neck_pct, rig_hip_pct, rig_preset,
                 tighten_bands, auto_crop, target_w, target_h, multi_pass_downscale)
        if key_b != self._key_b or self._res_b is None:
            img = self._res_a
            if body_rig and mode == "sprite":
                ratios = BODY_RIG_PRESETS.get(rig_preset, BODY_RIG_PRESETS["chibi"])
                img = apply_body_rig_3band(img, rig_neck_pct, rig_hip_pct, ratios,
                                           tighten_bands=tighten_bands)
            if auto_crop:
                img = auto_crop_subject(img, padding_ratio=0.06)
                img = pad_to_aspect(img, target_w / max(1, target_h))
            img = downscale(img, target_w, target_h, multi_pass=multi_pass_downscale)
            self._res_b = img
            self._key_b = key_b
            self._key_c = None

        # === Stage C : quantize (sprite ou BG selon mode) + post-process ===
        # Les post-process (outline + mirror + cleanup + eye) ne s'appliquent qu'en mode
        # sprite ; en mode BG ils violeraient la contrainte 3-couleurs-par-tile.
        key_c = (self._key_b, mode, num_colors, quantize_method, dither,
                 num_palettes, black_is_transparent,
                 outline if mode == "sprite" else False,
                 selout if mode == "sprite" else False,
                 mirror if mode == "sprite" else False,
                 face_priority if mode == "sprite" else False,
                 island_cleanup if mode == "sprite" else False,
                 eye_preserve if mode == "sprite" else False)
        if key_c != self._key_c or self._res_c is None:
            if mode == "bg":
                self._res_c, self._stats_c = tile_aware_quantize(
                    self._res_b,
                    num_palettes=num_palettes,
                    black_is_transparent=black_is_transparent,
                )
            else:
                result = quantize_ngpc(
                    self._res_b, num_colors=num_colors,
                    dither=dither, method=quantize_method,
                    face_priority=face_priority,
                )
                # Priorité : selout > uniform outline > rien
                if outline and selout:
                    result = apply_selout(result)
                elif outline:
                    result = apply_silhouette_outline(result)
                if mirror:
                    result = apply_symmetric_mirror(result)
                if island_cleanup:
                    result = apply_island_cleanup(result)
                if eye_preserve and self._last_source_img is not None:
                    result = apply_eye_preservation(
                        result, self._last_source_img,
                        effective_crop=self._effective_crop,
                    )
                self._res_c = result
                self._stats_c = None
            self._key_c = key_c

        return self._res_c


def process(source_img: Image.Image, crop_rect=None,
            target_w: int = 16, target_h: int = 16,
            num_colors: int = 3, remove_bg: bool = True,
            contrast: int = 0, brightness: int = 0, saturation: int = 0,
            smooth_edges: bool = False, sharpen: bool = False,
            dither: bool = False,
            quantize_method: str = "mediancut",
            auto_crop: bool = True,
            anti_halo: bool = True,
            multi_pass_downscale: bool = True,
            outline: bool = False,
            selout: bool = False,
            mirror: bool = False,
            body_rig: bool = False,
            rig_neck_pct: float = 33.0,
            rig_hip_pct: float = 66.0,
            rig_preset: str = "chibi",
            tighten_bands: bool = False,
            preserve_top: bool = False,
            face_priority: bool = False,
            island_cleanup: bool = False,
            face_crop: bool = False,
            cutout_method: str = "grabcut",
            eye_preserve: bool = False,
            super_res: bool = False,
            super_res_model: str = "x4plus",
            user_mask: np.ndarray | None = None) -> Image.Image:
    img = source_img
    mask = user_mask
    effective_crop = crop_rect
    if face_crop and crop_rect is None:
        face_bbox = detect_face_bbox(source_img)
        if face_bbox is not None:
            effective_crop = face_bbox
    _saved_effective_crop = effective_crop
    if effective_crop is not None:
        l, t, r, b = effective_crop
        l = max(0, l); t = max(0, t)
        r = min(img.width, r); b = min(img.height, b)
        if r - l >= 4 and b - t >= 4:
            img = img.crop((l, t, r, b))
            if mask is not None:
                mask = mask[t:b, l:r]
    if super_res:
        img = upscale_superres(img, model_name=super_res_model)
        if mask is not None and (img.width != mask.shape[1] or img.height != mask.shape[0]):
            mask_pil = Image.fromarray(mask, "L")
            mask_pil = mask_pil.resize((img.width, img.height), Image.NEAREST)
            mask = np.array(mask_pil)
    if contrast or brightness or saturation:
        img = apply_adjustments(img, contrast, brightness, saturation)
    if smooth_edges:
        img = bilateral_preprocess(img)
    if sharpen:
        img = sharpen_edges(img)
    if remove_bg:
        if cutout_method == "rembg" and _HAS_REMBG and (mask is None or not mask.any()):
            img = remove_background_rembg(img)
        else:
            top_ratio = 0.02 if preserve_top else None
            img = remove_background_grabcut(img, user_mask=mask, margin_top_ratio=top_ratio)
    else:
        img = img.convert("RGBA")
    if remove_bg and anti_halo:
        img = erode_alpha(img, px=1)
    if body_rig:
        ratios = BODY_RIG_PRESETS.get(rig_preset, BODY_RIG_PRESETS["chibi"])
        img = apply_body_rig_3band(img, rig_neck_pct, rig_hip_pct, ratios,
                                   tighten_bands=tighten_bands)
    if auto_crop:
        img = auto_crop_subject(img, padding_ratio=0.06)
        img = pad_to_aspect(img, target_w / max(1, target_h))
    img = downscale(img, target_w, target_h, multi_pass=multi_pass_downscale)
    img = quantize_ngpc(img, num_colors=num_colors, dither=dither,
                        method=quantize_method, face_priority=face_priority)
    if outline and selout:
        img = apply_selout(img)
    elif outline:
        img = apply_silhouette_outline(img)
    if mirror:
        img = apply_symmetric_mirror(img)
    if island_cleanup:
        img = apply_island_cleanup(img)
    if eye_preserve:
        img = apply_eye_preservation(img, source_img, effective_crop=_saved_effective_crop)
    return img
