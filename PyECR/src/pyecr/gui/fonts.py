"""Display-only font discovery, glyph coverage and reproducible diagnostics.

System candidates are optional; DejaVu comes from the Matplotlib dependency.
Coverage is checked before DearPyGui atlas construction. Fonts are not shipped
with PyECR; a local cache is used only when a native Windows path needs it.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from importlib.util import find_spec
import logging
import os
from pathlib import Path
import sys
import tempfile

log = logging.getLogger(__name__)
BODY_PROBE = "Time Signal Conductivity Data Processing 0123456789αβγΔλμστ∂±−×°Ωµ²³ cm/s S/cm"
SCIENCE_PROBE = "0123456789αβγδΔλμστ∂₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺±−×°Ωµ cm²/s S/cm"
ALL_PROBE = BODY_PROBE + SCIENCE_PROBE


@dataclass(frozen=True)
class FontFace:
    path: Path
    family: str
    sha256: str
    codepoints: frozenset[int]
    origin: str

    def missing(self, text: str) -> str:
        return "".join(dict.fromkeys(c for c in text if not c.isspace() and ord(c) not in self.codepoints))


@dataclass(frozen=True)
class FontPlan:
    body: FontFace | None
    scientific: FontFace | None
    diagnostics: tuple[str, ...]
    heading: FontFace | None = None

    def describe(self) -> str:
        lines = list(self.diagnostics)
        for role, face in (("UI / CJK", self.body), ("Scientific / plot", self.scientific),
                           ("Step headings / regular", self.heading)):
            if face is None:
                if role == 'Step headings / regular':
                    lines.append('Step headings: built-in regular fallback; coverage unavailable.')
                    continue
                lines.append(f"{role}: built-in fallback; coverage unavailable. Configure PYECR_UI_FONT.")
                continue
            lines.extend((f"{role}: {face.family} | {face.origin}", str(face.path), f"SHA256 {face.sha256}"))
            missing = face.missing(ALL_PROBE)
            lines.append("Outside this role's verified coverage: " + (" ".join(f"U+{ord(c):04X}" for c in missing) or "none"))
        lines.append("System/user fonts are discovered locally, not bundled. A local cache is used only for native Windows path compatibility. Scientific font uses the existing Matplotlib distribution. No global OS/font/DPI setting is changed.")
        return "\n".join(lines)


def bundled_scientific_font() -> Path | None:
    spec = find_spec("matplotlib")
    return None if spec is None or spec.origin is None else Path(spec.origin).parent / "mpl-data/fonts/ttf/DejaVuSans.ttf"


def system_cjk_candidates() -> tuple[Path, ...]:
    """Finite platform paths, checked for existence/readability/coverage later."""
    if sys.platform == "win32":
        folders = [Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"]
        if os.environ.get("LOCALAPPDATA"):
            folders.append(Path(os.environ["LOCALAPPDATA"]) / "Microsoft/Windows/Fonts")
        names = ("msyh.ttc", "NotoSansSC-Regular.ttf", "NotoSansSC-VF.ttf", "NotoSansCJKsc-Regular.otf", "simsun.ttc", "simhei.ttf")
    elif sys.platform == "darwin":
        folders = [Path("/System/Library/Fonts"), Path("/Library/Fonts"), Path.home()/"Library/Fonts"]
        names = ("PingFang.ttc", "STHeiti Light.ttc", "NotoSansCJKsc-Regular.otf")
    else:
        folders = [Path("/usr/share/fonts/opentype/noto"), Path("/usr/share/fonts/truetype/noto"), Path.home()/".local/share/fonts"]
        names = ("NotoSansCJK-Regular.ttc", "NotoSansCJKsc-Regular.otf", "NotoSansSC-Regular.ttf")
    return tuple(folder/name for folder in folders for name in names)


@lru_cache(maxsize=16)
def _inspect(path: str, size: int, mtime_ns: int) -> tuple[str, str, frozenset[int]]:
    from matplotlib.ft2font import FT2Font
    font = FT2Font(path)  # Same first face used by DPG for a TTC; no magic family fallback.
    return font.family_name, hashlib.sha256(Path(path).read_bytes()).hexdigest(), frozenset(font.get_charmap())


def inspect_font(path: Path, origin: str) -> FontFace:
    path = path.expanduser().resolve()
    stat = path.stat()
    if not path.is_file():
        raise ValueError("font path is not a regular file")
    family, digest, coverage = _inspect(str(path), stat.st_size, stat.st_mtime_ns)
    return FontFace(path, family, digest, coverage, origin)


def native_font_path(face: FontFace) -> str:
    """Return a path the Windows native font loader can open.

    Prefer the original file or an ASCII relative path. If the source is on
    another drive or has a non-ASCII filename, cache its verified bytes under
    an ASCII filename relative to the working folder. The working directory
    is never changed, and the cache is not part of the source distribution.
    """
    path = face.path
    if os.name != "nt" or str(path).isascii():
        return str(path)
    try:
        relative = os.path.relpath(path, Path.cwd())
    except ValueError:
        relative = ""
    if relative and relative.isascii():
        return relative

    cache = Path.cwd() / ".pyecr-font-cache"
    cache.mkdir(exist_ok=True)
    suffix = ".ttc" if path.suffix.lower() == ".ttc" else ".ttf"
    cached = cache / (face.sha256 + suffix)
    if not cached.is_file() or hashlib.sha256(cached.read_bytes()).hexdigest() != face.sha256:
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != face.sha256:
            raise OSError("Font changed after coverage inspection; restart PyECR.")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=cache, prefix="font-", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
            temporary.replace(cached)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return str(cached.relative_to(Path.cwd()))


def _is_bold_face(path):
    from matplotlib.ft2font import FT2Font, StyleFlags
    return bool(FT2Font(str(path)).style_flags & StyleFlags.BOLD)


def choose_heading_font(body, *, candidates=None, inspector=inspect_font, bold_check=_is_bold_face):
    """Reference a real, coverage-checked local bold face; never synthesize it.

    Old FontPlan constructors and the regular/scientific font policy remain valid.
    Fonts are discovered locally and are not included in the distribution.
    """
    if body is None:
        return None
    if candidates is None:
        path=body.path
        known={'msyh.ttc':'msyhbd.ttc','arial.ttf':'arialbd.ttf',
               'DejaVuSans.ttf':'DejaVuSans-Bold.ttf'}
        names=(known.get(path.name,''),path.name.replace('-Regular','-Bold'),
               path.stem+'-Bold'+path.suffix)
        candidates=tuple(dict.fromkeys((path,*(path.with_name(n) for n in names if n))))
    for path in candidates:
        try:
            face=inspector(Path(path),'local bold heading candidate')
            if not face.missing(BODY_PROBE) and bold_check(face.path):
                return face
        except (OSError,ValueError,RuntimeError):
            continue
    return None


def choose_fonts(configured: str = "", *, candidates=None, scientific_path=None, inspector=inspect_font) -> FontPlan:
    """Explicit suitable configuration first; optional CJK and scientific fallback.

    Injection points support missing/corrupt-font tests on hosts with no fonts.
    An unsuitable explicit face is diagnosed, never treated as verified CJK.
    """
    choices = [(Path(configured), "PYECR_UI_FONT")] if configured.strip() else []
    choices += [(p, "installed CJK candidate") for p in (system_cjk_candidates() if candidates is None else candidates)]
    scientific_path = bundled_scientific_font() if scientific_path is None else scientific_path
    if scientific_path is not None:
        choices.append((scientific_path, "Matplotlib DejaVu Sans (existing dependency)"))
    body = scientific = None
    notes = []
    for path, origin in choices:
        try:
            face = inspector(Path(path), origin)
        except (OSError, ValueError, RuntimeError) as exc:
            notes.append(f"Unavailable: {path} | {type(exc).__name__}: {exc}")
            continue
        if body is None:
            missing = face.missing(BODY_PROBE)
            if not missing:
                body = face
            else:
                codes = ' '.join(f'U+{ord(c):04X}' for c in missing)
                notes.append(f"Not verified for UI/CJK: {path} | missing {codes}")
        if scientific is None and not face.missing(SCIENCE_PROBE):
            scientific = face
        if body is not None and scientific is not None:
            break
    if body is None:
        notes.append("CJK font unavailable: built-in UI may show missing glyphs; set PYECR_UI_FONT to a readable CJK font.")
    if scientific is None:
        notes.append("Scientific font unavailable: Unicode symbols are not verified.")
    return FontPlan(body, scientific, tuple(notes))


def font_for_text(plan: FontPlan, text: str, *, scientific: bool = False) -> str | None:
    """Pick a whole-widget face without changing text or values.

    A mixed CJK + unsupported math string cannot be composed by DPG; return
    None when neither face covers it. Diagnostics retain the missing symbols.
    """
    faces = (("pyecr_ui_font", plan.body), ("pyecr_plot_font", plan.scientific))
    for tag, face in (tuple(reversed(faces)) if scientific else faces):
        if face is not None and not face.missing(text):
            return tag
    return None
