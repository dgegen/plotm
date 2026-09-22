import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Union

import yaml

from plotm.paths import DEFAULT_PROFILES_DIR, PROFILES_DIR, STYLES_DIR

__all__ = ["PlotProfile", "ProfileManager", "UsageType"]

logger = logging.getLogger("plotm")


class UsageType(Enum):
    DEFAULT = "default"
    THESIS = "thesis"
    PAPER_1COL = "paper_1col"
    PAPER_2COL = "paper_2col"
    PRESENTATION = "presentation"
    BEAMER = "beamer"
    CUSTOM = "custom"

    @classmethod
    def determine(cls, name):
        if isinstance(name, UsageType):
            return name
        elif not isinstance(name, str):
            raise ValueError("UsageType must be a string or UsageType enum")

        name = name.lower()
        if name in ("thesis", "report"):
            return cls.THESIS
        elif "paper" in name:
            if "2" in name or "double" in name or "two" in name:
                return cls.PAPER_2COL
            else:
                return cls.PAPER_1COL
        elif name in ("presentation", "talk"):
            return cls.PRESENTATION
        elif name in ("beamer",):
            return cls.BEAMER
        elif name in ("custom", "unknown", "user"):
            return cls.CUSTOM
        else:
            return cls.DEFAULT

    def defaults(self) -> dict[str, Any]:
        """Return default text_width (pt), rescale_height and default suffix for this usage."""
        default_dict = {
            "text_width": 483.69687,
            "rescale_height": 1.0,
            "suffix": "pdf",
        }
        if self in (UsageType.DEFAULT, UsageType.THESIS, UsageType.PAPER_1COL):
            return default_dict
        if self is UsageType.PAPER_2COL:
            return default_dict | {"text_width": default_dict["text_width"] / 2.0}
        if self is UsageType.PRESENTATION:
            return {
                "text_width": 600.0,
                "rescale_height": 0.6,
                "suffix": "svg",
                "save_kwargs": {"transparent": True, "bbox_inches": "tight"},
            }
        if self is UsageType.BEAMER:
            return default_dict | {
                "text_width": 307.28987,
                "rescale_height": 0.6,
                "save_kwargs": {"transparent": True},
            }
        if self is UsageType.CUSTOM:
            return default_dict
        return default_dict


@dataclass
class PlotProfile:
    """A profile for configuring plot sizes and saving options.

    Examples
    --------
    >>> from plotm import PlotProfile
    >>> profile = PlotProfile(name='aa', layout='2col', font_size=8)
    >>> profile.to_yaml('my_profile.yaml')
    """

    name: str = "default"
    usage_type: str | UsageType = "default"
    layout: str | None = None
    layouts: dict[str, dict[str, Any]] = field(default_factory=dict)
    text_width: float | str = "paper"
    rescale_height: float = 1.0
    suffix: str | None = None
    save_kwargs: dict[str, Any] | None = None
    rc_params: dict[str, Any] | None = None
    font_size: int | None = None
    style_path: str | None = None

    def __init__(
        self,
        name: None | str = None,
        usage_type: str | None = None,
        layout: str | None = None,
        layouts: dict[str, dict[str, Any]] | None = None,
        text_width: float | None = None,
        rescale_height: float | None = None,
        suffix: str | None = None,
        save_kwargs: dict[str, Any] | None = None,
        rc_params: dict[str, Any] | None = None,
        font_size: int | None = None,
        style_path: str | None = None,
    ):
        self.name = self._get_name(name, usage_type)
        lookup_key = usage_type if usage_type is not None else self.name
        profile = ProfileManager.load(lookup_key, layout=layout)

        self.usage_type = profile.get("usage_type", "default")
        self.layouts = layouts if layouts is not None else profile.get("layouts", {})
        self.layout = layout or profile.get("layout", None)

        if self.layout and self.layouts and self.layout in self.layouts:
            layout_cfg = self.layouts[self.layout] or {}
            base_text_width = layout_cfg.get("text_width", profile.get("text_width", 483.69687))
            base_rescale_height = layout_cfg.get(
                "rescale_height", profile.get("rescale_height", 1.0)
            )
        else:
            base_text_width = profile.get("text_width", 483.69687)
            base_rescale_height = profile.get("rescale_height", 1.0)

        self.text_width = text_width if isinstance(text_width, (float, int)) else base_text_width
        self.rescale_height = (
            rescale_height if isinstance(rescale_height, (float, int)) else base_rescale_height
        )
        self.save_kwargs = (
            save_kwargs if isinstance(save_kwargs, dict) else profile.get("save_kwargs", {})
        )
        self.suffix = str(suffix) if isinstance(suffix, str) else profile.get("suffix", "pdf")
        self.rc_params = rc_params if isinstance(rc_params, dict) else profile.get("rc_params", {})
        self.font_size = font_size if isinstance(font_size, int) else profile.get("font_size", 12)
        self.style_path = style_path if isinstance(style_path, str) else profile.get("style_path")

    @classmethod
    def from_yaml(cls, path: str | Path):
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        profile_dict = dict(self.__dict__)
        if isinstance(self.usage_type, UsageType):
            profile_dict["usage_type"] = self.usage_type.value
        return profile_dict

    def to_yaml(self, path: str | Path):
        profile_dict = self.to_dict()
        with open(path, "w") as f:
            yaml.dump(profile_dict, f)

    def fig_size(
        self,
        nrows: int | tuple | list = 1,
        ncols: int = 1,
        layout: str | None = None,
        rescale_height: float = 1.0,
        fraction: float = 1.0,
        scale_factor: float = 1.0,
    ):
        nrows, ncols = self._maybe_unpack_rows_and_columns(nrows, ncols)

        active_layout = layout or self.layout
        tw = self.text_width
        rh = self.rescale_height

        if active_layout and self.layouts and active_layout in self.layouts:
            layout_cfg = self.layouts[active_layout] or {}
            tw = layout_cfg.get("text_width", tw)
            rh = layout_cfg.get("rescale_height", rh)
        elif active_layout and (not self.layouts or active_layout not in self.layouts):
            logger.warning(
                f"Layout '{active_layout}' not found in profile '{self.name}'. "
                f"Using default dimensions."
            )

        return set_size(
            subplots=(nrows, ncols),
            text_width=tw,
            rescale_height=rh * rescale_height,
            fraction=fraction,
            scale_factor=scale_factor,
        )

    @staticmethod
    def _get_name(
        name: None | str,
        usage_type: UsageType | str | None = None,
    ) -> str:
        if name is not None:
            name = str(name)
        elif name is None and isinstance(usage_type, UsageType):
            name = usage_type.value
        elif name is None and isinstance(usage_type, str):
            name = usage_type
        else:
            name = "default"
        return name

    def _set_usage_type_defaults(self):
        if isinstance(self.usage_type, UsageType):
            usage_type_defaults = self.usage_type.defaults()
        else:
            usage_type_defaults = UsageType.determine(self.usage_type).defaults()
        self.text_width = usage_type_defaults["text_width"]
        self.rescale_height = usage_type_defaults["rescale_height"]
        self.suffix = usage_type_defaults["suffix"]
        self.save_kwargs = usage_type_defaults.get("save_kwargs", {})

    @staticmethod
    def _maybe_unpack_rows_and_columns(
        nrows: int | tuple | list, ncols: int = 1
    ) -> tuple[int, int]:
        if isinstance(nrows, (list, tuple)):
            if len(nrows) == 2:
                nrows, ncols = nrows
            elif len(nrows) == 1:
                nrows = nrows[0]
            else:
                raise ValueError("nrows should be an integer")

        assert isinstance(nrows, int) and isinstance(ncols, int), "nrows and ncols must be integers"
        return nrows, ncols


class ProfileManager:
    """Discover and load PlotProfile YAML files from the package profiles directory."""

    @staticmethod
    def profiles() -> dict[str, Path]:
        """List all available profile YAML files."""
        return {p.name.removesuffix(".yaml"): p for p in PROFILES_DIR.glob("*.yaml")}

    @staticmethod
    def default_profiles() -> dict[str, Path]:
        """List all available default profile YAML files."""
        return {p.name.removesuffix(".yaml"): p for p in DEFAULT_PROFILES_DIR.glob("*.yaml")}

    @classmethod
    def load(cls, name: str | None, layout: str | None = None) -> dict:
        """Return profile with given name, or a default profile if not found."""
        if name is None:
            return cls._load_default_yaml()

        name = str(name).lower()

        profiles = cls.profiles()
        if name in profiles:
            return cls._load_yaml(profiles[name], layout=layout)

        default_profiles = cls.default_profiles()
        if name in default_profiles:
            return cls._load_yaml(default_profiles[name], layout=layout)

        # Check for {base}_{layout} pattern (e.g. aa_2col -> base 'aa' with layout '2col')
        if "_" in name:
            base_name, potential_layout = name.rsplit("_", 1)
            if base_name in profiles:
                return cls._load_yaml(profiles[base_name], layout=layout or potential_layout)
            if base_name in default_profiles:
                return cls._load_yaml(
                    default_profiles[base_name], layout=layout or potential_layout
                )

        logger.warning(f"Profile '{name}' not found. Falling back to default profile.")
        return cls._load_default_yaml()

    @classmethod
    def _load_default_yaml(cls) -> dict:
        """Return the default profile."""
        default_path = PROFILES_DIR / "default.yaml"

        if default_path.exists():
            with open(default_path, "r") as f:
                default_data = yaml.safe_load(f) or {}
        else:
            default_data = {
                "usage_type": "default",
                "font_size": 12,
                "text_width": 483.69687,
                "rescale_height": 1.0,
                "suffix": "pdf",
                "layout": None,
                "layouts": {
                    "1col": {},
                    "2col": {"text_width": 241.848435},
                },
                "save_kwargs": {},
                "rc_params": {},
            }

        return default_data

    @classmethod
    def _load_yaml(cls, path: str | Path, layout: str | None = None) -> dict:
        if not isinstance(path, Path):
            path = Path(path)

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        data["usage_type"] = path.stem

        default_data = cls._load_default_yaml()
        merged = default_data | data

        if layout is not None:
            merged["layout"] = layout

        if merged.get("style_path"):
            style_path = Path(merged["style_path"])
            if (PROFILES_DIR / style_path).exists():
                merged["style_path"] = str(PROFILES_DIR / style_path)
            elif style_path.is_absolute() and style_path.exists():
                merged["style_path"] = str(style_path)
            else:
                logger.warning(f"Style path '{merged['style_path']}' does not exist.")
                merged["style_path"] = None

        return merged

    def save_profile(self, profile: PlotProfile):
        """Save a PlotProfile to a YAML file."""
        profile_path = PROFILES_DIR / f"{profile.name}.yaml"

        if profile.style_path is not None:
            style_path = Path(profile.style_path)
            if style_path.is_relative_to(STYLES_DIR):
                profile.style_path = str(style_path.relative_to(STYLES_DIR))

        profile.to_yaml(profile_path)


def set_size(
    subplots=(1, 1),
    text_width: Union[float, str] = "paper",
    rescale_height: float = 1.0,
    fraction: float = 1.0,
    scale_factor: float = 1.0,
):
    """Set figure dimensions to avoid scaling in LaTeX.

    Based largely on Jack Walton's post on ploting figures with matplotlib and LaTeX:
    https://jwalton.info/Embed-Publication-Matplotlib-Latex/

    Parameters
    ----------
        text_width: float or string
                Document width in points, or string of predefined document type.
        fraction: float, optional
                Fraction of the width which you wish the figure to occupy.
        subplots: array-like, optional
                The number of rows and columns of subplots.
        scale_factor: float
            Factor to scale width and height with.
        rescale_height: float
            Factor to rescale height.

    Returns
    -------
        fig_dim: tuple
                Dimensions of figure in inches
    """
    if text_width == "paper":
        width_pt = 483.69687
    elif text_width == "beamer":
        width_pt = 307.28987
    elif text_width == "presentation":
        width_pt = 600
    elif isinstance(text_width, (float, int)):
        width_pt = text_width
    else:
        raise ValueError("Textwidth has to be 'paper', 'beamer', 'presentation' or a float.")

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (scale_factor * fig_width_in, rescale_height * scale_factor * fig_height_in)
