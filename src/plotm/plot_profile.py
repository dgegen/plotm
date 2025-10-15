import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Union

import yaml

from plotm.paths import DEFAULT_PROFILES_DIR, PROFILES_DIR, STYLES_DIR

__all__ = ["PlotProfile"]

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
    >>> from lightkite.utils.plot import PlotProfile
    >>> profile = PlotProfile(name='paper', usage_type='paper_2col', font_size=8)
    >>> profile.to_yaml('my_profile.yaml')
    """

    name: str = "default"
    usage_type: UsageType = UsageType.DEFAULT
    text_width: float | str = "paper"
    rescale_height: float = 1.0
    suffix: str | None = None
    save_kwargs: dict[str, Any] | None = None
    rc_params: dict[str, Any] | None = None
    font_size: int | None = None
    style_path: str | None = None

    def __init__(
        self,
        name: None | str,
        usage_type: str | None = None,
        text_width: float | None = None,
        rescale_height: float | None = None,
        suffix: str | None = None,
        save_kwargs: dict[str, Any] | None = None,
        rc_params: dict[str, Any] | None = None,
        font_size: int | None = None,
        style_path: str | None = None,
    ):
        self.name = self._get_name(name, usage_type)
        # self.usage_type = UsageType.determine(self.name if usage_type is None else usage_type)
        profile = ProfileManager.load(usage_type if usage_type is not None else self.name)

        self.suffix = name if suffix is None else suffix
        self.usage_type = profile["usage_type"]

        self.text_width = (
            text_width if isinstance(text_width, (float, int)) else profile["text_width"]
        )
        self.rescale_height = (
            rescale_height
            if isinstance(rescale_height, (float, int))
            else profile["rescale_height"]
        )
        self.save_kwargs = (
            save_kwargs if isinstance(save_kwargs, dict) else profile.get("save_kwargs", {})
        )
        self.suffix = str(suffix) if isinstance(suffix, str) else profile.get("suffix", "pdf")
        self.rc_params = rc_params if isinstance(rc_params, dict) else profile.get("rc_params", {})
        self.font_size = font_size if isinstance(font_size, int) else profile["font_size"]

        self.style_path = style_path if isinstance(style_path, str) else profile.get("style_path")

    @classmethod
    def from_yaml(cls, path: str | Path):
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        profile_dict = self.__dict__
        profile_dict["usage_type"] = self.usage_type.value
        return profile_dict

    def to_yaml(self, path: str | Path):
        import yaml

        profile_dict = self.to_dict()

        with open(path, "w") as f:
            yaml.dump(profile_dict, f)

    def fig_size(
        self,
        nrows: int | tuple | list = 1,
        ncols: int = 1,
        rescale_height: float = 1.0,
        fraction: float = 1.0,
        scale_factor: float = 1.0,
    ):
        nrows, ncols = self._maybe_unpack_rows_and_columns(nrows, ncols)

        return set_size(
            subplots=(nrows, ncols),
            text_width=self.text_width,
            rescale_height=self.rescale_height * rescale_height,
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
        else:
            name = "default"
        return name

    def _set_usage_type_defaults(self):
        usage_type_defaults = self.usage_type.defaults()
        self.text_width = usage_type_defaults["text_width"]
        self.rescale_height = usage_type_defaults["rescale_height"]
        self.suffix = usage_type_defaults["suffix"]
        self.save_kwargs = usage_type_defaults.get("save_kwargs", {})

    @staticmethod
    def _maybe_unpack_rows_and_columns(
        nrows: int | tuple | list, ncols: int = 1
    ) -> tuple[int, int]:
        # args should be either a tuple of (nrows, ncols) or two integers
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
    def load(cls, name: str) -> dict:
        """Return profile with given name, or a default profile if not found."""
        name = str(name).lower()

        profiles = cls.profiles()
        if name in profiles:
            return cls._load_yaml(profiles[name])

        default_profiles = cls.default_profiles()
        if name in default_profiles:
            return cls._load_yaml(default_profiles[name])

        return cls._load_default_yaml()

    @classmethod
    def _load_default_yaml(cls) -> dict:
        """Return the default profile."""
        default_path = PROFILES_DIR / "default.yaml"

        if default_path.exists():
            with open(default_path, "r") as f:
                default_data = yaml.safe_load(f)
        else:
            default_data = {
                "usage_type": "default",
                "text_width": 483.69687,
                "rescale_height": 1.0,
                "suffix": "pdf",
                "save_kwargs": {},
            }

        return default_data

    @classmethod
    def _load_yaml(cls, path: str | Path) -> dict:
        if not isinstance(path, Path):
            path = Path(path)

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        data["usage_type"] = path.stem

        data = cls._load_default_yaml() | data
        
        if "style_path" in data:
            style_path = Path(data["style_path"])
            if (PROFILES_DIR / style_path).exists():
                data["style_path"] = str(PROFILES_DIR / style_path)
            elif style_path.is_absolute() and style_path.exists():
                data["style_path"] = str(style_path)
            else:
                logger.warning(f"Style path '{data['style_path']}' does not exist.")
                data["style_path"] = None
            

        return data

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
            Facto to scale width and height with.
        rescale_height: float
            Factor to rescale height.

    Returns
    -------
        fig_dim: tuple
                Dimensions of figure in inches
    """
    if text_width == "paper":
        # Textwidth of LaTeX file. Can be determined by typing
        # \the\text_width
        # in your latex file and then compiling.
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
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (scale_factor * fig_width_in, rescale_height * scale_factor * fig_height_in)
