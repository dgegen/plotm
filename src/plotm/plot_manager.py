import logging
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import Any, Literal, Sequence, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from plotm.paths import PROFILES_DIR, STYLES_DIR
from plotm.plot_profile import PlotProfile

__all__ = ["PlotManager"]


DEFAULT_STYLE = STYLES_DIR / "default.mplstyle"

logger = logging.getLogger("plotm")


class PlotManager:
    """
    A manager for creating and saving plots with different profiles.

    Examples
    --------
    >>> from plotm import PlotManager
    >>> plm = PlotManager(name="paper", font_size=8, configure=True, save=True)
    >>> plm.add_profile(name='presentation')
    >>> fig, axes = plm.subplots(nrows=2, ncols=2)
    >>> plm.savefig('test_figure')

    """

    PROFILES_DIR = PROFILES_DIR

    def __init__(
        self,
        name: str | None = None,
        usage_type: str | None = None,
        text_width: float | None = None,
        rescale_height: float | None = None,
        suffix: str | None = None,
        save_kwargs: dict[str, Any] | None = None,
        rc_params: dict[str, Any] | None = None,
        font_size: int | None = None,
        style_path: str | None = None,
        plot_dir: str | Path | None = None,
        save: bool | None = None,
        configure: bool = False,
    ):
        self.profiles = [
            PlotProfile(
                name=name,
                usage_type=usage_type or name,
                text_width=text_width,
                rescale_height=rescale_height,
                suffix=suffix,
                save_kwargs=save_kwargs,
                rc_params=rc_params,
                font_size=font_size,
                style_path=style_path,
            )
        ]
        self._last_figsize_kwargs = {}

        self._plot_dir = Path(plot_dir) if plot_dir is not None else None
        if self._plot_dir is not None:
            self._plot_dir.mkdir(exist_ok=True)
            save = True if save is None else save

        self.save = save if save is not None else False

        if configure:
            self.configure_plotting()

    @property
    def plot_dir(self) -> Path | None:
        return self._plot_dir

    @plot_dir.setter
    def plot_dir(self, value: str | Path | None):
        if value is None:
            self._plot_dir = None
        else:
            self._plot_dir = Path(value)
            self._plot_dir.mkdir(exist_ok=True)

    def add_profile(
        self,
        name: str,
        usage_type: str | None = None,
        text_width: float | None = None,
        rescale_height: float | None = None,
        suffix: str | None = None,
        save_kwargs: dict[str, Any] | None = None,
        rc_params: dict[str, Any] | None = None,
        font_size: int | None = None,
        style_path: str | None = None,
        profile: PlotProfile | None = None,
    ):
        """Add a new profile to the manager."""
        if isinstance(profile, PlotProfile):
            self.profiles.append(profile)
        else:
            profile = PlotProfile(
                name=name,
                usage_type=usage_type or name,
                text_width=text_width,
                rescale_height=rescale_height,
                suffix=suffix,
                save_kwargs=save_kwargs,
                rc_params=rc_params,
                font_size=font_size,
                style_path=style_path,
            )
        self.profiles.append(profile)
        return profile

    @property
    def profile(self) -> PlotProfile:
        if len(self.profiles) == 0:
            raise ValueError("No profiles defined.")
        return self.profiles[0]

    @property
    def profile_names(self) -> list[str]:
        return [p.name for p in self.profiles]

    @property
    def profile_use_types(self) -> list[str]:
        return [p.usage_type.value for p in self.profiles]

    def fig_size(
        self,
        nrows: int | tuple | list = 1,
        ncols: int = 1,
        rescale_height: float = 1.0,
        fraction: float = 1.0,
        scale_factor: float = 1.0,
    ):
        self._last_figsize_kwargs: dict = {
            "nrows": nrows,
            "ncols": ncols,
            "rescale_height": rescale_height,
            "fraction": fraction,
            "scale_factor": scale_factor,
        }
        return self.profile.fig_size(
            nrows=nrows,
            ncols=ncols,
            rescale_height=rescale_height,
            fraction=fraction,
            scale_factor=scale_factor,
        )

    def savefig(
        self,
        filename: str | Path,
        save=None,
        profiles: None | str | PlotProfile | Sequence[str] | Sequence[PlotProfile] = None,
        **kwargs,
    ):
        if not (self.save or save):
            return

        sel_profiles = self._resolve_profiles(profiles)
        if len(sel_profiles) == 0:
            return

        fig = plt.gcf()
        orig_size = fig.get_size_inches().copy()

        # use first selected profile as primary when computing filename
        primary = sel_profiles[0]
        base_path = self.get_file_path(filename, profile=primary)

        # iterate the selected profiles (not all self.profiles)
        for idx, prof in enumerate(sel_profiles):
            if idx == 0:
                out_path = base_path
            else:
                out_path = base_path.with_name(f"{base_path.stem}_{prof.name}.{prof.suffix}")

            if isinstance(self._last_figsize_kwargs, dict):
                size = prof.fig_size(**self._last_figsize_kwargs)
            else:
                size = prof.fig_size()

            style_ctx = self.style_context(prof.style_path) if prof.style_path else nullcontext()
            rc_ctx = mpl.rc_context(rc=prof.rc_params or {})
            with style_ctx, rc_ctx:
                if prof.font_size is not None:
                    self._apply_font_size_to_fig(fig, prof.font_size)
                fig.set_size_inches(*size, forward=True)
                plt.savefig(out_path, **(prof.save_kwargs or {}) | kwargs)

        fig.set_size_inches(*orig_size, forward=True)

    def _apply_font_size_to_fig(self, fig: Figure, base_size: int) -> None:
        """Apply base font size to existing text artists in the figure."""
        # axes titles, labels, ticks, legends
        for ax in fig.axes:
            with suppress(Exception):
                ax.title.set_fontsize(base_size)  # type: ignore
            try:
                ax.xaxis.label.set_size(base_size)  # type: ignore
                ax.yaxis.label.set_size(base_size)  # type: ignore
            except Exception:
                pass
            with suppress(Exception):
                ax.tick_params(axis="both", labelsize=base_size)
            try:
                legend = ax.get_legend()
                if legend is not None:
                    for text in legend.get_texts():
                        text.set_fontsize(base_size)
            except Exception:
                pass

        # figure suptitle
        try:
            st = getattr(fig, "_suptitle", None)
            if st is not None:
                st.set_fontsize(base_size)
        except Exception:
            pass

    def get_file_path(self, filename: str | Path, profile: PlotProfile | None = None) -> Path:
        """Return a Path for filename using profile.suffix (falls back to self.profile)."""
        profile = profile or self.profile

        if not str(filename).endswith(f".{profile.suffix}"):
            if Path(filename).is_dir():
                logger.warning(
                    f"Given filename '{filename}' is a directory. "
                    f"Saving to 'unknown.{profile.suffix}' instead."
                )
                filename = Path(filename) / "unknown"
            filename = Path(f"{filename}.{profile.suffix}")

        base_path = (self.plot_dir / filename) if self.plot_dir is not None else Path(filename)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        return base_path

    def _resolve_profiles(
        self,
        profiles: None | str | PlotProfile | Sequence[str] | Sequence[PlotProfile] = None,
    ) -> list[PlotProfile]:
        """Resolve user selection to a list of PlotProfile objects.

        Accepts:
          - None -> all profiles
          - single profile name (str) -> profile with that name
          - list of names -> list of profiles in that order
          - PlotProfile or list of PlotProfile -> passed through

        Raises ValueError if a named profile is not found.
        """
        if profiles is None:
            return list(self.profiles)

        # single PlotProfile
        if isinstance(profiles, PlotProfile):
            return [profiles]

        # single string name
        if isinstance(profiles, str):
            name = profiles
            for p in self.profiles:
                if p.name == name:
                    return [p]
            logger.warning(f"No profile named '{name}'. Skipping savefig().")
            return []

        # sequence of PlotProfile objects
        if (
            isinstance(profiles, (list, tuple))
            and len(profiles) > 0
            and isinstance(profiles[0], PlotProfile)
        ):
            return list(profiles)  # type: ignore

        # sequence of names
        if isinstance(profiles, (list, tuple)) and all(isinstance(x, str) for x in profiles):
            resolved: list[PlotProfile] = []
            for name in profiles:
                for p in self.profiles:
                    if p.name == name:
                        resolved.append(p)
                        break
                else:
                    raise ValueError(f"No profile named '{name}'")
            return resolved

        raise ValueError("profiles must be None, a profile name, PlotProfile or a sequence thereof")

    def open_plot_dir(self):
        if self.plot_dir is None:
            raise ValueError("plot_dir is not set.")

        open_dir(self.plot_dir)

    def open_profiles_dir(self):
        open_dir(self.PROFILES_DIR)

    def figure(
        self,
        rescale_height: float = 1.0,
        fraction: float = 1.0,
        scale_factor: float = 1.0,
        **kwargs,
    ) -> Figure:
        kwargs["figsize"] = self.fig_size(
            rescale_height=rescale_height,
            fraction=fraction,
            scale_factor=scale_factor,
        )
        return plt.figure(**kwargs)

    def subplots(
        self,
        nrows: int = 1,
        ncols: int = 1,
        rescale_height: float = 1.0,
        fraction: float = 1.0,
        scale_factor: float = 1.0,
        sharex: bool | Literal["none", "all", "row", "col"] = False,
        sharey: bool | Literal["none", "all", "row", "col"] = False,
        squeeze: Literal[True] = True,
        width_ratios=None,
        height_ratios=None,
        subplot_kw: dict | None = None,
        gridspec_kw: dict | None = None,
        **kwargs,
    ):
        size = self.fig_size(
            nrows=nrows,
            ncols=ncols,
            rescale_height=rescale_height,
            fraction=fraction,
            scale_factor=scale_factor,
        )
        return plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=size,
            sharex=sharex,
            sharey=sharey,
            squeeze=squeeze,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
            subplot_kw=subplot_kw,  # type: ignore
            gridspec_kw=gridspec_kw,  # type: ignore
            **kwargs,
        )

    @staticmethod
    def set_font_sizes(base_size=10):
        import matplotlib as mpl

        mpl.rcParams.update(
            {
                "font.size": base_size,
                "axes.titlesize": base_size,
                "axes.labelsize": base_size,
                "xtick.labelsize": base_size,
                "ytick.labelsize": base_size,
                "legend.fontsize": base_size,
                "figure.titlesize": base_size,
            }
        )

    def use_style(self):
        plt.style.use(DEFAULT_STYLE if self.profile.style_path is None else self.profile.style_path)

    @staticmethod
    def style_context(style_path: str | None = None):
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            old_rc = plt.rcParams.copy()
            try:
                plt.style.use(style_path or DEFAULT_STYLE)
                yield
            finally:
                plt.rcParams.update(old_rc)

        return _ctx()

    def set_inline_backend(self, fmt: str = "svg") -> bool:
        """Set IPython inline backend figure format (e.g. 'svg' or 'png').
        Returns True if applied; False if not running inside IPython."""
        try:
            from IPython import get_ipython  # type: ignore
        except Exception:
            return False

        ip = get_ipython()
        if ip is None:
            return False

        fmt = str(fmt)
        # run_line_magic(magic_name, line)
        ip.run_line_magic("config", f"InlineBackend.figure_format = '{fmt}'")
        return True

    def configure_plotting(self):
        self.use_style()
        self.set_inline_backend()

    def __repr__(self) -> str:
        return f"PlotManager(profiles={self.profiles}, plot_dir='{self.plot_dir}, save={self.save})"


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


def open_dir(path: str | Path):
    import os
    import subprocess
    import sys

    if not Path(path).exists():
        raise ValueError(f"Path '{path}' does not exist.")

    if sys.platform == "win32":
        os.startfile(path)  # type: ignore
    elif sys.platform == "darwin":
        subprocess.run(["open", path])
    else:
        subprocess.run(["xdg-open", path])
