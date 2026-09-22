import pytest
import matplotlib.pyplot as plt
from plotm import PlotManager, PlotProfile


def test_aa_profile_layouts():
    prof_default = PlotProfile("aa")
    assert prof_default.name == "aa"
    assert prof_default.layout is None
    assert prof_default.text_width == 250.38
    assert prof_default.rc_params.get("axes.spines.right") is True

    prof_2col = PlotProfile("aa", layout="2col")
    assert prof_2col.layout == "2col"
    assert prof_2col.text_width == 512.15
    assert prof_2col.rescale_height == 1.0
    assert prof_2col.rc_params.get("axes.spines.right") is True

    prof_alias = PlotProfile("aa_2col")
    assert prof_alias.text_width == 512.15
    assert prof_alias.rc_params.get("axes.spines.right") is True


def test_paper_empty_1col_layout():
    prof = PlotProfile("paper")
    assert prof.layout is None
    assert prof.text_width == 483.69687

    # Explicitly requesting 1col should use top-level defaults
    size_base = prof.fig_size(1, 1)
    size_1col = prof.fig_size(1, 1, layout="1col")
    assert size_base == size_1col

    # 2col overrides width
    size_2col = prof.fig_size(1, 1, layout="2col")
    assert pytest.approx(size_2col[0], 0.01) == 241.848435 / 72.27


def test_fig_size_layout_override():
    prof = PlotProfile("aa")
    # Default without layout
    size_default = prof.fig_size(1, 1)
    # 1col layout (empty / inherits default)
    size_1col = prof.fig_size(1, 1, layout="1col")
    assert size_default == size_1col

    # Override with 2col on the fly
    size_2col = prof.fig_size(1, 1, layout="2col")
    assert size_2col[0] > size_1col[0]
    assert pytest.approx(size_2col[0], 0.01) == 512.15 / 72.27
    assert pytest.approx(size_1col[0], 0.01) == 250.38 / 72.27


def test_nonexistent_profile_does_not_raise():
    prof = PlotProfile("completely_fake_profile_123")
    assert prof.name == "completely_fake_profile_123"
    assert prof.text_width == 483.69687  # default fallback

    plm = PlotManager("nonexistent_profile_xyz")
    assert len(plm.profiles) == 1
    assert plm.profile.name == "nonexistent_profile_xyz"


def test_nonexistent_layout_does_not_raise():
    prof = PlotProfile("aa")
    size_default = prof.fig_size(1, 1)
    size_fake_layout = prof.fig_size(1, 1, layout="fake_layout_abc")
    assert size_default == size_fake_layout


def test_plot_manager_subplots_and_savefig(tmp_path):
    plm = PlotManager("aa", layout="2col", plot_dir=tmp_path, save=True)
    fig, axs = plm.subplots(1, 2)
    assert fig is not None

    # subplots with 1col layout override
    fig2, ax = plm.subplots(2, 1, layout="1col")
    assert fig2 is not None

    plm.savefig("test_fig")
    assert (tmp_path / "test_fig.pdf").exists()

    plt.close("all")


def test_plot_manager_figure_and_multi_profile_savefig(tmp_path):
    plm = PlotManager("aa", layout="2col", plot_dir=tmp_path, save=True)
    plm.add_profile("presentation")

    fig = plm.figure(layout="1_5col")
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1])

    plm.savefig("test_multi")
    assert (tmp_path / "test_multi.pdf").exists()
    assert (tmp_path / "test_multi_presentation.svg").exists()

    plt.close("all")


def test_resolve_profiles_invalid_name(tmp_path):
    plm = PlotManager("aa", plot_dir=tmp_path, save=True)
    fig, ax = plm.subplots()
    # Passing non-existent profile name should not raise
    plm.savefig("test_invalid", profiles="nonexistent_profile")
    plt.close("all")
