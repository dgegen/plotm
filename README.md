# plot_manager

Lightweight plotting utilities and profile manager for consistent figure sizing and saving.

Features
- Define plot profiles (size, saving options, rc params, style) as YAML files.
- Profiles discovered at runtime from the package profiles directory (PROFILES_DIR / DEFAULT_PROFILES_DIR).
- PlotProfile and PlotManager helpers to compute figure sizes and save figures consistently.

Install (development)
- From project root:
  - pip install -e .

Quick usage
```python
from plotm import PlotManager

# Use a named profile (discovered from PlotManager.PROFILES_DIR)
plm = PlotManager(name="presentation", save=True, configure=True)

fig, axes = plm.subplots(nrows=1, ncols=1)
# ... draw on axes ...
plm.savefig("figure_name")
```

Profiles (YAML)
- Place profile YAML files under the package profiles directory used by the code (by default this is PROFILES_DIR; commonly `src/plotm/profiles/`).
- Filenames (without .yaml) are used as profile names. Profiles are merged with a default profile.

Example profile (save as e.g. `src/plotm/profiles/presentation.yaml`)
```yaml
text_width: 600.0,
rescale_height: 0.8,
suffix: svg
font_size: 12

save_kwargs: 
  transparent: true
  bbox_inches: tight"
```
