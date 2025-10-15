# `plotm`

`plotm` is a lightweight plotting utilities and profile manager for consistent figure sizing and saving.

## Main features
- Define plot profiles (size, saving options, rc params, style) as YAML files.
- Use multiple profiles simultaneously with PlotManager, computing figure sizes and other configurations when saving images.

## Installation
```bash
- pip install https://github.com/dgegen/plotm
```

## Usage
```python
from plotm import PlotManager

# Use a named profile (discovered from PlotManager.PROFILES_DIR)
plm = PlotManager(name="presentation", configure=True, plot_dir='.')

fig, axes = plm.subplots(nrows=1, ncols=1)
# ... draw on axes ...
plm.savefig("figure_name")
```

### Profiles
- Place profile YAML files under the package profiles directory used by the code (by default this is PROFILES_DIR; commonly `src/plotm/profiles/`).
- Filenames (without .yaml) are used as profile names. Profiles are merged with a default profile.

### Example profile

Profile that could be saved to `src/plotm/profiles/presentation.yaml`.
```yaml
text_width: 600.0,
rescale_height: 0.8,
suffix: svg
font_size: 12

save_kwargs: 
  transparent: true
  bbox_inches: tight"
```
