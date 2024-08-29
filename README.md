<p align="center">
  <img width="25%" src="public/icon.png" />
</p>

# Quokka - Interactive Volume visualization

[![PyPI](https://img.shields.io/pypi/v/quokka.svg)](https://pypi.org/project/quokka/)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)
](https://keksboter.github.io/quokka/docs/)

[👉 Click to run the web app 👈](https://keksboter.github.io/quokka)

<!--![Viewer screenshot](https://raw.githubusercontent.com/KeKsBoTer/quokka/master/screenshot.png)-->
![Viewer screenshot](images/screenshot_iso.png)
![Viewer screenshot](images/screenshot_volume.png)

## Installation

with pip
```
pip install quokka
```


## Usage

**Viewer** (jupyter notebook)
```python
from quokka import viewer
import numpy as np
from matplotlib import pyplot as plt

viewer(
    #[T,C,D,W,H]
    np.random.rand(1,1,32,32,32),
    plt.get_cmap("viridis"),   
)
```

**Render Image**
```python
import numpy as np
from quokka import diverging_alpha, render
import matplotlib.pyplot as plt

colormap = diverging_alpha(plt.get_cmap("magma"))
img = render(
        # [T,D,H,W]
        np.random.rand(2,32,32,32).astype(np.float32),
        colormap,
        0.5, # timestep
        width=1024,
        height=1024,
    )

plt.imshow(img)
plt.axis("off")

plt.savefig("test.png", bbox_inches="tight", pad_inches=0)
plt.show()
```


# View Development (Rust)

1. Install python and `matplotlib`,`seaborn` and `cmasher`
2. Run `python colormaps.py` to generate the colormaps
3. Run the viewer with `cargo run --bin quokka --features colormaps <volume_file>`