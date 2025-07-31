# Crystal Video Maker

Using plotting part of pymatvis source code
https://github.com/janosh/pymatviz/tree/main/pymatviz


## Work in process (WIP)
**STILL WORK IN PROCESS**

## Dependencies

Required:

- numpy
- plotly
- pymatgen
- tqdm
- moviepy
- imageio
- pillow (PIL)

Optional (for ASE/MSONAtoms support):

- ase
- pymatgen-io-ase


## Installation

```bash
git clone https://github.com/MDG-at-SKKU/crystal_video_maker.git
cd crystal_video_maker
pip install -e .
```


## Quick Start

```python
from crystal_video_maker import structure_3d
from crystal_video_maker.editor.image import prepare_image_byte
from crystal_video_maker.editor.media import video_maker

# Load or generate a list of pymatgen Structure objects
from pymatgen.core import Structure
struct1 = Structure.from_file("{structure_file1}")
struct2 = Structure.from_file("{structure_file2}")
struct3 = Structure.from_file("{structure_file3}")

structures = [struct1, struct2, struct3]

# or from XDATCAR
from pymatgen.io.vasp.outputs import Xdatcar
xdatcar = Xdatcar('XDATCAR_1')
structures = xdatcar.structures


# Render all structures into a single figure with subplots
fig = structure_3d(
    structures,
    atomic_radii=1.2,
    show_bonds=True,
    show_cell=True,
    n_cols=3,
    return_subplots_as_list=False
)
# Display in Jupyter
fig.show()

# Export as PNG bytes
png_bytes = prepare_image_byte(fig)

# Create an MP4 video from a sequence of figures
figs = structure_3d(
    structures,
    atomic_radii=1.2,
    show_bonds=True,
    show_cell=True
)
bytes_list = [prepare_image_byte(f) for f in figs]
video_path = video_maker(bytes_list, name="structures", fmt="mp4", fps=2)
print(f"Video saved to {video_path}")
```

### Progress bar and parallel processing

```python
from crystal_video_maker import structure_3d
from tqdm.contrib.concurrent import process_map

def render_one(struct):
    return structure_3d(
        [struct],
        atomic_radii=1.2,
        show_bonds=True,
        show_cell=True,
        use_internal_threads=False
    )

if __name__ == "__main__":
    # true parallelism on multiple cores
    figs = process_map(
        render_one,
        structures,
        max_workers=4,
        desc="Rendering structures"
    )
    # figs is a list of Plotly Figure objects
```


## Package Structure

```
crystal_video_maker/
├── common_types.py    # type aliases (Xyz, AnyStructure)
├── common_enum.py     # SiteCoords, LabelEnum
├── core/
│   ├── __init__.py
│   ├── structures.py
│   ├── geometry.py
│   └── structure_types.py
├── utils/
│   ├── __init__.py
│   ├── colors.py
│   ├── helpers.py
│   └── labels.py
├── rendering/
│   ├── __init__.py
│   ├── sites.py
│   ├── bonds.py
│   ├── cells.py
│   ├── vectors.py
│   └── styling.py
├── visualization/
│   ├── __init__.py
│   └── core.py
├── editor/
│   ├── __init__.py
│   ├── image.py
│   └── media.py
└── constants.py
```


## Link
[Materials Design Group @ SKKU](https://sites.google.com/site/jsparkphys/home)