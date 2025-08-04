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

## Main Functions

### Core Visualization

#### `structure_3d()`

Primary function for 3D crystal structure visualization.

```python
structure_3d(
    struct,                                    # Structure(s) to visualize
    *,
    # Layout and Resolution
    lattice_aspect=None,                       # Lattice aspect ratios (a,b,c)
    base_resolution=(800,800),                 # Base plot resolution
    n_cols=3,                                  # Subplot columns

    # Camera and View
    camera_config=None,                        # Custom camera configuration
    camera_preset="isometric",                 # Camera preset name
    crystal_view=None,                         # Crystal-specific view type

    # Atomic Visualization
    atomic_radii=None,                         # Atomic radii scaling/mapping
    atom_size=10,                             # Atom marker size
    scale=1,                                  # Overall scaling factor
    elem_colors=None,                         # Element color scheme

    # Structure Components
    show_sites=True,                          # Show atomic sites
    show_image_sites=True,                    # Show periodic image sites
    show_bonds=False,                         # Show chemical bonds
    show_cell=True,                           # Show unit cell
    show_cell_faces=True,                     # Show cell face surfaces

    # Labels and Text
    site_labels="legend",                     # Site labeling style
    show_subplot_titles=False,                # Show subplot titles
    subplot_title=None,                       # Custom title function

    # Advanced Features
    show_site_vectors=("force", "magmom"),    # Vector properties to display
    vector_kwargs=None,                       # Vector styling options
    hover_text=SiteCoords.cartesian_fractional, # Hover text format
    hover_float_fmt=".4",                     # Float formatting

    # Processing Options
    standardize_struct=None,                  # Standardize structures
    cell_boundary_tol=0.0,                    # Cell boundary tolerance
    bond_kwargs=None,                         # Bond styling options
    use_internal_threads=True,                # Use parallel processing

    # Output Options
    return_subplots_as_list=True,            # Return format
    return_json=False                         # Return JSON instead of Figure
) -> Union[go.Figure, List[go.Figure]]
```

**Parameters:**

- **struct**: Crystal structure(s) - accepts single Structure, dict, or sequence
- **lattice_aspect**: Tuple of (a,b,c) ratios for aspect-ratio-aware plotting
- **base_resolution**: Base plot dimensions (width, height) in pixels
- **camera_config**: Custom camera dictionary with 'eye', 'center', 'up' keys
- **camera_preset**: Predefined camera angles: "isometric", "top", "front", "side", "x_axis", "y_axis", "z_axis"
- **crystal_view**: Structure-aware views: "layer", "slab", "wire"
- **atomic_radii**: Scaling factor (float) or custom radii mapping (dict)
- **atom_size**: Base size for atomic markers
- **elem_colors**: Color scheme name ("VESTA", "Jmol", "CPK") or custom mapping
- **show_bonds**: Boolean, NearNeighbors instance, or per-structure dict
- **site_labels**: "symbol", "species", "legend", False, or custom mapping
- **show_site_vectors**: Vector properties to display as arrows
- **hover_text**: Coordinate format for hover tooltips
- **standardize_struct**: Apply crystallographic standardization
- **return_subplots_as_list**: Return individual figures vs. single subplot figure

**Returns:**

- Single `go.Figure` for single structure or combined subplot
- `List[go.Figure]` when `return_subplots_as_list=True`

### Media Creation

#### `video_maker()`

Create videos from image sequences.

```python
video_maker(
    image_bytes_list,          # List of image bytes
    name="video",              # Output filename
    fmt="mp4",                 # Video format
    fps=1,                     # Frames per second
    quality="standard",        # Quality setting
    output_dir="."            # Output directory
) -> str                      # Path to created video
```

#### `gif_maker()`

Create animated GIFs from image sequences.

```python
gif_maker(
    image_bytes_list,          # List of image bytes
    name="animation.gif",      # Output filename
    duration=100,              # Frame duration (ms)
    loop=0,                    # Loop count (0=infinite)
    optimize=True,             # Optimize file size
    output_dir="."            # Output directory
) -> str                      # Path to created GIF
```

#### `batch_save_images()`

Save multiple figures efficiently.

```python
batch_save_images(
    figs,                     # List of figures
    base_name="fig",          # Base filename
    result_dir="result",      # Output directory
    fmt="png",                # Image format
    numbering_rule="000",     # Numbering pattern
    parallel=True            # Use parallel processing
) -> List[str]               # List of saved filenames
```

## Utility Functions

### Color Management

#### `get_elem_colors()`

Get element color mappings.

```python
get_elem_colors(color_scheme="VESTA") -> Dict[str, str]
```

**Available schemes:**

- `"VESTA"`: Default VESTA colors
- `"Jmol"`: Jmol color scheme
- `"CPK"`: CPK color scheme
- Custom dict: `{"H": "#FFFFFF", "C": "#000000", ...}`

### Helper Functions

#### `get_atomic_radii()`

Get atomic radii with caching.

```python
get_atomic_radii(atomic_radii) -> Dict[str, float]
```

## Constants

### Element Properties

#### `ELEMENT_RADIUS`

Atomic radii database from VESTA.

```python
ELEMENT_RADIUS: Dict[str, float] = {
    "H": 0.46, "He": 1.22, "Li": 1.57, ...
}
```

#### Color Schemes

```python
VESTA_COLORS: Dict[str, str] = {"H": "#FFFFFF", ...}
JMOL_COLORS: Dict[str, str] = {"H": "#FFFFFF", ...}
CPK_COLORS: Dict[str, str] = {"H": "#FFFFFF", ...}
```

## Advanced Usage

### Custom Camera Configuration

```python
# Custom camera position
camera_config = {
    "eye": {"x": 2.0, "y": 2.0, "z": 2.0},
    "center": {"x": 0, "y": 0, "z": 0},
    "up": {"x": 0, "y": 0, "z": 1}
}

fig = structure_3d(struct, camera_config=camera_config)
```

### Crystal-Specific Views

```python
# Automatic view selection based on structure type
fig = structure_3d(layered_structure, crystal_view="layer")
fig = structure_3d(slab_structure, crystal_view="slab")
fig = structure_3d(chain_structure, crystal_view="wire")
```

### Aspect Ratio Control

```python
# For structures with extreme aspect ratios
lattice_aspect = (1.0, 1.0, 10.0)  # c-axis 10x longer
fig = structure_3d(struct, lattice_aspect=lattice_aspect)
```

## Link

[Materials Design Group @ SKKU](https://sites.google.com/site/jsparkphys/home)
