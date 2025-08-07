from setuptools import setup, find_packages

setup(
    name="crystal_video_maker",
    version="0.0.1",
    python_requires=">=3.12",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "plotly",
        "pymatgen",
        "moviepy",
        "imageio",
        "pillow",
        "ase",
        "kaleido >=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "",
        ],
    },
)
