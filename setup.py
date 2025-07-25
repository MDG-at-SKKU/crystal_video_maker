from setuptools import setup, find_packages

setup(
    name="crystal_video_maker",
    version="0.0.1",
    python_requires='>=3.10',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "plotly",
        "pymatgen",
        "pandas",
    ],
    entry_points={
        "console_scripts": [
            "",
        ],
    },
)