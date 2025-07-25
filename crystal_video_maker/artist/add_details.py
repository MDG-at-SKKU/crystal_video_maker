from typing import Any, Literal
from collections.abc import Callable, Sequence

from pymatgen.core import Structure

import plotly.graph_objects as go

def get_subplot_title(
    struct_i: Structure,
    struct_key: Any,
    idx: int,
    subplot_title: Callable[[Structure, str | int], str | dict[str, Any]] | None,
) -> dict[str, Any]:
    """Generate a subplot title based on the provided function or default logic."""
    title_dict: dict[str, str | float | dict[str, str | float]] = {}

    if callable(subplot_title):
        sub_title = subplot_title(struct_i, struct_key)
        if isinstance(sub_title, str | int | float):
            title_dict["text"] = str(sub_title)
        elif isinstance(sub_title, dict):
            title_dict |= sub_title
        else:
            raise TypeError(
                f"Invalid subplot_title, must be str or dict, got {sub_title}"
            )

    if not title_dict.get("text"):
        if isinstance(struct_key, int):
            spg_num = struct_i.get_symmetry_dataset()["number"]
            title_dict["text"] = f"{idx}. {struct_i.formula} (spg={spg_num})"
        elif isinstance(struct_key, str):
            title_dict["text"] = str(struct_key)
        else:
            raise TypeError(f"Invalid {struct_key=}. Must be an int or str.")

    return title_dict

def configure_subplot_legends(
    fig: go.Figure,
    site_labels: Literal["symbol", "species", "legend", False]
    | dict[str, str]
    | Sequence[str],
    n_structs: int,
    n_cols: int,
    n_rows: int,
) -> None:
    """Configure legends for each subplot if site_labels is 'legend'."""
    if site_labels == "legend":
        for idx in range(1, n_structs + 1):
            row = (idx - 1) // n_cols + 1
            col = (idx - 1) % n_cols + 1

            # Calculate position within each subplot (bottom right)
            x_start = (col - 1) / n_cols
            x_end = col / n_cols
            y_start = 1 - row / n_rows
            y_end = 1 - (row - 1) / n_rows

            # Position legend much closer to bottom right of subplot
            legend_x = x_start + 0.98 * (x_end - x_start)
            legend_y = y_start + 0.02 * (y_end - y_start)

            # Position legend much closer to bottom right of subplot
            legend_x = x_start + 0.98 * (x_end - x_start)
            legend_y = y_start + 0.02 * (y_end - y_start)

            legend_key = "legend" if idx == 1 else f"legend{idx}"
            legend_config = dict(
                x=legend_x,
                y=legend_y,
                xref="paper",
                yref="paper",
                xanchor="right",
                yanchor="bottom",
                bgcolor="rgba(0,0,0,0)",  # Transparent background
                borderwidth=0,  # Remove border
                font=dict(size=12, weight="bold"),  # Larger and bold font
                itemsizing="constant",  # Keep legend symbols same size
                itemwidth=30,  # Min allowed
                tracegroupgap=2,  # Reduce vertical space between legend items
            )
            fig.layout[legend_key] = legend_config