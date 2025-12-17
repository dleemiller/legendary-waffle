"""Analysis tools for swipe keyboard model research."""

from .attention_extractor import (
    AttentionHookManager,
    compute_char_to_path_attention_profile,
    extract_path_to_char_attention,
    extract_special_token_to_path_attention,
    get_attention_statistics,
    identify_dominant_head,
)
from .attention_visualizer import (
    create_attention_timeline_plot,
    create_layer_comparison_grid,
    create_layer_pooled_visualization,
    create_single_layer_timeline_plot,
    create_summary_visualization,
    plot_attention_heatmap_on_path,
)

__all__ = [
    # Extraction
    "AttentionHookManager",
    "compute_char_to_path_attention_profile",
    "extract_path_to_char_attention",
    "extract_special_token_to_path_attention",
    "identify_dominant_head",
    "get_attention_statistics",
    # Visualization
    "plot_attention_heatmap_on_path",
    "create_layer_comparison_grid",
    "create_summary_visualization",
    "create_layer_pooled_visualization",
    "create_single_layer_timeline_plot",
    "create_attention_timeline_plot",
]
