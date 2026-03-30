import plotly.graph_objects as go
import plotly.express as px


def plot_correlation_heatmap(correlations: dict, method: str = "pearson") -> go.Figure:
    """
    Plots an interactive correlation/association heatmap using Plotly.

    Args:
        correlations : output from correlations_main()
        method   : one of 'pearson', 'spearman', 'cramers'
    """
    method_map = {
        "pearson" : ("Pearson Correlation Heatmap",    (-1, 1), "RdBu_r" ),
        "spearman": ("Spearman Correlation Heatmap",   (-1, 1), "RdBu_r" ),
        "cramers" : ("Cramér's V Association Heatmap", (0,  1), "YlOrRd" )
    }

    if method not in method_map:
        raise ValueError(f"method must be one of: {list(method_map.keys())}")

    title, (vmin, vmax), colorscale = method_map[method]
    matrix = correlations[method]["matrix"]

    if matrix is None:
        print(f"No matrix available for method: {method}")
        return None

    matrix_float = matrix.astype(float)
    cols         = matrix_float.columns.tolist()

    fig = go.Figure(
        data=go.Heatmap(
            z            = matrix_float.values,
            x            = cols,
            y            = cols,
            colorscale   = colorscale,
            zmin         = vmin,
            zmax         = vmax,
            text         = matrix_float.round(2).values,     # value shown on hover
            texttemplate = "%{text}",                         # show value in cell
            hovertemplate= (
                "<b>%{y}  ↔  %{x}</b><br>"
                "Value: %{z:.4f}<extra></extra>"
            ),
            colorbar=dict(
                title      = "r value" if method != "cramers" else "V value",
                thickness  = 15,
                len        = 0.8
            )
        )
    )

    fig.update_layout(
        title       = dict(text=title, font=dict(size=16), x=0.5),
        xaxis       = dict(tickangle=-45, tickfont=dict(size=10)),
        yaxis       = dict(tickfont=dict(size=10), autorange="reversed"),
        width       = 650,
        height      = 600,
        plot_bgcolor= "white",
        margin      = dict(l=120, r=40, t=60, b=120)
    )

    return fig

def plot_significant_pairs_bar(correlations: dict, method: str = "pearson", top_n: int = 10) -> go.Figure:
    """
    Plots an interactive horizontal bar chart of top N significant pairs
    using Plotly.

    Args:
        correlations : output from correlations_main()
        method   : one of 'pearson', 'spearman', 'cramers'
        top_n    : number of top pairs to display
    """
    method_map = {
        "pearson" : ("abs_r", "Pearson |r|",  "Top Pearson Correlation Pairs",   "steelblue" ),
        "spearman": ("abs_r", "Spearman |r|", "Top Spearman Correlation Pairs",  "seagreen"  ),
        "cramers" : ("abs_v", "Cramér's V",   "Top Cramér's V Association Pairs","darkorange")
    }

    if method not in method_map:
        raise ValueError(f"method must be one of: {list(method_map.keys())}")

    val_key, xlabel, title, color = method_map[method]
    significant_pairs = correlations[method]["significant"]

    if not significant_pairs:
        print(f"No significant pairs found for method: {method}")
        return None

    # Take top N
    top_pairs = significant_pairs[:top_n]

    # Build data
    labels    = [f"{p['column_a']}  ↔  {p['column_b']}" for p in top_pairs]
    values    = [p[val_key]      for p in top_pairs]
    strengths = [p["strength"]   for p in top_pairs]
    p_values  = [p["p_value"]    for p in top_pairs]
    direction = [p.get("direction", "n/a") for p in top_pairs]

    threshold = correlations[method]["summary"]["threshold_used"]

    fig = go.Figure()

    # ── Bars ─────────────────────────────────────────────────────────────
    fig.add_trace(go.Bar(
        x           = values,
        y           = labels,
        orientation = "h",
        marker_color= color,
        opacity     = 0.85,
        customdata  = list(zip(strengths, p_values, direction)),
        hovertemplate=(
            "<b>%{y}</b><br>"
            f"{xlabel}: " + "%{x:.4f}<br>"
            "Strength  : %{customdata[0]}<br>"
            "P-value   : %{customdata[1]}<br>"
            "Direction : %{customdata[2]}"
            "<extra></extra>"
        ),
        text        = [f"{v:.2f}  ({s})" for v, s in zip(values, strengths)],
        textposition= "outside",
        textfont    = dict(size=10, color="dimgray")
    ))

    # ── Threshold line ────────────────────────────────────────────────────
    fig.add_vline(
        x           = threshold,
        line_dash   = "dash",
        line_color  = "red",
        line_width  = 1.5,
        annotation_text    = f"Threshold ({threshold})",
        annotation_position= "top right",
        annotation_font    = dict(size=10, color="red")
    )

    fig.update_layout(
        title      = dict(text=title, font=dict(size=16), x=0.5),
        xaxis      = dict(
            title      = xlabel,
            range      = [0, min(1.0, max(values) + 0.25)],
            tickfont   = dict(size=10),
            showgrid   = True,
            gridcolor  = "lightgrey",
            gridwidth  = 0.5
        ),
        yaxis      = dict(
            tickfont   = dict(size=10),
            autorange  = "reversed"    # strongest pair on top
        ),
        plot_bgcolor= "white",
        width      = 650,
        height     = max(400, len(top_pairs) * 50),
        margin     = dict(l=200, r=120, t=60, b=60)
    )

    return fig


def patterns_2_main(correlations: dict, top_n: int = 10) -> dict:

    return {
        "pearson_heatmap" : plot_correlation_heatmap(correlations, method="pearson"),
        "spearman_heatmap": plot_correlation_heatmap(correlations, method="spearman"),
        "cramers_heatmap" : plot_correlation_heatmap(correlations, method="cramers"),
        "pearson_bar"     : plot_significant_pairs_bar(correlations, method="pearson",  top_n=top_n),
        "spearman_bar"    : plot_significant_pairs_bar(correlations, method="spearman", top_n=top_n),
        "cramers_bar"     : plot_significant_pairs_bar(correlations, method="cramers",  top_n=top_n)
    }