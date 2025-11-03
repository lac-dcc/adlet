import os
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio
from plotly.subplots import make_subplots

from run_einsum import RESULT_DIR

FIGURES_DIR = os.environ.get('FIGURES_DIR', 'figures/')


def method_label(row):
    if row["format"] == "dense":
        return "Dense"
    elif row["format"] == "sparse" and row["propagate"] == 0:
        return "Sparse (prop=false)"
    elif row["format"] == "sparse" and row["propagate"] == 1:
        return "Sparse (prop=true)"
    else:
        return "Other"

def figure7(file_name):
    df = pd.read_csv(f"{RESULT_DIR}/{file_name}")
    df.columns = df.columns.str.strip()
    df["benchmark_id"] = df["file_name"].factorize()[0] + 1
    df["method"] = df.apply(method_label, axis=1)
    df = df.sort_values(by=['benchmark_id', 'format'])

    df_sparse = df[df["method"] == "Sparse (prop=true)"]
    df_melt = df_sparse.melt(
        id_vars=["file_name", "method", "benchmark_id"],
        value_vars=["analysis", "compilation_time", "runtime"],
        var_name="metric",
        value_name="value"
    )
    
    color_map = {
        "analysis": "#2ca02c",  # Green for analysis
        "compilation_time": "#ff7f0e",  # Orange for compilation
        "runtime": "#d62728"  # Red for runtime
    }
    
    # Split benchmarks into two groups
    all_benchmarks = sorted(df_melt["benchmark_id"].unique())
    benchmarks_1 = all_benchmarks[:6]  # First 6 benchmarks
    benchmarks_2 = all_benchmarks[6:]  # Remaining benchmarks
    
    # Create subplots with 2 rows, 1 column
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Benchmarks 1-6', 'Benchmarks 7-12'),
        vertical_spacing=0.1,
        row_heights=[0.5, 0.5]
    )
    
    # Function to add bars to a subplot
    def add_benchmark_subplot(fig, benchmarks, row):
        for metric in ["analysis", "compilation_time", "runtime"]:
            metric_data = df_melt[(df_melt["metric"] == metric) & 
                                (df_melt["benchmark_id"].isin(benchmarks))]
            
            x_positions = []
            benchmark_ticks = []
            current_x = 0
            
            for benchmark_id in benchmarks:
                x_positions.extend([current_x, current_x + 1, current_x + 2])
                benchmark_ticks.append(current_x + 1)
                current_x += 4
            y_values = []
            for benchmark_id in benchmarks:
                benchmark_data = metric_data[metric_data["benchmark_id"] == benchmark_id]
                if not benchmark_data.empty:
                    y_values.append(benchmark_data["value"].iloc[0])
                else:
                    y_values.append(0)
            
            fig.add_trace(go.Bar(
                x=[i + (list(color_map.keys()).index(metric)) for i in range(0, len(benchmarks) * 4, 4)],
                y=y_values,
                name=metric.replace("_", " ").title(),
                marker_color=color_map[metric],
                marker_line_width=1,
                marker_line_color='black',
                opacity=0.9,
                width=0.8,
                showlegend=(row == 1),
                hovertemplate=(
                    'Benchmark: %{customdata}<br>' +
                    'Metric: ' + metric.replace("_", " ").title() + '<br>' +
                    'Time: %{y:.3f}s<br>' +
                    '<extra></extra>'
                ),
                customdata=benchmarks
            ), row=row, col=1)
    
    add_benchmark_subplot(fig, benchmarks_1, 1)
    add_benchmark_subplot(fig, benchmarks_2, 2)
    
    fig.update_layout(
        title=dict(
            text="Analysis Time vs Compilation Time vs Runtime (propagation=true)",
            x=0.5,
            xanchor='center',
            font=dict(size=14, color='#2c3e50'),
            y=0.95
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.07,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
            itemsizing='trace',
        ),
        margin=dict(t=0, b=0, l=0, r=50),
        font=dict(size=9),
        hovermode='closest',
        width=500,
        height=600
    )

    for row, benchmarks in enumerate([benchmarks_1, benchmarks_2], 1):
        fig.update_xaxes(
            tickvals=list(range(1, len(benchmarks) * 4, 4)),
            ticktext=[str(bid) for bid in benchmarks],
            row=row, col=1,
            showgrid=True,
            gridwidth=2,
            gridcolor='#f0f0f0',
            linecolor='black',
            linewidth=2,
            tickfont=dict(size=9),
            title_text="Benchmark ID" if row == 2 else ""
        )

    # Customize y-axes with log scale
    fig.update_yaxes(
        type='log',
        showgrid=True,
        gridwidth=1,
        gridcolor='#f0f0f0',
        title_font=dict(size=12),
        tickfont=dict(size=11),
        row=1, col=1
    )

    fig.update_yaxes(
        type='log',
        showgrid=True,
        gridwidth=1,
        gridcolor='#f0f0f0',
        linewidth=1,
        title_font=dict(size=12),
        tickfont=dict(size=11),
        row=2, col=1
    )

    # Improve legend labels
    fig.update_traces(
        name='Analysis',
        selector=dict(name='Analysis')
    )
    fig.update_traces(
        name='Compilation', 
        selector=dict(name='Compilation Time')
    )
    fig.update_traces(
        name='Exeuction',
        selector=dict(name='Runtime')
    )
    fig.add_annotation(
            xref="paper",
            yref="paper",
            x=-0.1,
            y=0.5,
            text="Time (s) - Log Scale",
            showarrow=False,
            textangle=-90
    )
    fig.update_annotations(font_size=12)

    pio.write_image(fig, f"./{FIGURES_DIR}/figure7.png", width=500, height=600)

