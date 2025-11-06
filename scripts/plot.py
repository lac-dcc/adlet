import os
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio
from plotly.subplots import make_subplots
from pathlib import Path

def method_label(row):
    if row["format"] == "dense":
        return "Dense"
    elif row["format"] == "sparse" and row["propagate"] == 0:
        return "Sparse (prop=false)"
    elif row["format"] == "sparse" and row["propagate"] == 1:
        return "Sparse (prop=true)"
    else:
        return "Other"

def figure12(result_path):
    #TODO: remove commented code
    def read_csv(file_name):
        df = pd.read_csv(file_name)
        df.columns = df.columns.str.strip()
        df["benchmark_id"] = df["file_name"].factorize()[0] + 1
        df["method"] = df.apply(method_label, axis=1)
        df = df.sort_values(by=['benchmark_id', 'format'])
        return df

    # script_path = os.path.dirname(os.path.realpath(__file__))
    # maps_path = os.path.join(script_path, "../analysis/einsum-data/")
    # files = os.listdir(maps_path)
    files = os.listdir(result_path)
    dfs = []
    for file in files:
        # if "complete.csv" not in file:
        #     continue
        # dfs.append(read_csv(f"{maps_path}/{file}"))
        dfs.append(read_csv(f"{result_path}/{file}"))

    df = pd.concat(dfs)

    dense_sizes = {}
    for benchmark_id in df['benchmark_id'].unique():
        dense_df = df[(df['benchmark_id'] == benchmark_id) & (df['format'] == 'dense')]
        if not dense_df.empty:
            dense_sizes[benchmark_id] = dense_df.iloc[0]['tensors-size']

    # Create a new column for normalized tensor size
    df['normalized_tensor_size'] = df.apply(
        lambda row: row['tensors-size'] / dense_sizes[row['benchmark_id']] 
        if row['benchmark_id'] in dense_sizes else np.nan, 
        axis=1
    )

    # Filter for sparse data only
    sparse_df = df[df['format'] == 'sparse'].copy()

    # Prepare data for plotting
    plot_data = []
    for benchmark_id in sorted(sparse_df['benchmark_id'].unique()):
        for sparsity in [0.3, 0.5, 0.7, 0.9]:
            sparse_no_prop = sparse_df[
                (sparse_df['benchmark_id'] == benchmark_id) & 
                (sparse_df['sparsity'] == sparsity) & 
                (sparse_df['propagate'] == 0)
            ]

            sparse_with_prop = sparse_df[
                (sparse_df['benchmark_id'] == benchmark_id) & 
                (sparse_df['sparsity'] == sparsity) & 
                (sparse_df['propagate'] == 1)
            ]

            if not sparse_no_prop.empty and not sparse_with_prop.empty:
                plot_data.append({
                    'benchmark_id': benchmark_id,
                    'sparsity': sparsity,
                    'sparse_no_prop_size': sparse_no_prop.iloc[0]['normalized_tensor_size'],
                    'sparse_with_prop_size': sparse_with_prop.iloc[0]['normalized_tensor_size'],
                    'sparse_no_prop_absolute': sparse_no_prop.iloc[0]['tensors-size'],
                    'sparse_with_prop_absolute': sparse_with_prop.iloc[0]['tensors-size'],
                    'dense_size': dense_sizes[benchmark_id]
                })

    plot_df = pd.DataFrame(plot_data)

    # Split benchmarks into two groups
    all_benchmarks = sorted(plot_df['benchmark_id'].unique())
    benchmarks_1 = all_benchmarks[:6]  # First 6 benchmarks
    benchmarks_2 = all_benchmarks[6:]  # Remaining benchmarks

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Benchmarks 1-6', 'Benchmarks 7-12'),
        vertical_spacing=0.1,
        row_heights=[0.5, 0.5]
    )
    fig.update_annotations(font_size=13) 

    # Colors and names
    sparsity_colors = {
        0.3: 'rgba(214, 39, 40, 0.8)',   # Red - 30%
        0.5: 'rgba(44, 160, 44, 0.8)',   # Green - 50% 
        0.7: 'rgba(255, 127, 14, 0.8)',  # Orange - 70%
        0.9: 'rgba(31, 119, 180, 0.8)'   # Blue - 90%
    }

    sparsity_names = {
        0.3: '30%',
        0.5: '50%',
        0.7: '70%', 
        0.9: '90%'
    }

    # Function to create plot for a subset of benchmarks
    def add_benchmark_subplot(fig, benchmarks, row):
        x_positions = []
        x_ticks = []
        benchmark_centers = []

        current_x = 0
        for benchmark_id in benchmarks:
            benchmark_data = plot_df[plot_df['benchmark_id'] == benchmark_id]

            benchmark_center = current_x + 1.5
            benchmark_centers.append(benchmark_center)

            for sparsity in [0.3, 0.5, 0.7, 0.9]:
                sparsity_data = benchmark_data[benchmark_data['sparsity'] == sparsity]
                if not sparsity_data.empty:
                    x_positions.append(current_x)
                    current_x += 1

            x_ticks.append(benchmark_center - 0.5)
            current_x += 1
        # Add bars for each sparsity level
        for sparsity in [0.3, 0.5, 0.7, 0.9]:
            sparsity_data = plot_df[(plot_df['sparsity'] == sparsity) & 
                                  (plot_df['benchmark_id'].isin(benchmarks))]
            x_pos = [x_positions[i] for i, row in enumerate(plot_df[plot_df['benchmark_id'].isin(benchmarks)].iterrows()) 
                    if row[1]['sparsity'] == sparsity]
            # Sparse without propagation
            fig.add_trace(go.Bar(
                x=x_pos,
                y=sparsity_data['sparse_no_prop_size'],
                name=f'{sparsity_names[sparsity]}' if row == 1 else '',
                marker_color=sparsity_colors[sparsity],
                marker_line_width=1.5,
                marker_line_color='darkgray',
                opacity=0.7,
                width=0.8,
                showlegend=(row == 1),
                hovertemplate=(
                    'Benchmark: %{customdata[0]}<br>' +
                    'Sparsity: %{customdata[1]}%<br>' +
                    'No Propagation: %{customdata[2]:.3f} (norm)<br>' +
                    '<extra></extra>'
                ),
                customdata=np.column_stack((
                    sparsity_data['benchmark_id'],
                    sparsity_data['sparsity'] * 100,
                    sparsity_data['sparse_no_prop_size']
                ))
            ), row=row, col=1)

            # Sparse with propagation
            fig.add_trace(go.Bar(
                x=x_pos,
                y=sparsity_data['sparse_with_prop_size'],
                name=f'{sparsity_names[sparsity]} (Prop)' if row == 1 else '',
                marker_color=sparsity_colors[sparsity],
                marker_line_width=1,
                marker_line_color='black',
                opacity=0.9,
                width=0.6,
                showlegend=(row == 1),
                hovertemplate=(
                    'Benchmark: %{customdata[0]}<br>' +
                    'Sparsity: %{customdata[1]}%<br>' +
                    'With Propagation: %{y:.3f} (normalized)<br>' +
                    '<extra></extra>'
                ),
                customdata=np.column_stack((
                    sparsity_data['benchmark_id'],
                    sparsity_data['sparsity'] * 100
                ))
            ), row=row, col=1)

        # Update x-axis for this subplot
        fig.update_xaxes(
            tickvals=x_ticks,
            ticktext=[str(bid) for bid in benchmarks],
            row=row, col=1,
            title_text="Benchmark ID" if row == 2 else ""
        )

        # Add horizontal line for dense baseline
        fig.add_hline(y=1, line_dash="dash", line_color="red", 
                     row=row, col=1)

    # Add both subplots
    add_benchmark_subplot(fig, benchmarks_1, 1)
    add_benchmark_subplot(fig, benchmarks_2, 2)

    # Customize x-axis
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
    )

    # Customize y-axis
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='black',
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=-0.07,
        y=0.5,
        text="Normalized Tensor Size (vs Dense)",
        showarrow=False,
        textangle=-90
    )

    fig.update_layout(
        barmode='overlay',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.10,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
            itemsizing="constant",
            entrywidth=60,
            valign="top",
        ),
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=0, b=0, l=0, r=40),
        font=dict(size=11)
    )

    fig.update_xaxes(
        title_font=dict(size=12),
        tickfont=dict(size=11)
    )
    pio.write_image(fig, f"{result_path}/figure.png", width=500, height=600, scale=5)

def figure10(result_path, file_name):
    df = pd.read_csv(file_name)
    df = df.sort_values(by=['row', 'col'])
    df['sparsity'] = df.apply(lambda row: f"({int(row['row'] * 100)}, {int(row['col'] * 100)})", axis=1)
    df['config'] = df.apply(lambda row: "Dense" if row['format'].strip() == 'DD' else( 'Sparse + Prop' if row['prop'] == 1 else 'Sparse'), axis=1)
    df = df.sort_values(by=['config'])
    fig = go.Figure()

    current_x = 0

    x_labels = []
    for config in df['config'].unique():
        x_positions = []
        x_labels = []
        for i in range(len(df[df['config'] == config])):
            x_positions.append(i * 4 + current_x)
            x_labels.append(i * 4 + 1)
        current_x += 1

        config_data = df[df['config'] == config]
        fig.add_trace(go.Bar(
            x=x_positions,
            y=config_data['run'],
            marker_line_width=1,
            marker_line_color='black',
            opacity=0.9,
            name=config,
            showlegend=True,
            hovertemplate=(
                'Sparsity: %{customdata[13]}<br>' +
                'Format: %{customdata[14]}<br>' +
                'Size: %{y:.3f}s<br>'
            ),
            customdata=config_data,
        ))

    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=x_labels,
            ticktext=[i for i in df['sparsity'].unique()],
            showgrid=True,
            gridwidth=1,
            gridcolor='#f0f0f0',
            linecolor='black',
            linewidth=2,
            tickfont=dict(size=13),
            title_text="% Sparsity (Row/Col)",
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='#f0f0f0',
            tickfont=dict(size=13),
            title_text="Runtime (s)",
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=13),
        width=500,
        height=250,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5,
            font=dict(size=13),
            itemsizing='trace',
        ),
        margin=dict(t=0, b=0, l=0, r=40),
    )
    pio.write_image(fig, f"{result_path}/figure.png", width=500, height=600, scale=5)

def figure7(result_path, file_name):
    df = pd.read_csv(file_name)
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
            y=-0.1,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
            itemsizing='trace',
        ),
        font=dict(size=9),
        hovermode='closest',
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
            tickfont=dict(size=13),
            title_text="Benchmark ID" if row == 2 else ""
        )

    # Customize y-axes with log scale
    fig.update_yaxes(
        type='log',
        showgrid=True,
        gridwidth=1,
        gridcolor='#f0f0f0',
        title_font=dict(size=15),
        tickfont=dict(size=13),
        row=1, col=1
    )

    fig.update_yaxes(
        type='log',
        showgrid=True,
        gridwidth=1,
        gridcolor='#f0f0f0',
        linewidth=1,
        title_font=dict(size=15),
        tickfont=dict(size=13),
        row=2, col=1
    )

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
            x=-0.2,
            y=0.5,
            text="Time (s) - Log Scale",
            showarrow=False,
            textangle=-90
    )
    fig.update_annotations(font_size=14)

    pio.write_image(fig, f"{result_path}/figure.png", width=500, height=600, scale=5)

def figure9(result_path, file_name):
    pass
