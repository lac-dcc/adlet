import os
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio
from plotly.subplots import make_subplots
from pathlib import Path
import glob

EINSUM_ORDER = [
    "lm_batch_likelihood_brackets_3_16d.txt",
    "lm_batch_likelihood_sentence_3_12d.txt",
    "str_nw_mera_open_26.txt",
    "lm_batch_likelihood_sentence_4_8d.txt",
    "str_nw_ftps_open_30.txt",
    "str_matrix_chain_multiplication_100.txt",
    "str_nw_ftps_open_28.txt",
    "lm_batch_likelihood_sentence_4_4d.txt",
    "str_mps_varying_inner_product_200.txt",
    "str_nw_mera_closed_120.txt",
    "gm_queen5_5_3.wcsp.txt",
    "str_matrix_chain_multiplication_1000.txt"
]

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
    def read_csv(file_name):
        df = pd.read_csv(file_name)
        df.columns = df.columns.str.strip()
        df["benchmark_id"] = df["file_name"].factorize()[0] + 1
        df["method"] = df.apply(method_label, axis=1)
        df = df.sort_values(by=['benchmark_id', 'format'])
        return df

    files = os.listdir(result_path)
    dfs = []
    for file in files:
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

def figure8(result_path):
    spa_files = glob.glob('results/figure8/proptime_spa_result_*.csv')
    tesa_files = glob.glob('results/figure8/proptime_tesa_result_*.csv')

    spa_data = []
    tesa_data = []
    for file in spa_files:
        df = pd.read_csv(file)
        spa_data.append({
            'size': df['size'].iloc[0],
            'proptime': df['proptime'].iloc[0]
        })
    for file in tesa_files:
        df = pd.read_csv(file)
        tesa_data.append({
            'size': df['size'].iloc[0],
            'proptime': df['proptime'].iloc[0]
        })

    spa_df = pd.DataFrame(spa_data).sort_values('size').reset_index(drop=True)
    tesa_df = pd.DataFrame(tesa_data).sort_values('size').reset_index(drop=True)

    fig = go.Figure()
            
    fig.add_trace(
        go.Scatter(
            x=tesa_df['size'],
            y=tesa_df['proptime'],
            mode='lines+markers',
            name='SparTA',
            line=dict(color='blue', width=2),
            marker=dict(size=8, color='blue'),
            hovertemplate='<b>TeSA</b><br>' +
                         'Size: %{x}<br>' +
                         'Time: %{y:.6e}s<br>' +
                         '<extra></extra>'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=spa_df['size'],
            y=spa_df['proptime'],
            mode='lines+markers',
            name='SPA',
            line=dict(color='red', width=2),
            marker=dict(size=8, color='red'),
            hovertemplate='<b>SPA</b><br>' +
                         'Size: %{x}<br>' +
                         'Prop: %{y:.6e}<br>' +
                         '<extra></extra>'
        )
    )
            
    fig.update_xaxes(
        title_text="Size",
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
            
    fig.update_yaxes(
        title_text="Time (s) - (Log Scale)",
        type="log",
        tickformat='.2e',  # Scientific notation
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
            
    fig.update_layout(
        title={
            'text': 'Analysis Time of SPA vs SparTA',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        width=1000,
        height=600,
        hovermode='x unified',
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
            
    pio.write_image(fig, f"{result_path}/figure.png", width=500, height=500, scale=5)
    print(f"Plot saved as {result_path}/figure.png")
    
    

def figure9(result_path):
    def get_last_nonzero_ratios(group):
        ratios = []
        
        for col in ['fw_ratio', 'lat_ratio', 'bw_ratio']:
            nonzero_values = group[group[col] != 0][col]
            if not nonzero_values.empty:
                ratios.append(nonzero_values.iloc[-1])

        return max(ratios) if ratios else 0

    def get_config_label(config_string):
        parts = config_string.split(', ')
        
        fw = '1' in parts[0]  # run_fw
        lat = '1' in parts[1]  # run_lat  
        bw = '1' in parts[2]  # run_bw
        
        label_parts = []
        if fw:
            label_parts.append('F')
        if lat:
            label_parts.append('L') 
        if bw:
            label_parts.append('B')
        
        if not label_parts:
            return "No Prop"
        
        return f"{''.join(label_parts)}"

    def create_plotly_bar_plot(data, sparsity_level):
        data = data.sort_values('benchmark_id')
        
        unique_benchmarks = sorted(data['benchmark_id'].unique())
        unique_configs = sorted(data['config'].unique())
        
        # Color scheme similar to your second file
        #gray, red, blue, yellow, green
        colors = ['#7f7f7f','#d62728', '#ff7f0e','#1f77b4', '#2ca02c',]
        # Split benchmarks into two groups
        benchmarks_1 = unique_benchmarks[:6]  # First 6 benchmarks
        benchmarks_2 = unique_benchmarks[6:]  # Remaining benchmarks
        
        # Create subplots with 2 rows, 1 column
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Benchmarks 1-6', 'Benchmarks 7-12'),
            vertical_spacing=0.08,
            row_heights=[0.5, 0.5]
        )
        
        # Function to add bars to a subplot
        def add_benchmark_subplot(benchmarks, row):
            for i, config in enumerate(unique_configs):
                config_data = data[data['config'] == config].sort_values('benchmark_id')
                
                # Get values for each benchmark in this group
                values = []
                benchmark_labels = []
                for bench_id in benchmarks:
                    bench_data = config_data[config_data['benchmark_id'] == bench_id]
                    if not bench_data.empty:
                        values.append(bench_data['val'].iloc[0])
                        benchmark_labels.append(bench_data['filename'].iloc[0])
                    else:
                        values.append(0)
                        benchmark_labels.append(f"Benchmark {bench_id}")
                
                # Calculate x positions for this configuration
                n_configs = len(unique_configs)
                bar_width = 0.13
                x_offset = (i - n_configs/2 + 0.5) * bar_width
                x_positions = [x + x_offset for x in range(len(benchmarks))]
                
                # Create bars
                fig.add_trace(go.Bar(
                    x=[0, 1, 2, 3, 4, 5, 6],
                    y=values,
                    name=get_config_label(config),
                    marker_color=colors[i % len(colors)],
                    marker_line_width=1,
                    marker_line_color='black',
                    opacity=0.9,
                    width=bar_width,
                    showlegend=(row == 1),  # Only show legend for first row
                    hovertemplate=(
                        'Benchmark: %{customdata}<br>' +
                        'Config: ' + get_config_label(config) + '<br>' +
                        'Ratio: %{y:.2f}%<br>' +
                        '<extra></extra>'
                    ),
                    customdata=benchmark_labels
                ), row=row, col=1)
        
        # Add both subplots
        add_benchmark_subplot(benchmarks_1, 1)
        add_benchmark_subplot(benchmarks_2, 2)
        
        # Apply professional styling
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.09,
                xanchor="center",
                x=0.5,
                font=dict(size=11),
                itemsizing='constant',
            ),
            #aqui
            margin=dict(t=30, b=120, l=60, r=50),
            font=dict(size=12),
            hovermode='closest',
            width=800,
            height=1200
        )
        
        # Customize x-axes for both subplots
        fig.update_xaxes(
            tickvals=list(range(len(benchmarks_1))),
            ticktext=[str(bid) for bid in benchmarks_1],
            row=1, col=1,
            showgrid=True,
            gridwidth=2,
            gridcolor='#f0f0f0',
            linecolor='black',
            linewidth=2,
            tickfont=dict(size=12)
        )
        
        fig.update_xaxes(
            tickvals=list(range(len(benchmarks_2))),
            ticktext=[str(bid) for bid in benchmarks_2],
            row=2, col=1,
            showgrid=True,
            gridwidth=2,
            gridcolor='#f0f0f0',
            linecolor='black',
            linewidth=2,
            tickfont=dict(size=12),
            title_text="Benchmark ID"
        )

        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='#f0f0f0',
            #linecolor='black',
            linewidth=0,
            tickfont=dict(size=14),
            row=1, col=1
        )
        
        # Add single Y-axis label using annotation
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=-0.09,
            y=0.5,
            text="Ratio (%)",
            showarrow=False,
            textangle=-90,
            font=dict(size=16)
        )
        
        return fig

    def create_multi_sparsity_plot(sparsity_data_dict):
        """Create subplot with multiple sparsity levels"""
        n_plots = len(sparsity_data_dict)
        
        if n_plots <= 2:
            rows, cols = 1, n_plots
            subplot_titles = list(sparsity_data_dict.keys())
        else:
            rows = 2
            cols = (n_plots + 1) // 2
            subplot_titles = list(sparsity_data_dict.keys())
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f'Sparsity: {title}' for title in subplot_titles],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        colors = ['#7f7f7f','#d62728', '#ff7f0e','#1f77b4', '#2ca02c',]
        
        for idx, (sparsity_level, data) in enumerate(sparsity_data_dict.items()):
            row = (idx // cols) + 1
            col = (idx % cols) + 1
            
            # Sort by benchmark_id for consistent ordering
            data = data.sort_values('benchmark_id')
            unique_benchmarks = sorted(data['benchmark_id'].unique())
            unique_configs = sorted(data['config'].unique())
            
            n_configs = len(unique_configs)
            bar_width = 0.15
            
            for i, config in enumerate(unique_configs):
                config_data = data[data['config'] == config].sort_values('benchmark_id')
                
                # Get values for each benchmark
                values = []
                benchmark_labels = []
                for bench_id in unique_benchmarks:
                    bench_data = config_data[config_data['benchmark_id'] == bench_id]
                    if not bench_data.empty:
                        values.append(bench_data['val'].iloc[0])
                        benchmark_labels.append(bench_data['filename'].iloc[0])
                    else:
                        values.append(0)
                        benchmark_labels.append(f"Benchmark {bench_id}")
                
                # Calculate x positions for this configuration
                x_offset = (i - n_configs/2 + 0.5) * bar_width
                x_positions = [x + x_offset for x in range(len(unique_benchmarks))]
                
                # Create bars
                fig.add_trace(go.Bar(
                    x=x_positions,
                    y=values,
                    name=get_config_label(config) if idx == 0 else get_config_label(config),  # Show legend only for first subplot
                    marker_color=colors[i % len(colors)],
                    marker_line_width=1,
                    marker_line_color='black',
                    opacity=0.9,
                    width=bar_width,
                    showlegend=(idx == 0),  # Only show legend for first subplot
                    hovertemplate=(
                        'Benchmark: %{customdata}<br>' +
                        f'Config: {get_config_label(config)}<br>' +
                        'Ratio: %{y:.2f}%<br>' +
                        '<extra></extra>'
                    ),
                    customdata=benchmark_labels
                ), row=row, col=col)
            
            # Update x-axis for this subplot
            fig.update_xaxes(
                tickvals=list(range(len(unique_benchmarks))),
                ticktext=[str(bid) for bid in unique_benchmarks],
                showgrid=True,
                gridwidth=1,
                gridcolor='#f0f0f0',
                linecolor='black',
                linewidth=1,
                tickfont=dict(size=9),
                title_text="Benchmark ID" if row == rows else "",
                row=row, col=col
            )
            
            # Update y-axis for this subplot
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='#f0f0f0',
                linecolor='black',
                linewidth=1,
                tickfont=dict(size=9),
                title_text="Ratio (%)" if col == 1 else "",
                row=row, col=col
            )
        
        # Apply professional styling
        fig.update_layout(
            title=dict(
                text="Sparsity Analysis - Ratio Comparison Across Configurations",
                x=0.5,
                xanchor='center',
                font=dict(size=16, color='#2c3e50'),
                y=0.95
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.05,
                xanchor="center",
                x=0.5,
                font=dict(size=9),
                itemsizing='constant',
            ),
            margin=dict(t=100, b=120, l=60, r=50),
            font=dict(size=10),
            hovermode='closest',
            width=1000,
            height=600 if rows == 1 else 800
        )
        
        return fig

    order_dict = {filename: idx for idx, filename in enumerate(EINSUM_ORDER)}
    result_files = sorted(glob.glob(result_path + "/*.csv"))

    sparsity_data = {}
    for i, (level, filepath) in enumerate(zip([0.3, 0.5, 0.7, 0.9], result_files)):
        print(filepath, level)
        assert str(level) in filepath
        df = pd.read_csv(filepath)

        filtered = df[(df["run_fw"] == 1) & (df["run_lat"] == 1) & (df["run_bw"] == 1)]

        filtered = filtered.copy()
        filtered['diff'] = filtered['bw_ratio'] - filtered['initial_ratio']
        
        max_diff_row = filtered.loc[filtered['diff'].idxmax()]
            
        valid_files = set(order_dict.keys())
        grouped_data = []
        
        for filename in df['file_name'].unique():
            if valid_files and filename not in valid_files:
                continue
                
            file_data = df[df['file_name'] == filename]
            benchmark_id = order_dict[filename]
            
            # Get unique combinations of run flags
            combinations = file_data[['run_fw', 'run_lat', 'run_bw']].drop_duplicates()
            for _, combo in combinations.iterrows():
                combo_data = file_data[
                    (file_data['run_fw'] == combo['run_fw']) &
                    (file_data['run_lat'] == combo['run_lat']) &
                    (file_data['run_bw'] == combo['run_bw'])
                ]

                if not combo_data.empty:
                    val = get_last_nonzero_ratios(combo_data)
                    config = f"run_fw={combo['run_fw']}, run_lat={combo['run_lat']}, run_bw={combo['run_bw']}"
                    grouped_data.append({
                        'benchmark_id': benchmark_id + 1,
                        'filename': filename,
                        'config': config,
                        'val': val * 100, 
                    })

            grouped_data.append({
                'benchmark_id': benchmark_id + 1,
                'filename': filename,
                'config': f"run_fw=0, run_lat=0, run_bw=0",
                'val': combo_data["initial_ratio"].item() * 100, 
            })
        sparsity_data[level] = pd.DataFrame(grouped_data)


    single_fig = create_plotly_bar_plot(sparsity_data[0.5], "0.5")
    # single_fig.show()
    multi_fig = create_multi_sparsity_plot(sparsity_data)
    # multi_fig.show()

    single_fig.write_image(f"{result_path}/figure_single.png", height=800, width=600)
    multi_fig.write_image(f"{result_path}/figure_multi.png")


def figure11(result_path):
    def plot_runtime(data_frame):
        df = data_frame.copy()
        df['config'] = df.apply(lambda row: "Dense" if row['format'].strip() == 'dense' else( 'Sparse + Prop' if row['propagate'] == 1 else 'Sparse'), axis=1)
        df = df.sort_values(by=['config'])
        fig = go.Figure()

        current_x = 0

        
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
                y=config_data['runtime'],
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
                ticktext=[i*100 for i in df['sparsity'].unique()],
                showgrid=True,
                gridwidth=1,
                gridcolor='#f0f0f0',
                linecolor='black',
                linewidth=2,
                tickfont=dict(size=12),
                title_text="% Sparsity",
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='#f0f0f0',
                tickfont=dict(size=12),
                title_text="Runtime (s)",
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=11),
            width=500,
            height=250,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.25,
                xanchor="center",
                x=0.5,
                font=dict(size=12),
                itemsizing='trace',
            ),
            margin=dict(t=0, b=0, l=0, r=40),
        )
        # fig.show()
        pio.write_image(fig, f"{result_path}/figure.png", width=500, height=250)

    def read_csvs(result_path):
        df = pd.concat([pd.read_csv(file) for file in glob.glob(result_path + "/*.csv")])
        df = df.sort_values(by=['sparsity'])
        return df

    row_df = read_csvs(result_path)

    plot_runtime(row_df)
