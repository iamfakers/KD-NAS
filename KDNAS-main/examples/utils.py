import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_and_display_data(data_dir: Path | str):
    data_dir = Path(data_dir)

    with open(data_dir / "block_library.json", "r") as f:
        block_library = json.load(f)

    with open(data_dir / "measurement_info.json", "r") as f:
        measurement_info = json.load(f)

    with open(data_dir / "parent_block_stats.json", "r") as f:
        parent_block_stats = json.load(f)

    sample_layer = list(block_library.keys())[0]
    sample_configs = list(block_library[sample_layer].keys())

    first_stats = block_library[sample_layer][sample_configs[0]]["metrics"]
    target_metric = "kl_div" if "kl_div" in first_stats else "loss"

    print(f"{data_dir.name} Block Library:")
    print("=" * 40)
    print(f"Model: {data_dir.name}")
    print(f"Number of layers: {len(block_library)}")
    print(f"Block variants per layer: {len(block_library[sample_layer])}")
    print(f"Detected Target Metric: {target_metric}")

    print(f"\nMeasurement Configuration:")
    print(f"- Batch size: {measurement_info['batch_size']}")
    print(f"- Sequence length: {measurement_info['sequence_length']:,}")
    print(f"- Total tokens: {measurement_info['batch_size'] * measurement_info['sequence_length']:,}")
    print(f"- GPU: {measurement_info['gpu']}")
    print(f"- Data type: {measurement_info['dtype']}")

    print(f"\nParent Model (Original) Stats:")
    print(f"- Memory: {parent_block_stats['stats']['memory_mib']:.1f} MiB per layer")
    print(f"- Runtime: {parent_block_stats['stats']['runtime_ms']:.1f} ms per layer")
    print(f"- Attention runtime: {parent_block_stats['stats']['attention_runtime_ms']:.1f} ms")
    print(f"- FFN runtime: {parent_block_stats['stats']['ffn_runtime_ms']:.1f} ms")
    print(f"- KV heads: {parent_block_stats['stats']['num_kv_heads']}")
    print(f"- FFN size: {parent_block_stats['stats']['ffn_intermediate_size']:,}")

    print(f"\nSample block configurations:")
    for i in [0, min(11, len(sample_configs) - 1)]:
        config = sample_configs[i]
        stats = block_library[sample_layer][config]
        print(f"\n{i + 1}. {config}")
        print(f"   - {target_metric}: {stats['metrics'][target_metric]:.4f}")
        print(f"   - Memory: {stats['stats']['memory_mib']:.1f} MiB")
        print(f"   - Runtime: {stats['stats']['runtime_ms']:.1f} ms")

        speedup = (parent_block_stats['stats']['runtime_ms'] - stats['stats']['runtime_ms']) / \
                  parent_block_stats['stats']['runtime_ms'] * 100
        if speedup > 0:
            print(f"   -> {speedup:.1f}% speedup")

    return block_library, measurement_info, parent_block_stats


def calculate_throughput(runtime_ms, batch_size, seq_length):
    total_tokens = batch_size * seq_length
    return total_tokens / (runtime_ms / 1000) if runtime_ms > 0 else float('inf')


def display_solution_stats(total_value, total_costs,
                           block_library, parent_block_stats, measurement_info):
    throughput = calculate_throughput(total_costs['stats.runtime_ms'],
                                      measurement_info['batch_size'],
                                      measurement_info['sequence_length'])

    print(f"Solution Statistics:")
    print(f"  - Total Objective Value: {total_value:.6f}")
    print(f"  - Memory usage: {total_costs['stats']['memory_mib']:.1f} MiB")
    print(f"  - Runtime: {total_costs['stats']['runtime_ms']:.1f} ms")
    print(f"  - Theoretical Throughput: {throughput:,.0f} tokens/second")

    parent_memory = parent_block_stats['stats']['memory_mib'] * len(block_library)
    parent_runtime = parent_block_stats['stats']['runtime_ms'] * len(block_library)
    parent_throughput = calculate_throughput(parent_runtime,
                                             measurement_info['batch_size'],
                                             measurement_info['sequence_length'])

    print(f"\nCompared to Parent Model:")
    print(f"  - Memory: {total_costs['stats']['memory_mib'] / parent_memory * 100:.1f}% of parent")
    print(f"  - Runtime: {total_costs['stats.runtime_ms'] / parent_runtime * 100:.1f}% of parent")
    print(f"  - Speedup: at least {parent_runtime / total_costs['stats.runtime_ms']:.2f}x")
    print(f"  - Throughput increase: at least {(throughput - parent_throughput) / parent_throughput * 100:+.1f}%")


def create_solution_dataframe(solution, block_library, parent_block_stats):
    chosen_block_variants = solution["chosen_block_variants"]

    memory_per_layer = []
    runtime_per_layer = []
    metric_per_layer = []
    layer_indices = []

    memory_percent = []
    runtime_percent = []

    sample_layer = list(block_library.keys())[0]
    sample_config = list(block_library[sample_layer].keys())[0]
    first_stats = block_library[sample_layer][sample_config]["metrics"]
    target_metric = "kl_div" if "kl_div" in first_stats else "loss"

    for layer_id, block_config in chosen_block_variants.items():
        layer_idx = int(layer_id.split("_")[1])
        layer_indices.append(layer_idx)

        block_stats = block_library[layer_id][block_config]
        memory_per_layer.append(block_stats["stats"]["memory_mib"])
        runtime_per_layer.append(block_stats["stats"]["runtime_ms"])
        metric_per_layer.append(block_stats["metrics"][target_metric])

        memory_percent.append(block_stats["stats"]["memory_mib"] / parent_block_stats["stats"]["memory_mib"] * 100)
        runtime_percent.append(block_stats["stats"]["runtime_ms"] / parent_block_stats["stats"]["runtime_ms"] * 100)

    df = pd.DataFrame({
        'layer_idx': layer_indices,
        'memory_mib': memory_per_layer,
        'runtime_ms': runtime_per_layer,
        target_metric: metric_per_layer,
        'memory_percent': memory_percent,
        'runtime_percent': runtime_percent
    })

    return df.sort_values('layer_idx')


def visualize_runtime_reduction(solution, block_library, parent_block_stats, title="Runtime Reduction Across Layers",
                                static: bool = False):
    df = create_solution_dataframe(solution, block_library, parent_block_stats)

    df['runtime_saved'] = parent_block_stats["stats"]["runtime_ms"] - df['runtime_ms']
    df['memory_saved'] = parent_block_stats["stats"]["memory_mib"] - df['memory_mib']

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df['layer_idx'],
        y=df['runtime_ms'],
        name='Optimized Runtime',
        marker_color='steelblue',
        text=[f'{val:.0f}' for val in df['runtime_percent']],
        texttemplate='%{text}%',
        textposition='inside',
        textfont=dict(size=7),
        hovertemplate='Layer %{x}<br>Runtime: %{y:.1f} ms<br>%{text}% of parent<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        x=df['layer_idx'],
        y=df['runtime_saved'],
        name='Runtime Saved',
        marker_color='lightgreen',
        hovertemplate='Layer %{x}<br>Saved: %{y:.1f} ms<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Layer Index",
        yaxis_title="Runtime (ms)",
        barmode='stack',
        bargap=0,
        showlegend=True,
        height=400,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(
            showgrid=False,
            zeroline=True,
            linecolor='black',
            linewidth=1,
            dtick=10
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            linecolor='black',
            linewidth=1
        )
    )

    if static:
        try:
            png_bytes = fig.to_image(format="png", scale=2)
            from IPython.display import Image, display
            display(Image(data=png_bytes))
        except Exception:
            print("Static export requires 'kaleido' (pip install kaleido).")
    else:
        fig.show()

    total_runtime = df['runtime_ms'].sum()
    parent_total_runtime = parent_block_stats["stats"]["runtime_ms"] * len(df)
    speedup = parent_total_runtime / total_runtime

    print(f"\nRuntime Summary:")
    print(f"- Parent model runtime: {parent_total_runtime:.1f} ms")
    print(f"- Optimized runtime: {total_runtime:.1f} ms")
    print(f"- Total speedup: {speedup:.2f}x")
    print(f"- Average runtime reduction per layer: {(1 - total_runtime / parent_total_runtime) * 100:.1f}%")
    metric_col = "kl_div" if "kl_div" in df.columns else "loss"
    print(f"- Sum {metric_col}: {df[metric_col].sum():.4f}")

    return df


def compare_solutions(solutions, block_library, parent_block_stats, measurement_info, static: bool = True):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Runtime per Layer", "Memory per Layer",
                        "Objective Metric by Solution", "Throughput Comparison"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "box"}, {"type": "bar"}]],
        horizontal_spacing=0.15,
        vertical_spacing=0.15
    )

    colors = ['blue', 'red', 'green', 'purple', 'orange']

    throughputs = []
    names = []

    metric_col = "kl_div" if "kl_div" in create_solution_dataframe(solutions[0], block_library,
                                                                   parent_block_stats).columns else "loss"

    for i, solution in enumerate(solutions):
        df = create_solution_dataframe(solution, block_library, parent_block_stats)
        color = colors[i % len(colors)]
        name = solution.get('name', f'Solution {i + 1}')
        names.append(name)

        fig.add_trace(
            go.Scatter(x=df['layer_idx'], y=df['runtime_percent'],
                       mode='lines', name=name,
                       line=dict(color=color, width=2),
                       showlegend=True),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=df['layer_idx'], y=df['memory_percent'],
                       mode='lines', name=name,
                       line=dict(color=color, width=2),
                       showlegend=False),
            row=1, col=2
        )

        fig.add_trace(
            go.Box(y=df[metric_col], name=name,
                   marker_color=color,
                   showlegend=False),
            row=2, col=1
        )

        runtime_ms = solution['total_costs']['stats.runtime_ms']
        throughput = calculate_throughput(runtime_ms,
                                          measurement_info['batch_size'],
                                          measurement_info['sequence_length'])
        throughputs.append(throughput)

    fig.add_trace(
        go.Bar(x=names, y=throughputs,
               text=[f'{t:,.0f}' for t in throughputs],
               textposition='auto',
               marker_color=colors[:len(names)],
               showlegend=False),
        row=2, col=2
    )

    fig.update_xaxes(title_text="Layer Index", row=1, col=1)
    fig.update_xaxes(title_text="Layer Index", row=1, col=2)
    fig.update_xaxes(title_text="Solution", row=2, col=1)
    fig.update_xaxes(title_text="Solution", row=2, col=2)

    fig.update_yaxes(title_text="Runtime (% of parent)", row=1, col=1)
    fig.update_yaxes(title_text="Memory (% of parent)", row=1, col=2)
    fig.update_yaxes(title_text=f"{metric_col.capitalize()} per Layer", row=2, col=1)
    fig.update_yaxes(title_text="Tokens/Second", row=2, col=2)

    fig.update_layout(height=800, showlegend=True,
                      title_text="Puzzle Solutions Comparison")

    if static:
        try:
            png_bytes = fig.to_image(format="png", scale=2)
            from IPython.display import Image, display
            display(Image(data=png_bytes))
        except Exception:
            print("Static export requires 'kaleido' (pip install kaleido).")
        return None
    else:
        fig.show()
        return fig


def comprehensive_solution_comparison(solutions, block_library, parent_block_stats, measurement_info,
                                      static: bool = False):
    print("Comprehensive Solution Comparison")
    print("=" * 80)

    comparison_data = []
    for sol in solutions:
        runtime = sol['total_costs']['stats.runtime_ms']
        memory = sol['total_costs']['stats.memory_mib']
        objective_val = sol['total_value']
        throughput = calculate_throughput(runtime, measurement_info['batch_size'],
                                          measurement_info['sequence_length'])

        parent_runtime_total = parent_block_stats['stats']['runtime_ms'] * len(block_library)
        speedup = parent_runtime_total / runtime

        comparison_data.append({
            'Configuration': sol['name'],
            'Memory (GB)': memory / 1024,
            'Runtime (s)': runtime / 1000,
            'Throughput (tok/s)': throughput,
            'Speedup': speedup,
            'Objective Value': objective_val
        })

    comparison_df = pd.DataFrame(comparison_data)

    comparison_df = comparison_df.round({
        'Memory (GB)': 1,
        'Runtime (s)': 2,
        'Throughput (tok/s)': 0,
        'Speedup': 2,
        'Objective Value': 4
    })

    print(comparison_df.to_string(index=False))

    print("\nVisual Comparison:")
    _ = compare_solutions(solutions, block_library, parent_block_stats, measurement_info, static=static)