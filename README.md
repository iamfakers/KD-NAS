# KD-NAS: Hardware-Aware Knowledge Distillation for LLMs on In-vehicle Devices

This repository contains the implementation and demonstration of the **KD-NAS (Knowledge Distillation-based Neural Architecture Search)** framework, proposed in our paper: *"Toward Efficient Deployment of LLMs on Resource-constrained In-vehicle Devices: A Hardware-Aware Knowledge Distillation Approach"*.

The goal of this framework is to automatically search for and customize optimal student LLM architectures that strictly satisfy the memory and latency constraints of heterogeneous in-vehicle hardware (e.g., NVIDIA Jetson, RTX series).

## Key Features of KD-NAS

1. **Hardware-Aware Architecture Search**: Uses Mixed-Integer Programming (MIP) to select the optimal configuration (number of layers, hidden dimensions, attention heads, FFN sizes) from a predefined search space, ensuring the model fits within the vehicle's memory (\(M_k^{\max}\)) and latency (\(T_k^{\max}\)) budgets.
2. **Multi-Level Distillation Ready**: The searched architecture is designed to seamlessly integrate with our teacher-feedback-based RL fine-tuning method, transferring knowledge from the teacher LLM's embedding, intermediate, and prediction layers.
3. **Pareto-Optimal Selection**: Balances model accuracy (KL Divergence/Loss) with hardware efficiency. The solver can generate multiple diverse solutions (Pareto front) for different deployment preferences.

## Repository Structure

```text
.
├── examples/
│   ├── data/
│   │   └── Llama-2-7B/          # Pre-computed data for Llama-2-7B
│   │       ├── block_library.json      # Candidate architectures and their hardware stats
│   │       ├── measurement_info.json   # Target hardware constraints (e.g., Batch size, Seq length)
│   │       └── parent_block_stats.json # Baseline stats of the teacher model
│   ├── standalone_mip_nas_example.py   # Main entry point for the MIP-based KD-NAS demo
│   └── utils.py                        # Visualization and data loading utilities
├── KDNAS/                       # Core algorithm package
│   └── mip_nas.py               # MIP solver implementation for architecture search
├── requirements.txt
└── LICENSE
## Quick Start

### 1. Prerequisites & Installation

We recommend using a virtual environment. Install the required dependencies (including `mip` for mathematical optimization and `plotly` for visualization):

```bash
pip install -r requirements.txt
```

*(Note: To export visualization charts to static images, you may also need `pip install kaleido`)*

### 2. Run the KD-NAS Search

**Important:** Because this project uses module-level imports, please run the script from the **root directory** of the project.

Open your terminal in the root directory and execute:

```bash
python -m examples.standalone_mip_nas_example
```

### 3. Expected Output

The script will perform the following steps:

1. **Load Data**: Read the candidate block configurations and the target hardware constraints.
2. **MIP Optimization**: The solver searches for multiple diverse architectures that minimize the knowledge distillation loss while strictly respecting the memory and latency constraints (Equations 9-12 in the paper).
3. **Validation**: Print the search progress, constraint validation, and diversity check between solutions.
4. **Visualization**: Automatically open an interactive Plotly dashboard in your browser, comparing the original model and the optimized student models across:
   - **Runtime per Layer**: Demonstrating the latency reduction.
   - **Memory per Layer**: Showing memory footprint compliance.
   - **Objective Metric**: Comparing the expected KL Divergence/Loss.
   - **Throughput (Tokens/Second)**: Reflecting the inference efficiency on the target hardware.

## Customization

You can easily adapt this KD-NAS framework to other teacher models or different vehicle hardware platforms:

1. **Prepare Data**: Generate the search space and hardware profiling data for your target model. Place the three JSON files under `examples/data/Your-Model-Name/`.
2. **Set Constraints**: Modify the `DATA_DIR` and `CONSTRAINTS` variables in `examples/standalone_mip_nas_example.py` to match your target vehicle's memory limit (e.g., 8GB for Jetson Orin NX) and tolerable latency.

## Citation

If you find this code or our KD-NAS framework useful for your research, please cite our paper:

```bibtex
@article{zhang2024toward,
  title={Toward Efficient Deployment of LLMs on Resource-constrained In-vehicle Devices: A Hardware-Aware Knowledge Distillation Approach},
  author={Zhang, Yong and Li, Chunlin and Zhang, Shuai and Wan, Shaohua},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
  year={2024},
  publisher={IEEE}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
