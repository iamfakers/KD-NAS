"""
Usage:
python -m examples.standalone_mip_nas_example
"""

from pathlib import Path

from KDNAS.mip_nas import multi_solution_mip_nas
from examples.utils import load_and_display_data, comprehensive_solution_comparison

DATA_DIR = Path("examples/data/Llama-2-7B")

NUM_SOLUTIONS = 5
MINIMAL_DIVERSITY = 10

CONSTRAINTS = {
    "stats.memory_mib": 24000.0,
    "stats.runtime_ms": 350.0,
}


def mip_nas_example():
    block_library, measurement_info, parent_block_stats = load_and_display_data(DATA_DIR)

    solutions = multi_solution_mip_nas(
        block_library,
        objective="metrics.loss",
        constraints=CONSTRAINTS,
        bigger_is_better=False,
        num_solutions=NUM_SOLUTIONS,
        minimal_diversity=MINIMAL_DIVERSITY,
    )

    print()
    print(f"{CONSTRAINTS=}")

    print()
    print("Printing solutions.")

    for i_solution, solution in enumerate(solutions):
        print()
        print(f"solution {i_solution}:")
        for var_name, var_value in solution.items():
            print(f"{var_name}={var_value}")

    print(
        f"\nChecking differences between solutions, should be at least {MINIMAL_DIVERSITY}:"
    )
    for a in range(len(solutions)):
        for b in range(a + 1, len(solutions)):
            num_differences = 0
            for layer_id in block_library.keys():
                num_differences += (
                    solutions[a]["chosen_block_variants"][layer_id]
                    != solutions[b]["chosen_block_variants"][layer_id]
                )
            print(a, "<>", b, "=", num_differences)

    for i, sol in enumerate(solutions):
        sol["name"] = f"Solution-{i+1}"

    comprehensive_solution_comparison(
        solutions,
        block_library,
        parent_block_stats,
        measurement_info,
        static=False,
    )


if __name__ == "__main__":
    mip_nas_example()