import matplotlib.pyplot as plt
import os


def save_plot(data, metric_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    experiments = [result["experiment"] for result in data]
    metric_values = [result[metric_name] for result in data]

    plt.figure()
    plt.plot(experiments, metric_values, label=metric_name, marker="o")
    plt.xlabel("Experiment")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} by Experiment")
    plt.legend()

    output_path = os.path.join(output_dir, f"{metric_name}_plot.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Graph {metric_name} saved at {output_path}")
