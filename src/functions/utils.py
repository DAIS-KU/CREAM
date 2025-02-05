import matplotlib.pyplot as plt


def show_success_recall(success_recall_values):
    success = [value[0] for value in success_recall_values]
    recall = [value[1] for value in success_recall_values]
    x = range(len(success))

    # Success plot
    plt.plot(x, success, label="Success", color="red")
    for i, val in enumerate(success):
        plt.text(
            i, val, f"{val:.2f}", ha="center", va="bottom", fontsize=8, color="red"
        )

    # Recall plot
    plt.plot(x, recall, label="Recall", color="green")
    for i, val in enumerate(recall):
        plt.text(
            i, val, f"{val:.2f}", ha="center", va="bottom", fontsize=8, color="green"
        )

    # Graph details
    plt.xlabel("Batch")
    plt.ylabel("Values")
    plt.title("Success and Recall Over Batches")
    plt.legend()
    plt.show()


def show_loss(loss_values):
    x = range(len(loss_values))

    # Loss plot
    plt.plot(x, loss_values, label="All Training Loss", color="blue")
    for i, val in enumerate(loss_values):
        plt.text(
            i, val, f"{val:.2f}", ha="center", va="bottom", fontsize=8, color="blue"
        )

    # Graph details
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Training Loss for all sessions")
    plt.legend()
    plt.show()


def draw_elbow(k_values, sse):
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, sse, marker="o", linestyle="-", color="b", label="SSE")
    plt.xticks(k_values)
    plt.title("SSE vs Number of Clusters (k)", fontsize=14)
    plt.xlabel("Number of Clusters (k)", fontsize=12)
    plt.ylabel("SSE (Sum of Squared Errors)", fontsize=12)
    plt.grid(alpha=0.5)
    plt.legend(fontsize=10)
    plt.show()


def write_file(rank_file_path, result):
    with open(rank_file_path, "w") as f:
        for key, values in result.items():
            line = f"{key} " + " ".join(map(str, values)) + "\n"
            f.write(line)
