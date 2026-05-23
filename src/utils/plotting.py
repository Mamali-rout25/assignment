import os
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(style="whitegrid", font_scale=1.1)


def plot_accuracy_loss(history: dict, title: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    epochs = list(range(1, len(history.get("accuracy", [])) + 1))
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.get("accuracy", []), marker="o")
    plt.title(f"{title} Accuracy")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.get("loss", []), marker="o", color="tab:red")
    plt.title(f"{title} Loss")
    plt.xlabel("Round")
    plt.ylabel("Loss")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_communication(results: list, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    labels = [r["variant"] for r in results]
    uploaded = [r["uploaded_bytes"] for r in results]
    downloaded = [r["downloaded_bytes"] for r in results]

    x = range(len(labels))
    width = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar([i - width / 2 for i in x], uploaded, width=width, label="Uploaded bytes")
    plt.bar([i + width / 2 for i in x], downloaded, width=width, label="Downloaded bytes")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.title("Federated Communication Overhead")
    plt.ylabel("Bytes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
