import matplotlib.pyplot as plt

def plot_metrics(history):
    # Plot training and validation accuracy and loss
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].plot(history.history["accuracy"], label="Train Accuracy")
    axs[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axs[0].set_title("Accuracy")
    axs[0].legend()

    axs[1].plot(history.history["loss"], label="Train Loss")
    axs[1].plot(history.history["val_loss"], label="Val Loss")
    axs[1].set_title("Loss")
    axs[1].legend()

    plt.show()
