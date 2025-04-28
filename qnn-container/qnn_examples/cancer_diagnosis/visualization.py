import matplotlib.pyplot as plt

def plotResults(history):
    """
    Plots training and validation metrics (loss, accuracy, and maximum absolute gradients) over epochs.
    Parameters:
    - history (dict): A dictionary containing training and validation metrics.
                      Expected keys: 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'max_abs_grad'.
    Returns:
    - None
    """

    #********************************************  Loss Plot ********************************************
    ee1 = len(history['train_loss'])
    plt.plot(range(0, ee1), history['train_loss'], 'g', label='Training Loss')
    plt.plot(range(0, ee1), history['val_loss'], 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    #********************************************  Accuracy Plot ********************************************
    ee2 = len(history['train_acc'])
    plt.plot(range(0, ee2), history['train_acc'], 'g', label='Training Accuracy')
    plt.plot(range(0, ee2), history['val_acc'], 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    #********************************************  Maximum Absolute Gradient Plot ********************************************
    ee3 = len(history['max_abs_grad'])
    plt.plot(range(0, ee3), history['max_abs_grad'], 'b')
    plt.title('Maximum Absolute Gradient Values')
    plt.xlabel('Epochs')
    plt.ylabel('Max Abs Grad')
    plt.legend()
    plt.show()