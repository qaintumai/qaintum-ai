from sklearn.metrics import roc_auc_score
import time
import torch
import torch.nn as nn

def train(X_train, y_train, X_val, y_val, model, batch_size, learning_rate, epochs):
    """
    Trains the quantum neural network using the provided training and validation data.
    Parameters:
    - X_train: Training input data (array-like).
    - y_train: Training target labels (array-like).
    - X_val: Validation input data (array-like).
    - y_val: Validation target labels (array-like).
    - model: PyTorch model to be trained.
    - batch_size: Batch size for training.
    - learning_rate: Learning rate for the optimizer.
    - epochs: Number of training epochs.
    Returns:
    - history: Dictionary containing training and validation metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    # Convert inputs and targets to tensors
    X_train = torch.as_tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.as_tensor(y_train, dtype=torch.float32).to(device)
    X_val = torch.as_tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.as_tensor(y_val, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    # Initialize history storage
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'valAUC': [],
        'max_abs_grad': []
    }

    # Start the timer for overall training
    overall_start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct_train = 0
        total_train = 0
        max_grad = 0
        total_batches = (X_train.size(0) + batch_size - 1) // batch_size

        print(f"\nEpoch {epoch+1}/{epochs}")
        print("Training [", end='', flush=True)

        # Epoch Start Time
        epoch_start_time = time.time()

        # Batch processing
        permutation = torch.randperm(X_train.size(0))
        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            X_batch = X_train[indices]
            y_batch = y_train[indices]

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()

            # Track maximum gradient
            current_max_grad = max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None)
            max_grad = max(max_grad, current_max_grad)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            preds = outputs.argmax(axis=1).float()
            correct_train += (preds == y_batch.argmax(axis=1)).sum().item()
            total_train += y_batch.size(0)

            # Manual progress bar
            progress = (i // batch_size + 1) / total_batches
            bar_length = 50
            filled_length = int(bar_length * progress)
            bar = '>>' * filled_length + '..' * (bar_length - filled_length)
            percent = int(100 * progress)
            print(f"\rTraining [{bar}] {percent}%", end='', flush=True)

        # Calculate metrics
        train_loss = epoch_loss / (i // batch_size + 1)
        train_acc = correct_train / total_train

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val).squeeze()
            val_Eloss = criterion(val_outputs, y_val).item()
            val_acc = ((val_outputs.argmax(axis=1)).float() == y_val.argmax(axis=1)).float().mean().item()
            valEaucscore = roc_auc_score(y_val.argmax(axis=1).cpu().numpy(), val_outputs[:, 1].cpu().numpy())

        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_Eloss)
        history['val_acc'].append(val_acc)
        history['valAUC'].append(valEaucscore)
        history['max_abs_grad'].append(max_grad)

        print()

        # Print metrics
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_Eloss:.4f}, Val Acc: {val_acc:.4f}, Max Abs Grad: {max_grad:.6f}, ValAUC: {valEaucscore:.6f}")
        epoch_end_time = time.time()
        print(f"Total time for epoch {epoch + 1}: {epoch_end_time - epoch_start_time:.6f} seconds")
        print("------------------------------------------------------------------------------------")

    # Overall End Time
    overall_end_time = time.time()
    print(f"Total training time: {overall_end_time - overall_start_time:.6f} seconds")

    return history