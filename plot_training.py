import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

def plot_training_history(history, save_dir='plots'):
    # Create plots directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot training & validation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(save_dir, 'accuracy.png'))
    plt.close()
    
    # Plot training & validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    plt.close()

def main():
    # Load the model
    model = load_model('emotion_model.h5')
    
    # Load test data
    from download_model import load_data
    _, X_test, _, y_test = load_data("FER-2013")
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    
    # Print final metrics
    print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")
    print(f"Final Test Loss: {test_loss:.4f}")
    
    # Generate confusion matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = y_test
    
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('plots/confusion_matrix.png')
    plt.close()

if __name__ == '__main__':
    main() 