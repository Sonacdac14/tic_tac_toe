import pickle
import numpy as np
from synthetic_data import generate_synthetic_data
from parameters import parameters

if __name__ == "__main__":
    # Load the trained model from file
    with open(r'C:\Users\jyotsna\PycharmProjects\task_tic_tac_toe\trained_model.pkl', 'rb') as f:
        nn = pickle.load(f)

    # Generate synthetic test data
    X_test, y_test = generate_synthetic_data(200, parameters['layers'][0], parameters['layers'][-1])

    # Predict on the test data
    predictions = nn.predict(X_test)

    # Calculate accuracy
    accuracy = np.mean(predictions == y_test) * 100
    print(f"Accuracy on test data: {accuracy:.2f}%")
