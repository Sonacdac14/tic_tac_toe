# import configparser
# import numpy as np
# from neural_network import NeuralNetwork
# from synthetic_data import generate_synthetic_data
#
# def read_parameters(file_path):
#     config = configparser.ConfigParser()
#     config.read(file_path)
#     parameters = {
#         'layers': list(map(int, config['NETWORK']['layers'].split(','))),
#         'activation': config['NETWORK']['activation'],
#         'learning_rate': float(config['TRAINING']['learning_rate']),
#         'epochs': int(config['TRAINING']['epochs'])
#     }
#     return parameters
#
# if __name__ == "__main__":
#     params = read_parameters('parameters.txt')
#     nn = NeuralNetwork(params['layers'], params['activation'])
#
#     X_train, y_train = generate_synthetic_data(1000, params['layers'][0], params['layers'][-1])
#     X_test, y_test = generate_synthetic_data(200, params['layers'][0], params['layers'][-1])
#
#     nn.train(X_train, y_train, params['learning_rate'], params['epochs'])
#
#     predictions = nn.predict(X_test)
#     accuracy = np.mean((predictions > 0.5) == y_test)
#     print(f"Accuracy: {accuracy * 100:.2f}%")

#############

# from neural_network import NeuralNetwork
# from synthetic_data import generate_synthetic_data
# import pickle
# from parameters import parameters
#
# if __name__ == "__main__":
#     nn = NeuralNetwork(parameters['layers'], 'relu')
#
#     # Generate synthetic training data
#     X_train, y_train = generate_synthetic_data(1000, parameters['layers'][0], parameters['layers'][-1])
#
#     # Train the neural network
#     nn.train(X_train, y_train, parameters['learning_rate'], parameters['epochs'])
#
#     # Save the trained model to a file
#     with open('trained_model.pkl', 'wb') as f:
#         pickle.dump(nn, f)
#
#     print("Training completed and model saved.")




import numpy as np

from neural_network import NeuralNetwork
from synthetic_data import generate_synthetic_data
import pickle
from parameters import parameters

if __name__ == "__main__":
    params = parameters
    nn = NeuralNetwork(params['layers'], params['activation'], params['dropout_rate'])

    X_train, y_train = generate_synthetic_data(1000, params['layers'][0], params['layers'][-1])
    X_test, y_test = generate_synthetic_data(200, params['layers'][0], params['layers'][-1])

    nn.train(X_train, y_train, params['learning_rate'], params['epochs'], params['batch_size'])

    # Save the trained model to a file
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(nn, f)

    # Evaluate on the test set
    predictions = nn.predict(X_test)
    accuracy = np.mean(predictions == y_test) * 100
    print(f"Accuracy: {accuracy:.2f}%")