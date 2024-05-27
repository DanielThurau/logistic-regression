import numpy as np
import json
import logging as log
import copy
import matplotlib.pyplot as plt
import h5py


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def set_logging_level(level):
    log.basicConfig(level=level)
    log.info("Logging level set to " + str(level))


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def normalize_rows(x):
    x_norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    x = x / x_norm
    return x


def load_data():
    train_dataset = h5py.File('data/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('data//test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def sample_data(index, train_set_x_orig, train_set_y, classes):
    plt.imshow(train_set_x_orig[index])
    plt.show()
    print("y = " + str(train_set_y[0, index]) + ". It's a " + classes[train_set_y[0, index]].decode(
        "utf-8") + " picture.")


def flatten_input_images(train_set_x_orig, test_set_x_orig):
    assert (train_set_x_orig.shape == (209, 64, 64, 3))

    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]

    train_set_x = train_set_x_orig.reshape(m_train, -1).T
    test_set_x = test_set_x_orig.reshape(m_test, -1).T

    return train_set_x, test_set_x


def normalize_image_data(set_x):
    set_x = set_x / 255
    return set_x


def initialize_model_parameters(feature_dimension):
    w = np.zeros((feature_dimension, 1))
    b = 0.0
    return w, b


def propagate(w, b, X, Y):
    m_train = X.shape[1]

    Z = np.dot(w.T, X) + b
    assert (Z.shape == (1, m_train))

    A = sigmoid(Z)
    assert (A.shape == (1, m_train))

    J = np.sum(- (Y * np.log(A) + (1 - Y) * np.log(1 - A))) / m_train

    dZ = A - Y
    assert (dZ.shape == (1, m_train))

    dW = np.dot(X, np.transpose(dZ)) / m_train
    db = np.sum(dZ) / m_train

    return {"dw": dW, "db": db}, J


def optimize(w, b, X, Y, num_iterations=1000, learning_rate=0.009):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)

    costs = []
    assert num_iterations > 0
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - (learning_rate * dw)
        b = b - (learning_rate * db)

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
            log.debug("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):

        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1.0
        else:
            Y_prediction[0, i] = 0.0

    return Y_prediction


def model(X_train, Y_train, X_test, learning_rate=0.5, num_iterations=2000):
    m_train = X_train.shape[1]
    assert (m_train == 209)
    m_test = X_test.shape[1]
    assert (m_test == 50)
    feature_count = X_train.shape[0]
    assert (feature_count == 12288)

    w, b = initialize_model_parameters(feature_count)
    log.debug("Feature Count: " + str(w.shape[0]))

    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate)

    w = params["w"]
    b = params["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Return the model data 
    d = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations
    }

    return d


if __name__ == "__main__":
    set_logging_level(log.DEBUG)  # can be switched to use `log.INFO` or `log.DEBUG`
    log.getLogger('matplotlib.font_manager').disabled = True

    # Download and normalize the data for the rest of the modeling pipeline
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_data()
    train_set_x, test_set_x = flatten_input_images(train_set_x_orig, test_set_x_orig)
    train_set_x = normalize_image_data(train_set_x)
    test_set_x = normalize_image_data(test_set_x)

    # Define tunable parameters for model training
    learning_rates = [0.05]
    num_iterations = [2_000]
    models = {}
    accuracy_score = {}

    for learning_rate in learning_rates:
        for num_iteration in num_iterations:
            key = str(learning_rate) + str(num_iteration)
            print("Training a model with learning rate: " + str(learning_rate) + " and num_iterations: " + str(num_iteration))
            models[key] = model(train_set_x, train_set_y, test_set_x, num_iterations=num_iteration,
                                learning_rate=learning_rate)
            train_accuracy = 100 - np.mean(np.abs(models[key]["Y_prediction_train"] - train_set_y)) * 100
            test_accuracy = 100 - np.mean(np.abs(models[key]["Y_prediction_test"] - test_set_y)) * 100
            log.info("train accuracy: {}%".format(train_accuracy))
            log.info("test accuracy: {}%".format(test_accuracy))
            accuracy_score[key] = (train_accuracy, test_accuracy)

            print('\n' + "-------------------------------------------------------" + '\n')

    for key in models.keys():
        plt.plot(np.squeeze(models[key]["costs"]),
                 label="learning_rate=" + str(models[key]["learning_rate"]) + " num_iterations=" + str(
                     models[key]["num_iterations"]))

    plt.ylabel('cost')
    plt.xlabel('iterations (hundreds)')

    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()

    # Loop through the accuracy_score dictionary to plot each point
    for key, (train_accuracy, test_accuracy) in accuracy_score.items():
        # Plot each point with train_accuracy as x and test_accuracy as y
        plt.scatter(train_accuracy, test_accuracy,
                    label="learning_rate=" + str(models[key]["learning_rate"]) + " num_iterations=" + str(
                        models[key]["num_iterations"]))

    plt.xlabel('Train Accuracy (%)')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Train vs Test Accuracy for Different Learning Rates and Iterations')

    # Adjust the legend to make it more readable
    plt.legend(loc='best', bbox_to_anchor=(1, 1), title="Configurations")

    plt.grid(True)  # Adds a grid for better readability
    plt.tight_layout()  # Adjust layout to make room for the legend if necessary
    plt.show()

    best = ('', 0)
    for key, (train_accuracy, test_accuracy) in accuracy_score.items():
        score = (train_accuracy * 0.4 + test_accuracy * 0.6) / 2.0
        if score > best[1]:
            best = (key, score)
    filename = "models/" + best[0] + '_logistic_regression_model.json'
    log.debug("The best performing model is " + str(best) + " . Writing model data to " + filename)
    with open(filename, 'w') as f:
        # json.dump(models[best[0]], f)
        json_dump = json.dumps(models[best[0]],
                               cls=NumpyEncoder)
        f.writelines(json_dump)
