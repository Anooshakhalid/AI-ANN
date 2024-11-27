import random
import math

print("*** MULTILAYER PERCEPTRON MODEL ***\n\n")
print("CS22104")

def initialize_weights():
    w1 = [random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)]
    w2 = [random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)]
    v = [random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)]
    w0 = [0, 0]
    v0 = 0
    return w1, w2, v, w0, v0

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(y):
    return y * (1 - y)

def backpropagation():   # Main function
    alpha = float(input("Enter the learning rate (alpha): "))
    print("Enter the input for 2-input table in a vertical way:")
    x1 = []
    x2 = []
    for _ in range(4):
        inputs = input().split()
        x1.append(int(inputs[0]))
        x2.append(int(inputs[1]))

    print("Enter the target output for each input (space-separated):")
    target = list(map(int, input().split()))

    w1, w2, v, w0, v0 = initialize_weights()

    z = [0, 0]  # hidden layer output
    gradient = [0, 0]  # hidden layer gradients
    count = 0

    print('Training Started!!')

    while count < 60000:   # Training loop
        output = []
        for i in range(len(x1)):
            a1 = x1[i]
            a2 = x2[i]
            t = target[i]

            for j in range(len(z)):    # hidden layer outputs
                Zin = w0[j] + a1 * w1[j] + a2 * w2[j]
                z[j] = sigmoid(Zin)

            Yin = v0 + z[0] * v[0] + z[1] * v[1]
            y = sigmoid(Yin)  # final output
            output.append(y)

            error = (t - y) * sigmoid_derivative(y)   # error = (t-y)y(1-y)

            v0 = round(v0 + alpha * error, 3)   # Update weights from hidden to output layer
            for j in range(len(z)):
                gradient[j] = round(z[j] * (1 - z[j]) * error * v[j], 3)
                v[j] = round(v[j] + alpha * error * z[j], 3)

            for j in range(len(z)):    # Update weights from input to hidden layer
                w1[j] = round(w1[j] + alpha * gradient[j] * a1, 3)
                w2[j] = round(w2[j] + alpha * gradient[j] * a2, 3)
                w0[j] = round(w0[j] + alpha * gradient[j], 3)

        count += 1

        binary_output = [1 if o >= 0.5 else 0 for o in output]
        if binary_output == target:   # Stop training if the n/w produces the correct o/p
            print(f"Yess!! Training completed in {count} cycles.")
            break

    print("Final weights and biases after training:")
    print(f"w1: {w1}, w2: {w2}, v: {v}, w0: {w0}, v0: {v0}")

    print("\nLets try the model with new inputs .") # Test input option
    while True:
        test_input = input("Enter test inputs as 'x1 x2' or type 'exit' to quit: ")
        if test_input.lower() == 'exit':
            break
        try:
            a1, a2 = map(int, test_input.split())
            if a1 not in [0, 1] or a2 not in [0, 1]:
                print("Error: Test inputs must be 0 or 1.")
                continue

            # Calculate hidden layer output
            for j in range(len(z)):
                Zin = w0[j] + a1 * w1[j] + a2 * w2[j]
                z[j] = sigmoid(Zin)

            # Calculate final output
            Yin = v0 + z[0] * v[0] + z[1] * v[1]
            y = sigmoid(Yin)

            # Convert final output to binary
            predicted_output = 1 if y >= 0.5 else 0
            print(f"Predicted output for inputs ({a1}, {a2}): {predicted_output}")
        except ValueError:
            print("Invalid input. Please enter two binary numbers (0 or 1) or 'exit'.")

backpropagation()
