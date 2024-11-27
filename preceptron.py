print("*** PERCPTRON MODEL ***\n\n")
print("CS22104")


def initialization(num_inputs, initial_weights=None, bias=0):  # step 1: for initialize weights and bias
    if initial_weights is None:
        weights = [0] * num_inputs
    else:
        weights = initial_weights
    return weights, bias


def threshold_value(net_value):  # step5 threshold = 1
    return 1 if net_value >= 1 else 0


def calculate_net(inputs, weights, bias):  # (b + summation of xi)
    net_value = bias + sum([inputs[i] * weights[i] for i in range(len(inputs))])
    return net_value


def train_perceptron(x, target, alpha, initial_weights=None, max_epochs=30):
    num_inputs = len(x[0])
    weights, bias = initialization(num_inputs, initial_weights)

    for epoch in range(max_epochs):  # step 2-6 for training until stopping condition (max_epochs>=30 or error ==0)
        print(f"\nEpoch {epoch + 1}/{max_epochs}:")
        errors = 0

        for i in range(len(x)):
            a = x[i]
            target_output = target[i]
            net_value = calculate_net(a, weights, bias)
            observed_output = threshold_value(net_value)

            error = target_output - observed_output  # E=T-Y
            print(f"Input: {a}, Target: {target_output}, Observed: {observed_output}, Error: {error}")

            if error != 0:  # step7 update weight and bias if error !=0
                errors += 1
                for j in range(num_inputs):
                    weights[j] += alpha * error * a[j]
                bias += alpha * error
                print(f"Updated Weights: {[round(w, 2) for w in weights]}, Bias: {round(bias, 2)}")
            else:
                print(f"No update needed for input {a}")

        if errors == 0: # step8
            print("\nTraining complete: No errors found!")
            break

    if errors > 0:  # Check if errors were present after finishing all cycles
        print("\nOOPS! Cannot resolve errors after maximum epochs.\nNeed more epochs:((")

    return weights, bias


def test_perceptron(inputs, weights, bias):  # Test the perceptron after complete training
    net_value = calculate_net(inputs, weights, bias)
    return threshold_value(net_value)


# Inputs
# Main program
print("Enter number of input features:")
num_inputs = int(input())

print("Enter the input table (in a vertical way):")
features = []
for _ in range(2 ** num_inputs):
    inputs = list(map(int, input().split()))
    features.append(inputs)

print("Enter the target output (e.g. 0 0 0 1):")
target = list(map(int, input().split()))

print("Enter initial weights (space-separated):")
initial_weights = list(map(float, input().split()))

print("Enter learning rate:")
learning_rate = float(input())

print("\nTraining the perceptron...")
weights, bias = train_perceptron(features, target, learning_rate, initial_weights)

while True:  # Test the perceptron with new inputs
    print("Enter test inputs (or type 'exit' to quit):")
    test_input = input()
    if test_input.lower() == 'exit':
        break
    test_input = list(map(int, test_input.split()))
    result = test_perceptron(test_input, weights, bias)
    print(f"Observed Output: {result}")
