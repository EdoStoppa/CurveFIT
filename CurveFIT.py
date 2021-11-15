import numpy as np
import matplotlib.pyplot as plt

# Function to plot the points and the learned curve
def plot_results(Xs: np.ndarray, Ds: np.ndarray, W: list, epoch: int):
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(color="#989898", zorder=0)

    # Plot points
    plt.scatter(Xs, Ds, zorder=2.9, s=8, label='Data')
    # Plot Learned Curve using training data
    ins_train, outs_train = get_predicted(Xs, W)
    plt.plot(ins_train, outs_train, zorder=3, label='Learned Curve', color='#ff0000')

    plt.legend()
    plt.tight_layout()
    plt.savefig('fitting_' + str(epoch) + '.png')
    plt.clf()
    return

# Plot the curve drawn by the MSE descent
def plot_MSE(MSEs):
    epochs = list(range(len(MSEs)))
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title('MSE Descent')
    plt.grid(color="#989898", zorder=0)

    # Plot curve
    plt.plot(epochs, MSEs, zorder=3)

    plt.tight_layout()
    plt.savefig('MSE_curve.png')
    plt.clf()
    return

# Obtain a list of predicted output from the network (ascending order)
def get_predicted(Xs, W):
    predicted = []
    for x in Xs:
        # Obtain output of the network from a single input
        out, lf = forward(W, x)
        predicted.append(out[-1])

    zipped = list(zip(Xs, np.array(predicted)))
    zipped.sort(key=(lambda k: k[0]))
    ins, outs = zip(*zipped)

    return ins, outs

# Generate the dataset
def generate_data():
    Xs = np.random.rand(300, 1)
    Vs = np.random.uniform(-0.1, 0.1, size=(300, 1))
    Ds = np.sin(20*Xs) + 3*Xs + Vs

    return Xs, Ds

# Initialize the Network's weights
def init_weights():
    # W0 is a fictitious weight layer needed for other parts of the algorithm
    W0 = np.ones((1, 1))
    W1 = np.random.uniform(-1, 1, size=(24, 2))
    W2 = np.random.uniform(-1, 1, size=(1, 25))

    return [W0, W1, W2]

# Define layer 1 activation function
def phi_1(x): return np.tanh(x)
# Define layer 2 activation function
def phi_2(x): return x
# Given the preferred layer, return the corresponding activation function
def phi(x):
    if x == 1: return phi_1
    else:      return phi_2

# Define layer 1 activation function derivative
def phi_prime_1(x): return 1.0 - np.tanh(x) ** 2
# Define layer 2 activation function derivative
def phi_prime_2(x): return 1.0
# Given the preferred layer, return the corresponding activation function derivative
def phi_prime(x):
    if x == 1: return phi_prime_1
    else:      return phi_prime_2

# Calculate the Mean Squared Error (MSE)
def MSE(Xs, Ds, W):
    tot = 0
    for idx in range(len(Xs)):
        outs, lf = forward(W, Xs[idx])
        tot += (Ds[idx] - outs[-1]) ** 2

    return tot / len(Xs)

# Calculate the output of the NN
def forward(W, x):
    outs = [x]
    lfs = [0]

    for i in range(1, 3):
        # Calculate the local field
        lfs.append(np.matmul(W[i], np.concatenate(([1], outs[i-1]))))
        # Activate
        outs.append(phi(i)(lfs[-1]))

    return outs, lfs

# Calculate the gradients through backpropagation
def backward(outs, truth, W, lfs):
    # Compute the two deltas
    de_2 = np.multiply(truth - outs[-1], phi_prime(2)(lfs[2])).reshape((1, 1))
    de_1 = np.multiply(np.delete(W[2], 0).T * de_2, phi_prime(1)(lfs[1])).reshape(24, 1)
    deltas = [0, de_1, de_2]

    # Compute the two gradients
    grad_1 = np.matmul(deltas[1], np.concatenate(([1], outs[0])).reshape((1, 2)))
    grad_2 = np.matmul(deltas[2], np.concatenate(([1], outs[1])).reshape((1, 25)))
    grads = [1, grad_1, grad_2]

    return grads

# Update the NN weights using the gradients from backpropagation
def update(W, grads, eta):
    W1 = W[1] + eta * grads[1]
    W2 = W[2] + eta * grads[2]

    return [W[0], W1, W2]

# Train the Network
def train(Xs, Ds, eta, epsilon):
    epoch = 0
    W = init_weights()
    MSEs = [MSE(Xs, Ds, W)]
    plot_results(Xs, Ds, W, epoch)

    # Unfortunately, the only effective way to stop gradient descent is using
    # a combination of max iteration and minimum MSE
    while epoch <= 15000 and MSEs[-1] > epsilon:
        epoch += 1
        for idx in range(len(Xs)):
            # Feed forward
            outs, lfs = forward(W, Xs[idx])
            # Backpropagate
            grads = backward(outs, Ds[idx], W, lfs)
            # Get new weights
            W = update(W, grads, eta)

        MSEs.append(MSE(Xs, Ds, W))
        # If the error increased, reduce eta
        if MSEs[-1] > MSEs[-2]: eta = eta * 0.9
        if epoch % 1000 == 0: plot_results(Xs, Ds, W, epoch)

    return W, MSEs

def main():
    Xs, Ds = generate_data()
    W, MSEs = train(Xs, Ds, 0.01, 0.01)

    print('Starting MSE: ' + str(MSEs[0]) + ', Final MSE: ' + str(MSEs[-1]))

    plot_results(Xs, Ds, W, len(MSEs))
    plot_MSE(MSEs)

    return

if __name__ == '__main__':
    main()