import numpy as np

def relu(x): return x * (x > 0)

N = 2
T = 20000
learning_rate = .06
init_factor = .04


def build_parameters(N, init_factor=.1, input_size=2, output_size=1):
	W1 = np.random.rand(N, input_size) * init_factor 
	b1 = np.random.rand(N) * init_factor
	W2 = np.random.rand(output_size, N) * init_factor
	b2 = np.random.rand(output_size) * init_factor
	return {
		'W1': W1,
		'b1': b1,
		'W2': W2,
		'b2': b2
	}


def forward_pass(parameters, x):
	W1 = parameters['W1']
	b1 = parameters['b1']
	W2 = parameters['W2']
	b2 = parameters['b2']
	o1 = np.matmul(W1, x) + b1
	h = relu(o1)
	o2 = np.matmul(W2, h) + b2
	y = relu(o2)
	return {
		'x': x,
		'o1': o1,
		'h': h,
		'o2': o2,
		'y': y
	}


def backpropagate(parameters, activations, y, gradients):
	error = activations['y'] - y
	o2_cutoff = activations['o2'] >= 0
	o1_cutoff = activations['o1'] >= 0
	gradients['dW2'] = gradients['dW2'] + 2 * error * activations['h'] * o2_cutoff
	gradients['db2'] = gradients['db2'] + 2 * error * o2_cutoff
	gradients['dW1'] = gradients['dW1'] + 2 * error * (activations['x'] * parameters['W2']) * o1_cutoff
	gradients['db1'] = gradients['db1'] + 2 * error * parameters['W2'][0] * o1_cutoff


def regularization_term(parameters, lmbd=.01):
	term = 0
	for p in parameters:
		term += np.linalg.norm(parameters[p])
	return lmbd * term


def update_parameters(parameters, gradients, learning_rate=.03):
	parameters['W1'] = parameters['W1'] - learning_rate * gradients['dW1']
	parameters['b1'] = parameters['b1'] - learning_rate * gradients['db1']
	parameters['W2'] = parameters['W2'] - learning_rate * gradients['dW2']
	parameters['b2'] = parameters['b2'] - learning_rate * gradients['db2']


def zero_gradients(N, init_factor=.01, input_size=2, output_size=1):
	return {
		'dW1': np.zeros([N, 2]),
		'db1': np.zeros([N]),
		'dW2': np.zeros([1, N]),
		'db2': np.zeros([1])
	}


def run_xor_network(data):
	parameters = build_parameters(2)

	for t in range(T):
		cost = 0
		gradients = zero_gradients(2)
		for io_pair in data:
			x, y = io_pair
			activations = forward_pass(parameters, x)
			cost += (y - activations['y']) ** 2 + regularization_term(parameters)
			backpropagate(parameters, activations, y, gradients)
		cost /= len(data)
		if t % 10 == 0:
			print('step {}: cost = {}'.format(t, cost))
		for grad in gradients:
			gradients[grad] /= len(data)
		update_parameters(parameters, gradients, learning_rate)



xor_data = [
	(np.array([0, 0]), 0),
	(np.array([1, 0]), 1),
	(np.array([0, 1]), 1),
	(np.array([1, 1]), 0)
]

run_xor_network(xor_data)


