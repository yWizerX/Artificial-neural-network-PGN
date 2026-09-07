"""
PGN (Polynomial Gated Network) - V9 Release 1

Usage example:
--------------
from neural import PGN
import numpy as np

# Create model (e.g. 4 input features, 3 output classes)
model = PGN(input_dim=4, output_dim=3, lr=0.01)

# Training loop
X = np.random.randn(100, 4)
y = np.random.randint(0, 3, size=100)

for epoch in range(500):
    loss = model.train_step(X, y)

# Predict
predictions = model.predict(X)
"""

import numpy as np


class PGN:
    def __init__(self, input_dim, output_dim, lr=0.01):
        self.lr = lr
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Branch G1
        self.W_g1 = np.random.randn(input_dim, 2) * np.sqrt(2.0 / input_dim)
        self.b_g1 = np.zeros((1, 2))
        self.W_g1_out = np.random.randn(2, 1) * np.sqrt(2.0 / 2)

        # Branch G2
        self.W_g2 = np.random.randn(input_dim, 2) * np.sqrt(2.0 / input_dim)
        self.b_g2 = np.zeros((1, 2))
        self.W_g2_out = np.random.randn(2, 1) * np.sqrt(2.0 / 2)

        # Polynomial & Gating Node
        self.b_red = np.zeros((1, 1))

        # Output Layer Fusion (G1_out + G2_out + Red_node = 3 signals)
        self.W_fuse = np.random.randn(3, output_dim) * np.sqrt(2.0 / 3)
        self.b_out = np.zeros((1, output_dim))

    def _leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, x * alpha)

    def _leaky_relu_grad(self, x, alpha=0.01):
        dx = np.ones_like(x)
        dx[x <= 0] = alpha
        return dx

    def _sigmoid(self, x):
        x_clipped = np.clip(x, -12.0, 12.0)
        return 1.0 / (1.0 + np.exp(-x_clipped))

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        self.X = X

        # Branch 1
        self.z_g1 = np.dot(X, self.W_g1) + self.b_g1
        self.a_g1 = self._leaky_relu(self.z_g1)
        self.out_g1 = np.dot(self.a_g1, self.W_g1_out)

        # Branch 2
        self.z_g2 = np.dot(X, self.W_g2) + self.b_g2
        self.a_g2 = self._leaky_relu(self.z_g2)
        self.out_g2 = np.dot(self.a_g2, self.W_g2_out)

        # Polynomial Gated Node (Red Node)
        self.poly = (self.out_g1 ** 2) + (self.out_g1 * self.out_g2) + (self.out_g2 ** 2)
        self.gate = self._sigmoid(self.out_g2)
        self.red_node = (self.poly * self.gate) + self.b_red

        # Fusion & Classification
        self.signals = np.hstack([self.out_g1, self.out_g2, self.red_node])
        self.logits = np.dot(self.signals, self.W_fuse) + self.b_out
        self.probs = self._softmax(self.logits)

        return self.probs

    def backward(self, y_true_onehot):
        N = self.X.shape[0]

        dlogits = (self.probs - y_true_onehot) / N
        dW_fuse = np.dot(self.signals.T, dlogits)
        db_out = np.sum(dlogits, axis=0, keepdims=True)

        dsignals = np.dot(dlogits, self.W_fuse.T)
        dout_g1_fused = dsignals[:, 0:1]
        dout_g2_fused = dsignals[:, 1:2]
        dred_node = dsignals[:, 2:3]

        db_red = np.sum(dred_node, axis=0, keepdims=True)
        dpoly = dred_node * self.gate
        dgate = dred_node * self.poly

        dout_g2_gate = dgate * (self.gate * (1.0 - self.gate))

        dout_g1_poly = dpoly * (2.0 * self.out_g1 + self.out_g2)
        dout_g2_poly = dpoly * (2.0 * self.out_g2 + self.out_g1)

        dout_g1 = dout_g1_fused + dout_g1_poly
        dout_g2 = dout_g2_fused + dout_g2_poly + dout_g2_gate

        dW_g1_out = np.dot(self.a_g1.T, dout_g1)
        da_g1 = np.dot(dout_g1, self.W_g1_out.T)
        dz_g1 = da_g1 * self._leaky_relu_grad(self.z_g1)
        dW_g1 = np.dot(self.X.T, dz_g1)
        db_g1 = np.sum(dz_g1, axis=0, keepdims=True)

        dW_g2_out = np.dot(self.a_g2.T, dout_g2)
        da_g2 = np.dot(dout_g2, self.W_g2_out.T)
        dz_g2 = da_g2 * self._leaky_relu_grad(self.z_g2)
        dW_g2 = np.dot(self.X.T, dz_g2)
        db_g2 = np.sum(dz_g2, axis=0, keepdims=True)

        self.W_fuse -= self.lr * dW_fuse
        self.b_out -= self.lr * db_out
        self.b_red -= self.lr * db_red

        self.W_g1_out -= self.lr * dW_g1_out
        self.W_g1 -= self.lr * dW_g1
        self.b_g1 -= self.lr * db_g1

        self.W_g2_out -= self.lr * dW_g2_out
        self.W_g2 -= self.lr * dW_g2
        self.b_g2 -= self.lr * db_g2

    def train_step(self, X, y):
        N = X.shape[0]
        y_onehot = np.zeros((N, self.output_dim))
        y_onehot[np.arange(N), y] = 1.0

        probs = self.forward(X)
        loss = -np.sum(y_onehot * np.log(probs + 1e-12)) / N
        self.backward(y_onehot)

        return loss

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
