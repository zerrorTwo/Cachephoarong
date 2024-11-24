import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=24, hidden_size=48, output_size=4):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.score = 0
        
        # Khởi tạo trọng số ngẫu nhiên
        self.weights = [
            np.random.randn(input_size, hidden_size) * np.sqrt(1.0/input_size),
            np.random.randn(hidden_size, output_size) * np.sqrt(1.0/hidden_size)
        ]
        
        # Khởi tạo bias
        self.biases = [
            np.zeros((1, hidden_size)),
            np.zeros((1, output_size))
        ]
        
    def forward(self, x):
        """Forward pass qua neural network"""
        # Hidden layer
        hidden = np.dot(x, self.weights[0]) + self.biases[0]
        hidden = np.tanh(hidden)  # activation function
        
        # Output layer
        output = np.dot(hidden, self.weights[1]) + self.biases[1]
        output = self.softmax(output)
        
        return output
    
    @staticmethod
    def softmax(x):
        """Hàm softmax cho output layer"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=1, keepdims=True)