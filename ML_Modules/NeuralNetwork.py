import torch
import torch.nn as nn
import torch.optim as optim


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, layer_size_list):
        super(NeuralNetwork, self).__init__()
        layer_size_list = [input_size] + layer_size_list + [output_size]
        self.fc = nn.ModuleList([
            nn.Linear(layer_size_list[i], layer_size_list[i + 1])
            for i in range(len(layer_size_list) - 1)
        ])
    
    def forward(self, x):
        for layer in self.fc[:-1]:
            x = torch.relu(layer(x))
        x = self.fc[-1](x)
        return x

def Train_NN(input_data, output_values, model, learning_rate, epochs):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(input_data)
        loss = criterion(outputs, output_values)
        loss.backward()
        optimizer.step()

        if epoch % int(epochs * (5 / 100)) == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')