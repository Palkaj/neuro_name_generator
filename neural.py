import torch
import torch.nn as nn
import random

# Задаем алфавит и символы, которые могут использоваться в именах
alphabet = 'abcdefghijklmnopqrstuvwxyz'
symbols = alphabet + ' .'

# Функция для генерации случайной буквы из алфавита
def random_letter():
    return random.choice(alphabet)

# Функция для генерации случайного имени персонажа заданной длины
def generate_name(length):
    return ''.join(random_letter() for _ in range(length))

# Класс для нейронной сети, генерирующей имена персонажей
class CharacterGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharacterGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, dropout=0.2)  # Используем LSTM с dropout
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden):
        output, hidden = self.rnn(input.view(1, 1, -1), hidden)
        output = self.fc(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))

# Функция для генерации имени с помощью нейронной сети
def generate_name_with_model(model, max_length):
    model.eval()
    with torch.no_grad():
        input = torch.zeros(1, max_length, len(symbols))
        hidden = model.init_hidden()
        name = ''
        for i in range(max_length):
            output, hidden = model(input[:, i, :].unsqueeze(0), hidden)
            _, topi = output.topk(1)
            topi = topi.squeeze().item()
            if topi == len(symbols) - 1:
                break
            else:
                name += symbols[topi]
        return name

# Параметры модели
input_size = len(symbols)
hidden_size = 256  # Увеличиваем размер скрытого слоя
output_size = len(symbols)
max_length = 20

# Создание и обучение модели
model = CharacterGenerator(input_size, hidden_size, output_size)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Обучение на примерах имен персонажей
num_epochs = 20000
for epoch in range(num_epochs):
    name = generate_name(random.randint(1, max_length))
    target = torch.tensor([symbols.index(c) for c in name])
    optimizer.zero_grad()
    hidden = model.init_hidden()
    loss = 0
    for i in range(len(name)):
        input_tensor = torch.zeros(1, 1, len(symbols))
        input_tensor[0, 0, symbols.index(name[i])] = 1  # one-hot кодирование
        output, hidden = model(input_tensor, hidden)
        loss += criterion(output.squeeze(0), torch.tensor([symbols.index(name[i])]))
    loss.backward()
    optimizer.step()
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Генерация имени с использованием обученной модели
generated_name = generate_name_with_model(model, max_length)
print("Generated Name:", generated_name)
