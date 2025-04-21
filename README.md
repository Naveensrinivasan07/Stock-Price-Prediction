# Stock-Price-Prediction


## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset


## Design Steps

### Step 1:
Write your own steps

### Step 2:

### Step 3:



## Program
#### Name:NAVEEN S
#### Register Number:212222240070
Include your code here
```Python 
# Define RNN Model
class RNNModel(nn.Module):
  def __init__(self, input_size=1,hidden_size=64, num_layers=2, output_size=1):
    super(RNNModel, self).__init__()
    self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)
  def forward(self, x):
    out, _ = self.rnn(x)
    out = self.fc(out[:, -1, :])
    return out
model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Train the Model

criterion = nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

epochs=20
model.train()
train_losses=[]
for epoch in range(epochs):
  epoch_loss=0
  for x_batch,y_batch in train_loader:
    x_batch,y_batch=x_batch.to(device),y_batch.to(device)
    optimizer.zero_grad()
    outputs=model(x_batch)
    loss=criterion(outputs,y_batch)
    loss.backward()
    optimizer.step()
    epoch_loss+=loss.item()
  train_losses.append(epoch_loss/len(train_loader))
  print(f"Epoch [{epoch+1}/{epochs}] Loss: {train_losses[-1]:.4f}")
```

## Output

### True Stock Price, Predicted Stock Price vs time

![ஸ்கிரீன்ஷாட் 2025-04-21 111334](https://github.com/user-attachments/assets/1838e2e3-ac58-4d15-beb1-fa071a2bcb46)


### Predictions 


![ஸ்கிரீன்ஷாட் 2025-04-21 111424](https://github.com/user-attachments/assets/dd777064-4680-4cd2-bb25-881b449ea2f6)


## Result
Thus , a Recurrent Neural Network model for stock price prediction has successfully been devoloped.




