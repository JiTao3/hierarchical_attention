from model.encoder import Encoder
from util.dataset import PlanDataset
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = PlanDataset(root_dir="/data1/jitao/dataset/cardinality/deep_plan")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

train_size = int(len(dataset) * 0.9)
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

encoder = Encoder(d_feature=9 + 6 + 64, d_model=512, d_ff=512, N=6)

criterion = nn.MSELoss()
optimizer = optim.SGD(encoder.parameters(), lr=0.001, momentum=0.9)


epoch_size = 10


def train():
    for epoch in range(epoch_size):
        print("epoch : ", epoch)
        running_loss = 0.0
        for i, data in enumerate(train_dataset):
            tree, nodemat, leafmat, label = data
            optimizer.zero_grad()
            output = encoder(tree, nodemat, leafmat)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 0 and i != 0:
                print("[%d, %5d] loss: %4d" % (epoch + 1, i + 1, loss / 200))
                running_loss = 0.0
        result = []
        test_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(test_dataset):
                tree, nodemat, leafmat, label = data
                test_output = encoder(tree, nodemat, leafmat)
                if epoch == epoch_size - 1:
                    result.append((label, test_output))
                loss = criterion(test_output, label)
                test_loss += loss.item()
                print("test loss: ", test_loss / test_size)
        return result


if __name__ == "__main__":
    print(device)
    result = train()
    with open("data/resutlv1.0.txt", "w") as f:
        f.write("\n".join("{} {}".format(x[0], x[1])) for x in result)
