import math
from model.encoder import Encoder
from util.dataset import PlanDataset
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = PlanDataset(root_dir="/data1/jitao/dataset/cardinality/deep_plan")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

train_size = int(len(dataset) * 0.9)
test_size = len(dataset) - train_size


# train_temp = [dataset[i] for i in range(10)]
# test_temp = [dataset[i] for i in range(5)]

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

encoder = Encoder(d_feature=9 + 6 + 64, d_model=512, d_ff=512, N=6)

criterion = nn.MSELoss()
optimizer = optim.SGD(encoder.parameters(), lr=0.0001, momentum=0.9)


epoch_size = 2


def train():
    result = []
    for epoch in range(epoch_size):
        print("epoch : ", epoch)
        running_loss = 0.0
        for i, data in enumerate(train_dataset):
            tree, nodemat, leafmat, label = data
            optimizer.zero_grad()
            output = encoder(tree, nodemat, leafmat)
            output = output.squeeze(-1)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if math.isnan(running_loss):
                print("nan: ", i, "\t", running_loss)

            if i % 200 == 0 and i != 0:
                print("[%d, %5d] loss: %4f" % (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
        test_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(test_dataset):
                tree, nodemat, leafmat, label = data
                test_output = encoder(tree, nodemat, leafmat)
                if epoch == epoch_size - 1:
                    result.append((label, test_output))
                loss = criterion(test_output, label)
                test_loss += loss.item()
                if i % 200 == 0 and i != 0:
                    print("test loss: ", test_loss / test_size)
    return result


def dataset_test():
    for i, data in enumerate(test_dataset):
        tree, nodemat, leafmat, label = data
        print(label)


if __name__ == "__main__":
    result = train()
    # result = [(1.1, 2.2), (3.3, 4.4), (5.5, 6.6)]
    with open("data/resutlv1.0.txt", "w") as f:
        f.write("\n".join("{} {}".format(x[0], x[1]) for x in result))

    # torch.save(encoder, "model_parameter/encoderv1.0.pkl")
    # dataset_test()
