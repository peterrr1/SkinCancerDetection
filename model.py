from torch import nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from pipeline import ProcessDataset
from torch.utils.data import DataLoader, random_split
from torch import Generator
from torch import optim
import torch
from tqdm import tqdm





def train(model, train_data, valid_data, num_epochs):
    model.train()

    criterion = nn.BCEWithLogitsLoss()
    activation_layer = nn.Sigmoid()
    optimizer = optim.Adamax(params = model.parameters(), lr = 0.001, weight_decay=0.0001)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    threshold = 0.5

    if not next(model.parameters()).is_cuda:
        model.to(device)


    for layers in model.features.parameters():
        layers.requires_grad = False

    for epoch in range(num_epochs):
        for _, (images, labels) in enumerate(pbar := tqdm(train_data)):
            
            images, labels = images.to(device), labels.to(device)
            images, labels = images.float(), labels.float()
            labels = labels.unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            outputs = activation_layer(outputs)

            result = (outputs > threshold)
 
            labels = labels

            accuracy = (result == labels).sum().item() / len(result)
            

            pbar.set_description(f'TRAINING - EPOCH [{epoch + 1}/{num_epochs}]; Loss [{loss.item()}]; Accuracy [{accuracy}]')
        validate(model, valid_data)




def validate(model, valid_data):
    model.eval()

    activation_layer = nn.Sigmoid()
    criterion = nn.BCEWithLogitsLoss()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    threshold = 0.5

    if not next(model.parameters()).is_cuda:
            model.to(device)

    avg_loss, avg_acc = 0, 0

    with torch.no_grad():
        for _, (images, labels) in enumerate(valid_data):
            images, labels = images.to(device), labels.to(device)
            images, labels = images.float(), labels.float()
            labels = labels.unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            outputs = activation_layer(outputs)
            result = (outputs > threshold)


            accuracy = (result == labels).sum().item() / len(result)

            avg_acc += accuracy
            avg_loss += loss.item()

        
    print(f'VALIDATION AVERAGE - Avg loss [{avg_loss/len(valid_data)}]; Avg accuracy [{avg_acc/len(valid_data)}]')




def test(model, test_data):
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    activation_layer = nn.Sigmoid()
    threshold = 0.5

    avg_loss, avg_acc = 0, 0
    
    if not next(model.parameters()).is_cuda:
        model.to(device)

    with torch.no_grad():
        for _, (images, labels) in enumerate(test_data):
            images, labels = images.to(device), labels.to(device)
            images, labels = images.float(), labels.float()
            labels = labels.unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            outputs = activation_layer(outputs)
            result = (outputs > threshold)

            accuracy = (result == labels).sum().item() / len(result)

            avg_acc += accuracy
            avg_loss += loss.item()

    
    print(f'TEST AVERAGE - Loss [{avg_loss/len(test_data)}]; Accuracy [{avg_acc/len(test_data)}]')




def main():

    DATA_PATH = '/home/peter/Documents/ASZTALI/KaggleCompetitions/SkinCancerDetection/data/train-image/image'
    LABEL_PATH = '/home/peter/Documents/ASZTALI/KaggleCompetitions/SkinCancerDetection/data/train-metadata.csv'
    PATH_TO_SAVE = './models/isic_2024_3.pth'
    BATCH_SIZE = 32
    EPOCHS = 100

    model = efficientnet_v2_s(weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1)

    _classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features = 1280, out_features = 512),
        nn.Dropout(p = 0.5, inplace=True),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(in_features=512, out_features = 256),
        nn.Dropout(p = 0.5, inplace = True),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(in_features = 256, out_features = 96),
        nn.Dropout(p = 0.5, inplace=True),
        nn.BatchNorm1d(96),
        nn.ReLU(),
        nn.Linear(in_features=96, out_features=1)
    )

    model.classifier = _classifier

    ## Modify the first convolutional layer since it uses a single channel grayscale image
    model.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    
    

    dataset = ProcessDataset(data_path=DATA_PATH, label_path=LABEL_PATH)

    train_ds, valid_ds, test_ds = random_split(dataset, [0.7, 0.2, 0.1], generator = Generator().manual_seed(0))

    train_set = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle = True)
    valid_set = DataLoader(valid_ds, batch_size = BATCH_SIZE, shuffle = True)
    test_set = DataLoader(test_ds, batch_size = BATCH_SIZE, shuffle = True)

    
    
    train(model, train_set, valid_set, EPOCHS)
    test(model, test_data=test_set)


    #torch.save(model.state_dict(), PATH_TO_SAVE)
main()
    