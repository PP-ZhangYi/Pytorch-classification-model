import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from collections import Counter

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def Load_Data(img_size, imgs_path):
    transform = transforms.Compose([transforms.Resize(img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    testset = torchvision.datasets.ImageFolder(root=imgs_path, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=0)
    return (testset, testloader)


def Load_Model(model_path):
    ## 訓練時只保存權重參數，需加載網絡
    # net = torchvision.models.mobilenet_v2(pretrained=False)
    # classifier = nn.Sequential(
    #     nn.Dropout(0.2),
    #     nn.Linear(net.last_channel, 3),
    # )
    # net.classifier = classifier
    # model = net.to(device)
    # state_dict = torch.load(model_path)
    # model.load_state_dict(state_dict)

    # 訓練時保存了整個模型
    model = torch.load(model_path)
    return model


def test(data, model):
    testset, testloader = data
    labels_list = testset.targets
    classes_list = testset.classes
    dict_right = {}
    dict_count = {}
    for i in range(len(classes_list)):
        dict_right[i] = 0
        dict_count[i] = Counter(labels_list)[i]
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs.to(device)
            labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            print('predicted:', predicted, 'labels:', labels)
            for i in range(len(labels)):
                if predicted[i] == labels[i]:
                    dict_right[int(labels[i])] += 1

    for i in range(len(classes_list)):
        acc = dict_right[i] / dict_count[i]
        print('Class {} Acc: {:.4f}'.format(classes_list[i], acc))


if __name__ == '__main__':
    model_path = r'./model/epoch0_acc0.9044_loss0.2180.pt'
    imgs_path = './data/class/val'
    img_size = 224
    model = Load_Model(model_path)
    testdata = Load_Data(img_size=img_size, imgs_path=imgs_path)
    test(testdata, model)
