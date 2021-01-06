import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
import copy
from utils.focal_loss import FocalLoss
import argparse, os
import time, datetime


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net-name', dest='net_name', type=str, default='resnet50')
    parser.add_argument('--data', dest='data', type=str, default='./data/class', help='The data path!')
    parser.add_argument('--resize', dest='resize', type=int, default=300, help='img resize')
    parser.add_argument('--crop-size', dest='crop_size', type=int, default=224, help='crop resized img enter net!')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=16)
    parser.add_argument('--epochs', dest='epochs', type=int, default=100)
    parser.add_argument('--classes', dest='classes', type=int, default=3, help='class number')
    parser.add_argument('--save-path', dest='save_path', type=str, default='./model', help='save model path!')
    parser.add_argument('--pre', dest='pre_training', action='store_true', help='pre_training or not!')
    parser.add_argument('--model-path', dest='model_path', type=str, default='./model/epoch0_acc0.3676_loss1.0442.pt',
                        help='pre_training model path!')
    parser.add_argument('--focal-loss', dest='focal_loss', action='store_true', help='use focal loss!')
    parser.add_argument('--fe', dest='feature_extract', default=True,
                        help='Flag for feature extractiing, When False, wei finetune the whole mode, When True, we only update the reshaped layer paras!')
    args = parser.parse_args()
    return args


args = parse_args()

transform = {
    'train': transforms.Compose([transforms.Resize(args.resize),
                                 transforms.CenterCrop(args.crop_size),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'val': transforms.Compose([transforms.Resize(args.resize),
                               transforms.CenterCrop(args.crop_size),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def Net(feature_extract, net_name):
    net = None
    ## 設置 requires_grad=False 凍結參數,以便在backward()中不計算梯度.
    if net_name in ['Mobilenet_v2', 'mobilenet_v2']:
        net = torchvision.models.mobilenet_v2(pretrained=True)
        set_parameter_requires_grad(net, feature_extract)
        classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(net.last_channel, args.classes),
        )
        net.classifier = classifier

    elif net_name in ['googlenet', 'Googlenet']:
        net = torchvision.models.googlenet(pretrained=True)
        set_parameter_requires_grad(net, feature_extract)
        in_features = net.fc.in_features
        fc = nn.Linear(in_features, args.classes, bias=True)
        net.fc = fc

    elif net_name in ['inception_v3', 'Inception_v3']:
        net = torchvision.models.inception_v3(pretrained=True)
        set_parameter_requires_grad(net, feature_extract)
        num_ftrs = net.AuxLogits.fc.in_features
        AuxLogits = nn.Linear(num_ftrs, args.classes)
        net.AuxLogits.fc = AuxLogits
        num_ftrs = net.fc.in_features
        fc = nn.Linear(num_ftrs, args.classes)
        net.fc = fc

    elif net_name in ['resnet50', 'Resnet50']:
        net = torchvision.models.resnet50(pretrained=True)
        set_parameter_requires_grad(net, feature_extract)
        in_features = net.fc.in_features
        fc = nn.Linear(in_features, args.classes)
        net.fc = fc

    elif net_name in ['densenet121', 'Densnet121']:
        net = torchvision.models.densenet121(pretrained=True)
        set_parameter_requires_grad(net, feature_extract)
        in_features = net.classifier.in_features
        classifier = nn.Linear(in_features, args.classes)
        net.classifier = classifier

    else:
        assert net, 'please add yourself net or input right net name!'

    print(list(net.children()))
    net = net.to(device)
    return net


def train(net, data_loader, optim, criterion, exp_lr_scheduler, epochs, net_name):
    best_acc = 0
    best_model_wts = copy.deepcopy(net.state_dict())
    for epoch in range(epochs):
        print('Epoch{}/{}'.format(epoch, epochs - 1))

        for phase in ['train', 'val']:
            if phase == 'train':
                exp_lr_scheduler.step()
                net.train()
            if phase == 'val':
                net.eval()

            running_loss = 0
            running_corects = 0
            step = 1
            steps = len(data_loader[phase])
            for inputs, labels in data_loader[phase]:
                strat_time = time.time()
                inputs = inputs.to(device)
                labels = labels.to(device)

                optim.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    if net_name in ['inception_v3', 'Inception_v3'] and phase == 'train':
                        outputs, aux_outputs = net(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = net(inputs)
                        loss = criterion(outputs, labels)
                    _, predict = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optim.step()

                    end_time = time.time()
                    residual_time = str(datetime.timedelta(seconds=(steps - step) * (end_time - strat_time)))[:-7]
                    print("\r%d/%d [%s>%s] -ETA: %s - loss: %4f\n" % (
                        step, steps, '=' * int(29 * step / steps), '.' * (29 - int(29 * step / steps)), residual_time, loss), end='', flush=True)
                    step += 1

                running_loss += loss.item() * inputs.size(0)
                running_corects += torch.sum(predict == labels.data)

            epoch_loss = running_loss / data_size[phase]
            epoch_acc = running_corects.double() / data_size[phase]

            print('{} Loss:{:.4f} acc:{:.4f}'.format(phase, epoch_loss, epoch_acc))
            if epoch_acc > best_acc and phase == 'val':
                best_acc = epoch_acc
                torch.save(net.state_dict(), os.path.join(args.save_path,
                                                          'epoch{}_acc{:.4f}_loss{:.4f}.pt'.format(epoch, epoch_acc, epoch_loss)))
                best_model_wts = copy.deepcopy(net.state_dict())

    print('Best val acc', best_acc)
    net.load_state_dict(best_model_wts)
    torch.save(net.state_dict(), os.path.join(args.save_path, 'best_model_acc{:.4f}.pt'.format(best_acc)))


if __name__ == '__main__':
    # load data
    imgs_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(args.data, x), transform=transform[x]) for x in ['train', 'val']}
    data_loader = {
        x: torch.utils.data.DataLoader(imgs_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=0) for x in ['train', 'val']}
    data_size = {x: len(imgs_datasets[x]) for x in ['train', 'val']}
    img_class = imgs_datasets['train'].classes

    net = Net(args.feature_extract, args.net_name)

    # load model
    if args.pre_training:
        print('load model:{}'.format(args.model_path))
        net.load_state_dict(torch.load(args.model_path))

    # Observe that all parameters are being optimized
    if args.feature_extract:
        params_to_update = []
        for name, param in net.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
    else:
        params_to_update = net.parameters()
    optim = torch.optim.Adam(params=params_to_update, lr=0.001)
    # optim = torch.optim.SGD(params=net.parameters(), lr=0.001, momentum=0.9)

    # Loss function
    criterion = FocalLoss() if args.focal_loss else nn.CrossEntropyLoss()
    print('criterion: ', criterion)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=7, gamma=0.1)
    train(net, data_loader, optim, criterion, exp_lr_scheduler, args.epochs, args.net_name)
