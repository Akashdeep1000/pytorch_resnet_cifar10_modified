import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet
import glob

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

# print(model_names)

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',default= True,
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
best_prec1 = 0

args = parser.parse_args()

# print(model)


def main():
    global args, best_prec1
    args = parser.parse_args()

    model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    model.cuda()
    model.eval()

    if args.pretrained:
        print("=> Using pre-trained model")
        architecture_prefix = args.arch.split('-')[0]  # Get the architecture name before the hyphen
        pretrained_path = os.path.join("pretrained_models", f"{architecture_prefix}-*.th")
        
        pretrained_files = glob.glob(pretrained_path)
        if pretrained_files:
            latest_pretrained_file = max(pretrained_files, key=os.path.getctime)
            checkpoint = torch.load(latest_pretrained_file)
            model.load_state_dict(checkpoint['state_dict'])
            print(f"=> Loaded pre-trained checkpoint: {latest_pretrained_file}")
        else:
            print("=> No pre-trained checkpoint found for the selected architecture")


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

    num_correct = 0
    total_images = 0

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()  # Move input to the same device as the model
            target = target.cuda()

            output = model(input)
            probabilities = torch.nn.functional.softmax(output, dim=1)

            predicted_labels = torch.argmax(probabilities, dim=1)
            ground_truth_labels = target

            num_correct += (predicted_labels == ground_truth_labels).sum().item()
            total_images += input.size(0)

        classification_error = (1 - (num_correct / total_images)) * 100
        print(f"Total images: {total_images}")
        print(f"Correct predictions: {num_correct}")
        print(f"Classification error: {classification_error:.2f}%")
                
            

if __name__ == '__main__':
    main()




