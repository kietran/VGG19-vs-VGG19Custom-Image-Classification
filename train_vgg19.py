import torch
from torch.utils.data import DataLoader
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import vgg19
from dataset import MotorbikeDataset
from torch import optim, nn
from tqdm.autonotebook import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from torchvision.transforms import Resize, Compose, ToTensor

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', '-d', type=str, default='Dataset/motorbike')
    parser.add_argument("--img_size", "-i", type=int, default=224)
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=16)
    parser.add_argument('--momentum', '-m', type=int, default=0.9)
    parser.add_argument('--learning_rate', '-l', type=float, default=0.001)
    parser.add_argument('--checkpoint_load', '-c', type=str, default=None)
    parser.add_argument('--checkpoint_save', '-s', type=str, default='vgg19_checkpoint')
    parser.add_argument('--tensorboad_dir', '-t', type=str, default='vgg19_tensorboard')
    args = parser.parse_args()
    return args

def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: 
    plt.imshow(cm, interpolation='nearest', cmap='Wistia')
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    writer.add_figure('confusion_matrix', figure, epoch)

def train(args):
    if not os.path.isdir(args.checkpoint_save):
        os.makedirs(args.checkpoint_save)

    if not os.path.isdir(args.tensorboad_dir):
        os.makedirs(args.tensorboad_dir)

    transform = Compose([
        ToTensor(),
        Resize((args.img_size, args.img_size), antialias=True)
    ])
    writer = SummaryWriter(log_dir=args.tensorboad_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data = MotorbikeDataset(root_path=args.data_path, transform=transform, is_train=True)
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    valid_data = MotorbikeDataset(root_path=args.data_path, transform=transform, is_train=False)
    valid_dataloader = DataLoader(
        dataset=valid_data,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    model = vgg19()
    model.classifier[6] = nn.Linear(in_features=4096, out_features=len(train_data.categories))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    
    if args.checkpoint_load and os.path.isfile(args.checkpoint_load):
        checkpoint = torch.load(args.checkpoint_load)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_params'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(start_epoch)
    else:
        best_acc = -1
        start_epoch = 0

    num_iters = len(train_dataloader)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_progress_bar = tqdm(iterable=train_dataloader, colour='cyan')
        train_losses = []
        for iter, (images, labels) in enumerate(train_progress_bar):
            images = images.to(device)
            labels = labels.to(device)

            # Forward
            output = model(images)
            loss = criterion(output, labels)
            train_losses.append(loss.item())
            loss_val = np.mean(train_losses)
            train_progress_bar.set_description(f"Train: Epoch {epoch}/{args.epochs}. Loss: {loss_val}")
            writer.add_scalar('Train/Loss', loss_val, epoch*num_iters+iter)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        all_predictions = []
        all_labels = []
        valid_losses = []

        for iter, (images, labels) in enumerate(valid_dataloader):
            with torch.inference_mode():
                images = images.to(device)
                labels = labels.to(device)

                predictions = model(images)
                all_predictions.extend(torch.argmax(predictions, dim=1).tolist())
                all_labels.extend(labels.tolist())

                loss = criterion(predictions, labels)
                valid_losses.append(loss.item())

        acc_score = accuracy_score(all_labels, all_predictions)
        loss_val = np.mean(valid_losses)
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        writer.add_scalar('Val/Loss', loss_val, epoch)
        writer.add_scalar('Val/Acc', acc_score, epoch)
        plot_confusion_matrix(writer, conf_matrix, train_data.categories, epoch)
        print(f'Valid: Epoch {epoch}/{args.epochs}. Loss: {loss_val:0.4f} . Acc: {acc_score:0.2f}')

        checkpoint = {
            'model_params': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch+1,
            'best_acc': best_acc
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_save, "last.pt"))
        if acc_score > best_acc:
            best_acc = acc_score
            torch.save(model.state_dict(), os.path.join(args.checkpoint_save, "best.pt"))

if __name__ == "__main__":
    args = get_args()
    train(args)