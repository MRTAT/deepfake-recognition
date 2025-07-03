from argparse import ArgumentParser
from tqdm import tqdm
import os

import shutil    # using for delete tensorboard

from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch
from torchvision.transforms.v2 import ToTensor, Resize, Compose, RandomAffine, ColorJitter
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

from dataset import Deepfake
from cnn_model import Cnn

def get_args():
    parser = ArgumentParser(description="Deepfake training scrip")
    parser.add_argument("--root", "-r", type=str, default="dataset_std")
    parser.add_argument("--epochs", "-e", type=int, default=88, help="Number of epoch")
    parser.add_argument("--batch_size", "-b", type=int, default=256, help="Enter number of batch size")
    parser.add_argument("--image_size", "-s", type=int, default=224, help="Image size")
    parser.add_argument("--num_classes", "-c", type=int, default=2, help="Number of class")

    parser.add_argument("--num_workers", type=int, default=8, help="Number of core in your CPU (higher if pc stronger)")
    parser.add_argument("--lr", "-l", type=float, default=1e-3, help="Learning rate: 1e-3")
    parser.add_argument("--momentum", "-m", type=float, default=0.95, help="Momentum : 0.95")

    parser.add_argument("--logging", type=str, default="tensorboard", help="History training")
    parser.add_argument("--trained_model", default="trained_model", help="File trained")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file")

    args = parser.parse_args()
    return args

def plot_confusion_matrix(writer, cm , class_names, epoch):
    figure = plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap='Wistia')
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    threshold = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
    plt.tight_layout()
    plt.ylabel("True_Label")
    plt.xlabel("Predict_Label")
    writer.add_figure("confusion_matrix", figure, epoch)

if __name__ == '__main__':
    # epoches = 100     # | change by parser

    args = get_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else: device = torch.device("cpu")

    # --------------------------------------------

    train_trans = Compose([
        RandomAffine(  # Data augmentation about : rotation - shear - zoom
            degrees=(-5, 5), # rotation random from -5 to 5 deg
            translate=(0.05, 0.05),  # move random 5 per by hori and verti
            scale=(0.85, 1.15),  # zoom to 1.15 and zoom nho 0.85
            shear=5
        ),
        Resize((args.image_size,args.image_size)),
        ColorJitter(
            brightness=0.5,
            contrast=0.5,
            saturation=0.25,
            hue=0.1
        ),
        ToTensor(),
    ])

    test_trans = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor(),
    ])


    # --------------------------------------------
    train_dataset = Deepfake(
        root=args.root,
        train=True,
        transform=train_trans
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    # # Visualization image cause using Data augmentation
    # image, _ = train_dataset.__getitem__(888)
    # image = (torch.permute(image, (1, 2, 0))*255.).numpy().astype(np.uint8)  # s1: convert to BGR fix To tensor (cv2)  s2: convert tensor to numpy and astype(np.uint8) -> cause photos in opencv have to be int khong dau 8 bit
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # cv2.imshow("Test image", image)
    # cv2.waitKey(0)
    # exit()
    # End -----------------

    test_dataset = Deepfake(
        root=args.root,
        train=False,
        transform=test_trans

    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True
    )

    # cuz logging after train_videos then having many so we will xoa bot no di
    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)

    # Create path to save model
    if not os.path.isdir(args.trained_model):  # check exit
        os.mkdir(args.trained_model)  # if not yet then create

    writer = SummaryWriter(args.logging)  # write on tensorboard
    iters = len(train_loader)

    # Initialization
    model = Cnn(num_classes=args.num_classes).to(device)  # to(device) -> set cuda
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


    # Load model
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        best_accuracy = checkpoint["best_accuracy"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    else:
        start_epoch = 0
        best_accuracy = 0
    # End load




    # Training
    for epoch in range(start_epoch, args.epochs):  # start_epoch in case: train_videos continuous
        model.train()
        progress_bar = tqdm(train_loader, colour="green")  #cyan
        train_loss = []

        for iter, (images, labels) in enumerate(progress_bar):  # progress_bar change for train_loader
            images = images.to(device)
            labels = labels.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)


            # Backward
            optimizer.zero_grad()  # not storage gradient after backward
            loss.backward()
            optimizer.step()

            # Average loss
            train_loss.append(loss.item())
            mean_loss = np.mean(train_loss)

            # progress bar
            progress_bar.set_description("epoch: {}/{} | interation: {}/{} | Loss: {:.3f} ".format(epoch + 1, args.epochs, iter + 1, iters,mean_loss))
            writer.add_scalar("Train/Loss", mean_loss, epoch * iters + iter)  # history of loss

        model.eval()
        all_prediction = list()
        all_labels = list()
        for iter, (images, labels) in enumerate(test_loader):
            all_labels.extend(labels)

            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                predict = model(images)

                indices = torch.argmax(predict, dim=1)
                all_prediction.extend(indices)
                loss = criterion(predict, labels)

        all_labels = [label.item() for label in all_labels]
        all_prediction = [predict.item() for predict in all_prediction]

        # Confusion matrix
        plot_confusion_matrix(writer, confusion_matrix(all_labels, all_prediction), class_names=test_dataset.categories, epoch=epoch)
        # test_dataset.categories : --> Extract list of classes
        # End

        accuracy = accuracy_score(all_labels, all_prediction)
        print("Epoch: {}: Accuracy: {}".format(epoch+1, accuracy))
        writer.add_scalar("Val/Accuracy", accuracy, epoch)  # just epoch cuz in test_videos : done every epoch then save

        # Tensorboard for Weight and Bias
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)

        # SAVING MODEL -------------------------------------

        # Save last model for tomorrow train_videos
        checkpoint = {
            "epoch" : epoch+1,  # epoch+1 -> next epoch train_videos
            "model" : model.state_dict(),  # Save wight and bias
            "optimizer" : optimizer.state_dict()  # train_videos dung learning rate hom qua stop
        }
        torch.save(checkpoint, "{}/last_cnn.pt".format(args.trained_model))  # if hom nay trained 20 epochs, ngay mai muon train_videos 80 epochs then dung last

        # Save best accuracy
        if accuracy > best_accuracy:
            checkpoint = {
                "epoch": epoch + 1,  # epoch+1 -> next epoch train_videos
                "best_accuracy": best_accuracy,
                "model": model.state_dict(),  # Save wight and bias
                "optimizer": optimizer.state_dict()  # train_videos dung learning rate hom qua stop
            }
            torch.save(checkpoint, "{}/best_cnn.pt".format(args.trained_model))  # using if wanna sent cho colleague or boss
            best_accuracy = accuracy
