import torch
import math
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples, adjust_lr
from dataloader import get_loader
from naiveAE import NaiveAE
from val_loader_test import val_coco


def train():
    # validation results saving root

    val_image_folder = "D:/MScocoCaption数据集/val2014"
    batch_size = 4

    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_loader, train_dataset = get_loader(root_folder="D:/MScocoCaption数据集/train2014",
                                             annotation_file="coco/train2014_captions.txt",
                                             transform=transform,
                                             num_workers=2,
                                             batch_size=batch_size
                                             )

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cuda:0')
    load_model = False
    save_model = False
    train_CNN = False
    is_best = False

    # Hyper_parameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(train_dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    lrf = 0.1
    num_epochs = 3001
    best_bleu4 = 0

    # for tensorboard
    writer = SummaryWriter("runs/coco")
    step = 0

    # Initialize model, loss etc
    model = NaiveAE(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / num_epochs)) / 2) * (1 - lrf) + lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(pg, lr=learning_rate, momentum=0.9, weight_decay=0.005)

    # Only fine-tune the CNN
    for name, param in model.encoder.resnet50.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = train_CNN

    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    # model.train()
    data, targets = next(iter(train_loader))
    for epoch in range(num_epochs):
        print(f"epoch:{epoch} / {num_epochs}")
        #  取消下一行的注释，查看测试样例
        # print_examples(model, device, train_dataset)
        if save_model and epoch % 1000 == 0 and epoch > 0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint)

        # training procedure
        # adjust_lr(optimizer, epoch)
        model.train()

        # for idx, (imgs, captions) in tqdm(
        #         enumerate(train_loader), total=len(train_loader), leave=False
        # ):

        imgs = data.to(device)
        captions = targets.to(device)
        print(imgs.shape)
        print(captions.shape)
        outputs = model(imgs, captions[:-1])
        loss = criterion(
            outputs.reshape(-1, outputs.shape[2]),
            captions.reshape(-1)
        )

        writer.add_scalar("Training loss", loss.item(), global_step=step)
        writer.add_scalar("Learning rate", optimizer.param_groups[0]["lr"], global_step=step)
        step += 1

        optimizer.zero_grad()
        loss.backward(loss)
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
        optimizer.step()
        print(f"Loss {loss}")

        scheduler.step()

        # validating procedure
        # recent_bleu4 = val_coco(model=model, val_image_folder=val_image_folder, device=device,
        #                         train_dataset=train_dataset, result_i=epoch, transform=transform,
        #                         batch_size=batch_size)
        #
        # is_best = recent_bleu4 > best_bleu4
        # best_bleu4 = max(recent_bleu4, best_bleu4)
        # writer.add_scalar("BLEU_4", recent_bleu4, epoch)
        # writer.add_histogram(tag='fc',
        #                      values=model.encoder.resnet50.fc.weight,
        #                      global_step=epoch)


if __name__ == "__main__":
    train()
