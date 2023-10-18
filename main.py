import os
import argparse
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import FrameDataset
from model.jepa import JEPA, JEPA_XEncoder_Predictor, JEPA_YEncoder
from model.encoder import ViViT, HierarchicalAttentionEncoder, FeedForward, Transformer
from model.vae import VanillaVAE, VAE, PICVAE
from torchvision import transforms
from x_transformers import Decoder, Encoder
import copy
import numpy as np
import torchmetrics

JEPAs = [{
    "model": None,
} for _ in range(11)]

def train(model, train_loader, val_loader, optimizer, scheduler, criterion, args):
    best_acc = 0
    for epoch in range(args.epochs):
        train_loss = 0
        train_acc = 0
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)

            output = model(data)
            
            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_acc += pred.eq(target.view_as(pred)).sum().item()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)
        print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            train_loss, train_acc *
            len(train_loader.dataset), len(train_loader.dataset),
            100. * train_acc))

        val_loss = 0
        val_acc = 0
        model.eval()
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(args.device), target.to(args.device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                val_acc += pred.eq(target.view_as(pred)).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)
        print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            val_loss, val_acc *
            len(val_loader.dataset), len(val_loader.dataset),
            100. * val_acc))

        scheduler.step()
        if val_acc > best_acc:
            best_acc = val_acc
            if args.save_model:
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)
                torch.save(model.state_dict(), os.path.join(
                    args.save_dir, f'model_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.pth'))
                
        print('Best accuracy: {:.0f}%'.format(100. * best_acc))

def unsupervised_train(model, unlabel_loader, optimizer, criterion, scheduler, args, skip=0):
    model.train()
    for epoch in range(args.epochs):
        for batch_idx, data in enumerate(unlabel_loader):
            data = data.to(args.device)
            optimizer.zero_grad()

            # TODO: figure out how to do unsupervised training
            # What is input?
            #   - a batch is a list of videos (each video is a list of frame images)
            #   - each video is a list of frame images
            #   - each frame image is a tensor of shape (3, 160, 240)
            #   - each batch is a tensor of shape (batch_size, num_frames?, 3, 160, 240)
            # What is target?
             # split the frames into x and y
            # x: (b, 11, c, h, w)
            # y: (b, 1, c, h, w)
            data_x = data[:, :11, :, :, :]
            data_y = data[:, 11 + skip, :, :, :].unsqueeze(1)
   
            # print("x: ", x.shape)
            # print("y: ", y.shape)

            # rearrange for encoder (b, num_frames=22, c, h, w) to (b, c, num_frames=22, h, w)
            data_x = data_x.permute(0, 2, 1, 3, 4)
            data_y = data_y.permute(0, 2, 1, 3, 4)

            # encode with previous JEPAs
            encoded_xs = []
            encoded_ys = []
            if not args.train_one:
                with torch.no_grad():
                    for i in range(skip):
                        enc_x = JEPAs[i]["model"].encoder_x(data_x)
                        enc_y = JEPAs[i]["model"].encoder_y(data_y)
                        encoded_xs.append(enc_x)
                        encoded_ys.append(enc_y)

            x, target, latent_loss = model(data_x, data_y, encoded_xs, encoded_ys)

            loss = criterion(x, target) + (args.lamb * latent_loss)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Unsupervised Train Epoch for JEPA #{} : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    skip + 1, epoch, batch_idx * len(data), len(unlabel_loader.dataset),
                    100. * batch_idx / len(unlabel_loader), loss.item()))  
        scheduler.step()
                
    # save model
    if args.save_model:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        torch.save(model.state_dict(), os.path.join(
            args.save_dir, f'pretrain_JEPA_skips{skip}.pth'))

def get_iou(pred, label):
    # intersection over union of segmentation masks
    # to cpu
    _pred = pred.cpu().detach().numpy()
    _label = label.cpu().detach().numpy()
    intersection = np.logical_and(_pred, _label)
    union = np.logical_or(_pred, _label)
    return np.sum(intersection) / np.sum(union)


def finetune_VAE(model, train_loader, val_loader, optimizer, scheduler, args):
    best_acc = 0
    # jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49).to(args.device)
    for epoch in range(args.epochs):
        train_loss = 0
        train_acc = 0
        model.train()
        for batch_idx, data in enumerate(train_loader):
            frames = data
            frames = frames.to(args.device)
            # mask = mask.to(args.device)
            # last frame
            mask = frames[:, -1, :, :, :].float()

            # split 
            data_x = frames[:, :11, :, :, :]
            # data_y = frames[:, 11 + model.skip, :, :, :].unsqueeze(1)
   
            # print("x: ", x.shape)
            # print("y: ", y.shape)

            # rearrange for encoder (b, num_frames=22, c, h, w) to (b, c, num_frames=22, h, w)
            data_x = data_x.permute(0, 2, 1, 3, 4)
            # data_y = data_y.permute(0, 2, 1, 3, 4)

            optimizer.zero_grad()
            pred_frame, mu, logvar, z = model(data_x)
            # pred_frame: ([4, 1, 161, 241])) -> ([4, 160, 240]))
            pred_frame = pred_frame[:, :, :160, :240] #.squeeze(1)

            loss = model.loss_function([pred_frame, mask, mu, logvar])
            t_loss, recon_loss, KLD = loss['loss'], loss['Reconstruction_Loss'], loss['KLD']

            t_loss.backward()
            optimizer.step()

            train_loss += t_loss
            iou = 0#jaccard(pred_frame, mask)
            train_acc += iou
            
            if batch_idx % args.log_interval == 0:
                print('Finetune VAE Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), t_loss.item()))
                print('IoU: {:.6f}, Recon Loss: {:.6f}, KLD: {:.6f}'.format(iou, recon_loss, KLD))
        scheduler.step()
        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)
        print('Finetune VAE Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            train_loss, train_acc *
            len(train_loader.dataset), len(train_loader.dataset),
            100. * train_acc))

        val_loss = 0
        val_acc = 0
        model.eval()
        with torch.no_grad():
            for data in val_loader:
                frames, mask = data
                frames = frames.to(args.device)
                # mask = mask.to(args.device)
                # last frame
                mask = frames[:, -1, :, :, :].float()

                data_x = frames[:, :11, :, :, :]

                data_x = data_x.permute(0, 2, 1, 3, 4)

                pred_frame, mu, logvar, z = model(data_x)

                pred_frame = pred_frame[:, :, :160, :240] #.squeeze(1)
                loss = model.loss_function([pred_frame, mask, mu, logvar])
                t_loss, recon_loss, KLD = loss['loss'], loss['Reconstruction_Loss'], loss['KLD']

                val_loss += t_loss.item()
                iou = 0 # jaccard(pred_frame, mask)
                val_acc += iou
        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)
        print('Finetune VAE Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            val_loss, val_acc *
            len(val_loader.dataset), len(val_loader.dataset),
            100. * val_acc))

        scheduler.step()
        if val_acc > best_acc:
            best_acc = val_acc
            if args.save_model:
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)
                torch.save(model.state_dict(), os.path.join(
                    args.save_dir, f'finetune_VAE_{args.model}_best.pth'))

    if args.save_model:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        torch.save(model.state_dict(), os.path.join(
            args.save_dir, f'finetune_VAE_{args.model}_last.pth'))
                      

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_model', action='store_true', default=True)
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='model')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--finetune', action='store_true', default=False)
    parser.add_argument('--unsupervised', action='store_true', default=False)
    parser.add_argument('--frame_skip', type=int, default=0)
    parser.add_argument('--debug_dataloader', action='store_true', default=False)
    parser.add_argument('--lamb', type=float, default=0.1)
    parser.add_argument('--train_one', action='store_true', default=False)
    parser.add_argument('--list_of_jepas', type=str, default='0_1_2_3_4_5_6_7_8_9_10')
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Set device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # check which device is available
    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("Using CPU")


    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5061, 0.5045, 0.5008], [0.0553, 0.0551, 0.0591])
    ])

    # Load data
    train_loader = DataLoader(
        dataset=FrameDataset(args.data_dir + "/" + 'unlabeled', labeled=False, transform=transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset=FrameDataset(args.data_dir + "/" + 'val', labeled=True, transform=transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.unsupervised:
        ds = FrameDataset(args.data_dir + "/" + 'unlabeled', labeled=False, transform=transform)
        # ds = torch.utils.data.Subset(ds, list(range(0, 100)))
        unlabel_loader = DataLoader(
            dataset=ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    if args.debug_dataloader:
        for batch_idx, data in enumerate(train_loader):
            frames, mask = data
            print(frames.shape)
            print(mask.shape)
            break

    # Layers
    encoder_x = ViViT(
        image_size=(160, 240),
        image_patch_size=(8, 8),
        frames=11,
        frame_patch_size=1,
        num_classes=512,
        dim=512,
        spatial_depth=6,
        temporal_depth=6,
        heads=8,
        mlp_dim=2048,
    )

    encoder_y = ViViT(
        image_size=(160, 240),
        image_patch_size=(8, 8),
        frames=1,
        frame_patch_size=1,
        num_classes=512,
        dim=512,
        spatial_depth=6,
        temporal_depth=6,
        heads=8,
        mlp_dim=2048,
    )

    # encoder predictor module (bs, 512) -> (bs, 512)
    # 
    predictor = nn.Sequential(
        nn.Linear(512, 2048),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(2048, 512),
        nn.Dropout(0.1)
    )

    # predictor = Transformer(
    #     dim=512,
    #     depth=6,
    #     heads=8,
    #     dim_head=64,
    #     mlp_dim=2048,
    #     dropout=0.1,
    # )

    # Load model
    # model = JEPA(img_size=(160, 240), patch_size=(8, 8), in_channels=3,
    #              embed_dim=512, 
    #              encoder_x=encoder_x, 
    #              encoder_y=encoder_y, 
    #              predictor=predictor, 
    #              skip=args.frame_skip
    #              ).to(args.device)

    # # Load optimizer
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # # Load scheduler
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # # Load loss function
    # criterion = nn.MSELoss()

    # for i in range(len(JEPAs)):
    #     m = JEPA(img_size=(160, 240), patch_size=(8, 8), in_channels=3,
    #                 embed_dim=512, 
    #                 encoder_x=copy.deepcopy(encoder_x), 
    #                 encoder_y=copy.deepcopy(encoder_y), 
    #                 predictor=copy.deepcopy(predictor), 
    #                 skip=i
    #                 ).to(args.device)
    #     JEPAs[i]["model"] = m
    #     JEPAs[i]["optimizer"] = optim.SGD(m.parameters(), lr=args.lr, momentum=0.9)
    #     JEPAs[i]["scheduler"] = optim.lr_scheduler.StepLR(JEPAs[i]["optimizer"], step_size=10, gamma=0.1)
    #     JEPAs[i]["criterion"] = nn.MSELoss()

    print("Models loaded")

    # Load pretrained model
    if args.load_model is not None:
        model.load_state_dict(torch.load(args.load_model))

    # Load checkpoint
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']

    # Train model
    if args.unsupervised and not args.train_one:
        # unsupervised_train(model, unlabel_loader, optimizer, criterion, scheduler, args)
        for i in range(len(JEPAs)):
            hsa_x = HierarchicalAttentionEncoder(
                num_encoders=i + 1,
                embed_dim=512,
                hidden_dim=512,
            )
            hsa_y = copy.deepcopy(hsa_x)
            m = JEPA(img_size=(160, 240), patch_size=(8, 8), in_channels=3,
                    embed_dim=512, 
                    encoder_x=copy.deepcopy(encoder_x), 
                    encoder_y=copy.deepcopy(encoder_y), 
                    hsa_x=hsa_x,
                    hsa_y=hsa_y,
                    predictor=copy.deepcopy(predictor), 
                    skip=i
                    ).to(args.device)
            # JEPAs[i]["model"] = m
            optimizer = optim.SGD(m.parameters(), lr=args.lr, momentum=0.9)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
            criterion = nn.MSELoss()  
            print(f"Training JEPA {i}")
            unsupervised_train(m, unlabel_loader, optimizer, criterion, scheduler, args, i)
            print(f"Finished training JEPA {i}")
            JEPAs[i]["model"] = m
            torch.cuda.empty_cache()
    if args.train_one and args.unsupervised:
        hsa_x = HierarchicalAttentionEncoder(
            num_encoders=1,
            embed_dim=512,
            hidden_dim=512,
        )
        hsa_y = copy.deepcopy(hsa_x)
        m = JEPA(img_size=(160, 240), patch_size=(8, 8), in_channels=3,
                embed_dim=512, 
                encoder_x=encoder_x, 
                encoder_y=encoder_y,
                hsa_x=hsa_x,
                hsa_y=hsa_y,
                predictor=predictor, 
                skip=args.frame_skip
                ).to(args.device)
        # JEPAs[i]["model"] = m
        optimizer = optim.SGD(m.parameters(), lr=args.lr, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        criterion = nn.MSELoss()  
        print(f"Training JEPA")
        unsupervised_train(m, unlabel_loader, optimizer, criterion, scheduler, args, args.frame_skip)
        print(f"Finished training JEPA")
        # JEPAs[0]["model"] = m
        torch.cuda.empty_cache()
    
    elif args.finetune:

        list_of_jepas = args.list_of_jepas.split("_")

        JEPAs = [
            {
                "model": None,
            }
            for i in range(len(list_of_jepas))
        ]

        # Load saved JEPAs
        for idx, i in enumerate(list_of_jepas):
            hsa_x = HierarchicalAttentionEncoder(
                num_encoders=idx + 1,
                embed_dim=512,
                hidden_dim=512,
            )
            hsa_y = copy.deepcopy(hsa_x)
            m = JEPA(img_size=(160, 240), patch_size=(8, 8), in_channels=3,
                    embed_dim=512, 
                    encoder_x=copy.deepcopy(encoder_x), 
                    encoder_y=copy.deepcopy(encoder_y), 
                    hsa_x=hsa_x,
                    hsa_y=hsa_y,
                    predictor=copy.deepcopy(predictor), 
                    skip=int(i)
                    ).to(args.device)
            optimizer = optim.SGD(m.parameters(), lr=args.lr, momentum=0.9)
            JEPAs[idx]["model"] = m
            JEPAs[idx]["model"].load_state_dict(torch.load(args.save_dir + f"/pretrain_JEPA_skips{i}.pth"))
            print(f"Loaded JEPA {i}")

        encoders_x = []
        encoders_y = []
        for i in range(len(JEPAs)):
            encoders_x.append(JEPAs[i]["model"].encoder_x)
            encoders_y.append(JEPAs[i]["model"].encoder_y)

        jepa_encoder = JEPA_XEncoder_Predictor(
            embed_dim=512,
            encoders_x=encoders_x,
            predictor=JEPAs[-1]["model"].predictor,
            hsa_x=JEPAs[-1]["model"].hsa_x,
        )
#conv_layers, z_dimension, pool_kernel_size,
                 # conv_kernel_size, input_channels, height, width, hidden_dim, use_cuda
        # model = VAE(
        #     conv_layers=3,
        #     z_dimension=512,
        #     pool_kernel_size=0,
        #     conv_kernel_size=4,
        #     input_channels=3,
        #     encoder=jepa_encoder,
        #     height=160,
        #     width=240,
        #     hidden_dim=512,
        #     use_cuda=True,
        # ).to(args.device)

        model = PICVAE(
            conv_layers=3,
            z_dimension=512,
            pool_kernel_size=0,
            conv_kernel_size=4,
            input_channels=3,
            encoder=jepa_encoder,
            height=160,
            width=240,
            hidden_dim=512,
            use_cuda=True,
        ).to(args.device)

        learning_rate = 1e-3

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        finetune_VAE(model, train_loader, val_loader, optimizer, scheduler, args)

    # Save model
    if args.save_model and not args.unsupervised:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        torch.save(model.state_dict(), os.path.join(
            args.save_dir, f'model_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pth'))

# Train unsupervised command
# python <path-to>/main.py --batch_size=<bs> --epochs=<epochs> --lr=<lr> --unsupervised --data_dir=<path-to-data> --save_dir=<path-to-save-model>
