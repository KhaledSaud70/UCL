import datetime
import math
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import argparse
from pathlib import Path
import data
import models
import losses
import time
import wandb
import torch.utils.tensorboard

from torchvision import transforms
from util import AverageMeter, NViewTransform, accuracy, ensure_dir, set_seed, arg2bool, warmup_learning_rate, save_on_master
from lars import LARS

def parse_arguments():
    parser = argparse.ArgumentParser(description="Product Recognition Training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--device', type=str, help='torch device', default='cuda')
    parser.add_argument('--print_freq', type=int, help='print frequency', default=10)
    parser.add_argument('--trial', type=int, help='random seed / trial id', default=0)
    parser.add_argument('--log_dir', type=str, help='tensorboard log dir', default='logs')
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--save_every', default=None, type=int, help='save model every epochs')

    parser.add_argument('--data_dir', type=str, help='path of data dir', required=True, default='/data')
    parser.add_argument('--dataset', type=str, help='dataset (format name_attr e.g. biased-mnist_0.999)', required=True)
    parser.add_argument('--batch_size', type=int, help='batch size', default=256)
    parser.add_argument('--test_percent', type=float, help='percentage of data to be used for testing', default=0.1)
    parser.add_argument('--full_training', help='Perform training on full data only (no testing)', default=False)

    parser.add_argument('--epochs', type=int, help='number of epochs', default=100)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.01)
    parser.add_argument('--warm', type=arg2bool, help='warmup lr', default=False)
    parser.add_argument('--lr_decay', type=str, help='type of decay', choices=['cosine', 'step', 'none'], default='cosine')
    parser.add_argument('--lr_decay_epochs', type=str, help='steps of lr decay (list)', default="100,150")
    parser.add_argument('--optimizer', type=str, help="optimizer (adam, sgd or lars)", choices=["adam", "sgd", "lars"], default="sgd")
    parser.add_argument('--momentum', type=float, help='momentum', default=0.9)
    parser.add_argument('--weight_decay', type=float, help='weight decay', default=1e-4)

    parser.add_argument('--model', type=str, help='model architecture')

    parser.add_argument('--method', type=str, help='loss function', choices=['infonce', 'infonce-strong'], default='infonce')
    parser.add_argument('--n_views', type=int, help='number of different views', default=1)
    parser.add_argument('--selfsup', type=arg2bool, help='do not use labels', default=False)
    parser.add_argument('--augplus', type=arg2bool, help='use simclr aug (for selfsup)', default=False)

    parser.add_argument('--form', type=str, help='loss form (in or out)', default='out')
    parser.add_argument('--temp', type=float, help='supcon/infonce temperature', default=0.1)
    parser.add_argument('--epsilon', type=float, help='infonce epsilon', default=0.)
    parser.add_argument('--lr_epsilon', type=float, help='epsilon lr', default=1e-4)
    
    parser.add_argument('--lambd', type=float, help='lagrangian weight for debiasing', default=0.)
    parser.add_argument('--lambd_alpha_ratio', type=float, help='compute lambd as alpha*ratio (0=disabled)', default=0)
    parser.add_argument('--kld', type=float, help='weight of std term', default=1.)

    parser.add_argument('--alpha', type=float, help='infonce weight', default=1.)
    parser.add_argument('--alpha_rand', action='store_true', help='sample alpha randomly in [0,1]')

    parser.add_argument('--beta', type=float, help='cross-entropy weight WITH supcon', default=0)
    
    parser.add_argument('--feat_dim', type=int, help='size of projection head', default=128)
    parser.add_argument('--mlp_lr', type=float, help='mlp lr', default=0.001)
    parser.add_argument('--mlp_lr_decay', type=str, help='mlp lr decay', default='constant')
    parser.add_argument('--mlp_max_iter', type=int, help='mlp training epochs', default=500)
    parser.add_argument('--mlp_optimizer', type=str, help='mlp optimizer', default='adam')
    parser.add_argument('--mlp_batch_size', type=int, help='mlp batch size', default=None)
    parser.add_argument('--test_freq', type=int, help='test frequency', default=1)
    parser.add_argument('--train_on_head', type=arg2bool, help="train clf on projection head features", default=True)

    parser.add_argument('--amp', action='store_true', help='use amp')

    cfg = parser.parse_args()
    if cfg.alpha_rand:
        cfg.alpha = random.random()
        print("Sampling random alpha", cfg.alpha)

    if cfg.lambd_alpha_ratio > 0:
        cfg.lambd = cfg.lambd_alpha_ratio * cfg.alpha
        print("lambda/alpha ratio ->", cfg.lambd_alpha_ratio, "*", cfg.alpha, "=", cfg.lambd) 

    if cfg.selfsup and cfg.n_views == 1:
        print("n_views must be > 1 if selfsup is true")
        exit(1)

    return cfg

def load_data(cfg):
    if cfg.dataset != 'danube':
        ValueError(f"Can\'t build {cfg.dataset}")

    if cfg.dataset == 'danube':
        mean = (0.5583, 0.4707, 0.4320)
        std = (0.2333, 0.2389, 0.2386)

    if cfg.dataset == 'danube':
        resize_size = 224
        T_train = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        if cfg.augplus:
            T_train = transforms.Compose([
                transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
                    
        T_test = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        if cfg.augplus:
            T_test = transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

    if hasattr(cfg, 'n_views'):
        T_train = NViewTransform(T_train, cfg.n_views)


    if cfg.dataset == 'danube':
        if cfg.full_training:
            cfg.test_percent = 0.0

        train_dataset = data.DanubeDataset(data_dir=cfg.data_dir, transform=T_train)
        cfg.n_classes = train_dataset.num_classes
        train_dataset = data.MapDataset(train_dataset, lambda x, y: (x, y, 0))
        print(cfg.n_classes)

        if cfg.test_percent > 0.0:
            test_dataset = data.DanubeDataset(data_dir=cfg.data_dir, transform=T_test)
            test_dataset = data.MapDataset(test_dataset, lambda x, y: (x, y, 0))

            data_size = len(train_dataset)
            indices = list(range(data_size))
            np.random.shuffle(indices)
            split = int(np.floor(cfg.test_percent * data_size))
            train_idx, valid_idx = indices[split:], indices[:split]

            train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
            test_dataset = torch.utils.data.Subset(test_dataset, valid_idx)
            print(len(test_dataset), 'test images')     

        print(len(train_dataset), 'training images')
        
    
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                               num_workers=8, persistent_workers=True)
    if not cfg.full_training:
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8, 
                                                  persistent_workers=True)
        return train_loader, test_loader

    return train_loader

def load_model(cfg):
    if 'resnet' in cfg.model:
        model = models.SupConResNet(cfg.model, feat_dim=cfg.feat_dim,
                                    num_classes=cfg.n_classes,
                                    train_on_head=cfg.train_on_head)
        
    else:
        ValueError(f'Unsupported model name {cfg.model}')

    cfg.feat_dim = model.feat_dim

    if cfg.device == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using multiple CUDA devices ({torch.cuda.device_count()})")
        model = torch.nn.DataParallel(model)
    model = model.to(cfg.device)

    if "infonce" in cfg.method:
        if "strong" in cfg.method:
            criterion = losses.EpsilonSupCon(temperature=cfg.temp, form=cfg.form, epsilon=cfg.epsilon)
        else:
            criterion = losses.EpsilonSupInfoNCE(temperature=cfg.temp, form=cfg.form, epsilon=cfg.epsilon)

    else:
        raise ValueError('Unsupported loss function', cfg.method)

    criterion = criterion.to(cfg.device)
    
    return model, criterion

def load_optimizer(model, criterion, cfg):
    if torch.cuda.device_count() > 1:
        parameters = [{'params': model.module.encoder.parameters()},
                    {'params': model.module.head.parameters()}]
    else:
        parameters = [{'params': model.encoder.parameters()},
                    {'params': model.head.parameters()}]

    if cfg.beta > 0:
        parameters = model.parameters()

    if "auto" in cfg.method:
        parameters.append({'params': criterion.epsilon,
                           'lr': cfg.lr_epsilon,
                           'weight_decay': 0})

    if cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=cfg.lr, 
                                    momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adam":
        optimizer = torch.optim.Adam(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)

    else:
        optimizer = LARS(parameters, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        
    if cfg.lr_decay == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, 
                                                               verbose=True)
    elif cfg.lr_decay == 'step':
        milestones = [int(s) for s in cfg.lr_decay_epochs.split(',')]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, verbose=True)
    
    elif cfg.lr_decay == 'none':
        scheduler = None

    optimizer_fc, scheduler_fc = None, None

    if cfg.beta == 0:
        fc_params = model.fc.parameters() if torch.cuda.device_count() <= 1 else model.module.fc.parameters()
        if cfg.mlp_optimizer == "sgd":
            optimizer_fc = torch.optim.SGD(fc_params, lr=cfg.mlp_lr, momentum=0.9,
                                        weight_decay=0)
        elif cfg.mlp_optimizer == "adam":
            optimizer_fc = torch.optim.Adam(fc_params, lr=cfg.mlp_lr,
                                            weight_decay=0)
        
        if cfg.mlp_lr_decay == 'cosine':
            scheduler_fc = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_fc, T_max=cfg.epochs, 
                                                                    verbose=True)
        else:
            scheduler_fc = None

    print((optimizer, scheduler), (optimizer_fc, scheduler_fc))
    return (optimizer, scheduler), (optimizer_fc, scheduler_fc)

def train(train_loader, model, criterion, optimizers, cfg, epoch, scaler=None):
    loss = AverageMeter()
    nce = AverageMeter()
    ce = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    optimizer, optimizer_fc = optimizers

    all_outputs, all_labels = [], []

    t1 = time.time()
    for idx, (images, labels, bias_labels) in enumerate(train_loader):
        data_time.update(time.time() - t1)

        images = torch.cat(images, dim=0)
        images, labels, bias_labels = images.to(cfg.device), labels.to(cfg.device), bias_labels.to(cfg.device)
        bsz = labels.shape[0]
        
        warmup_learning_rate(cfg, epoch, idx, len(train_loader), optimizer)

        with torch.set_grad_enabled(True):
            with torch.cuda.amp.autocast(scaler is not None):
                projected, feats, logits = model(images)
                
                projected = torch.split(projected, [bsz]*cfg.n_views, dim=0)
                projected = torch.cat([f.unsqueeze(1) for f in projected], dim=1)

                feats = torch.split(feats, [bsz]*cfg.n_views, dim=0)
                feats = torch.cat([f.unsqueeze(1) for f in feats], dim=1)

                logits = torch.split(logits, [bsz]*cfg.n_views, dim=0)
                logits = torch.cat([f.unsqueeze(1) for f in logits], dim=1)

                running_nce = criterion(projected, feats[:, 0], logits, labels, bias_labels)
                running_ce = F.cross_entropy(logits[:, 0], labels)

                running_loss = running_nce
                if cfg.beta > 0:
                    running_loss = running_nce + cfg.beta*running_ce
          
        optimizer.zero_grad()

        if optimizer_fc is not None:
            optimizer_fc.zero_grad()

        if scaler is None:
            if optimizer_fc is not None:
                running_ce.backward(retain_graph=True) # Backward cross-entropy from last layer
                optimizer_fc.step()
                optimizer.zero_grad() # Stop-gradient on the encoder

            running_loss.backward() # Backward infonce loss on the encoder
            optimizer.step()
        else:
            if optimizer_fc is not None:
                scaler.scale(running_ce).backward(retain_graph=True)
                scaler.step(optimizer_fc)
                optimizer.zero_grad()

            scaler.scale(running_loss).backward()
            scaler.step(optimizer)
            
            scaler.update()
        
        loss.update(running_loss.item(), bsz)
        nce.update(running_nce.item(), bsz)
        ce.update(running_ce.item(), bsz)
        batch_time.update(time.time() - t1)
        t1 = time.time()
        eta = batch_time.avg * (len(train_loader) - idx)

        if (idx + 1) % cfg.print_freq == 0:
            print(f"Train: [{epoch}][{idx + 1}/{len(train_loader)}]:\t"
                  f"BT {batch_time.avg:.3f}\t"
                  f"ETA {datetime.timedelta(seconds=eta)}\t"
                  f"NCE {nce.avg:.3f}\t"
                  f"CE {ce.avg:.3f}\t"
                  f"loss {loss.avg:.3f}\t")
        
        all_outputs.append(logits[:, 0].detach())
        all_labels.append(labels)
    
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    accuracy_train = accuracy(all_outputs, all_labels)[0]

    return loss.avg, accuracy_train, batch_time.avg, data_time.avg

def measure_similarity(feat, labels, bias_labels):
    bsz = feat.shape[0]

    if labels.shape[0] != bsz:
        raise ValueError('Num of labels does not match num of features')
    if bias_labels.shape[0] != bsz:
        raise ValueError('Num of bias_labels does not match num of features')

    similarity = torch.matmul(feat, feat.T)

    labels = labels.view(-1, 1)
    positive_mask = torch.eq(labels, labels.T)
    negative_mask = ~positive_mask

    bias_labels = bias_labels.view(-1, 1)
    aligned_mask = torch.eq(bias_labels, bias_labels.T)
    conflicting_mask = ~aligned_mask

    pos_aligned = positive_mask * aligned_mask
    aligned_sim = similarity * pos_aligned
    aligned_sim_mean = (aligned_sim.sum(dim=1) - 1) / (torch.count_nonzero(aligned_sim, dim=1)+1)

    pos_conflicting = positive_mask * conflicting_mask
    conflicting_sim = similarity * pos_conflicting
    conflicting_sim_mean = conflicting_sim.sum(dim=1) / (torch.count_nonzero(conflicting_sim, dim=1)+1)

    # print(conflicting_sim.sum(dim=1), torch.count_nonzero(conflicting_sim, dim=1))

    neg_aligned = negative_mask * aligned_mask
    negative_aligned_sim = similarity * neg_aligned
    negative_aligned_sim_mean = (negative_aligned_sim.sum(dim=1)) / (torch.count_nonzero(neg_aligned, dim=1)+1)
    
    neg_conflicting = negative_mask * conflicting_mask
    negative_conflicting_sim = similarity * neg_aligned
    negative_conflicting_sim_mean = (negative_conflicting_sim.sum(dim=1)) / (torch.count_nonzero(neg_conflicting, dim=1)+1)

    # print(f"pos-aligned: {pos_aligned.sum() - bsz}, pos-conflicting: {pos_conflicting.sum() - bsz}")
    return (aligned_sim[torch.nonzero(pos_aligned, as_tuple=True)], aligned_sim_mean.mean()), \
           (conflicting_sim[torch.nonzero(pos_conflicting, as_tuple=True)], conflicting_sim_mean.mean()), \
           (negative_aligned_sim[torch.nonzero(neg_aligned, as_tuple=True)], negative_aligned_sim_mean.mean()), \
           (negative_conflicting_sim[torch.nonzero(neg_conflicting, as_tuple=True)], negative_conflicting_sim_mean.mean())
           
def test(test_loader, model, criterion, cfg):
    model.eval()

    loss = AverageMeter()
    all_outputs, all_labels = [], []

    aligned_sim, conflicting_sim, negative_aligned_sim, negative_conflicting_sim = [], [], [], []
    aligned_similarity = AverageMeter()
    conflicting_similarity = AverageMeter()
    negative_aligned_similarity = AverageMeter()
    negative_conflicting_similarity = AverageMeter()

    for images, labels, bias_labels in test_loader:
        images, labels = images.to(cfg.device), labels.to(cfg.device)
        bias_labels = bias_labels.to(cfg.device)

        with torch.no_grad():
            projected, feats, logits = model(images)
            running_loss = criterion(projected[:, None], feats, logits, labels, bias_labels)
            (al_sim, al_sim_mean), (con_sim, con_sim_mean), \
            (neg_al_sim, neg_al_sim_mean), (neg_confl_sim, neg_confl_sim_mean) \
                = measure_similarity(projected, labels, bias_labels)
        
        loss.update(running_loss.item(), images.shape[0])
        
        all_outputs.append(logits.detach())
        all_labels.append(labels)

        if not torch.isinf(al_sim_mean):
            aligned_similarity.update(al_sim_mean.item(), images.shape[0])
            aligned_sim.append(al_sim.view(-1))

        if not torch.isinf(con_sim_mean):
            conflicting_similarity.update(con_sim_mean.item(), images.shape[0])
            conflicting_sim.append(con_sim.view(-1))
        
        if not torch.isinf(neg_al_sim_mean):
            negative_aligned_similarity.update(neg_al_sim_mean.item(), images.shape[0])
            negative_aligned_sim.append(neg_al_sim.view(-1))

        if not torch.isinf(neg_confl_sim_mean):
            negative_conflicting_similarity.update(neg_confl_sim_mean.item(), images.shape[0])
            negative_conflicting_sim.append(neg_confl_sim.view(-1))

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    aligned_sim = torch.cat(aligned_sim)
    conflicting_sim = torch.cat(conflicting_sim)
    negative_aligned_sim = torch.cat(negative_aligned_sim)
    negative_conflicting_sim = torch.cat(negative_conflicting_sim)

    accuracy_test = accuracy(all_outputs, all_labels)[0]
    print(f'FC test accuracy: {accuracy_test:.2f}')

    return loss.avg, accuracy_test, (aligned_sim, aligned_similarity.avg), \
           (conflicting_sim, conflicting_similarity.avg), \
           (negative_aligned_sim, negative_aligned_similarity.avg), \
           (negative_conflicting_sim, negative_conflicting_similarity.avg)
        
if __name__ == '__main__':
    cfg = parse_arguments()
    set_seed(cfg.trial)

    if cfg.full_training:
        train_loader = load_data(cfg)
    else:
        train_loader, test_loader = load_data(cfg)

    model, infonce = load_model(cfg)
    (optimizer, scheduler), (optimizer_fc, scheduler_fc) = load_optimizer(model, infonce, cfg)
    
    if cfg.batch_size > 256:
        cfg.warm = True
    
    if cfg.warm:
        cfg.warm_epochs = 10
        cfg.warmup_from = 0.01
        cfg.model = f"{cfg.model}_warm"
        
        if cfg.lr_decay == 'cosine':
            eta_min = cfg.lr * (0.1 ** 3)
            cfg.warmup_to = eta_min + (cfg.lr - eta_min) * (1 + math.cos(math.pi * cfg.warm_epochs / cfg.epochs)) / 2
        else:
            cfg.warmup_to = cfg.lr

    ensure_dir(cfg.log_dir)
    method = cfg.method
    if cfg.selfsup:
        method = f"{method}_self"
    if cfg.augplus:
        method = f"{method}_aug+"

    run_name = (f"{method}_{cfg.form}_{cfg.dataset}_{cfg.model}_"
                f"{cfg.optimizer}_bsz{cfg.batch_size}_"
                f"lr{cfg.lr}_{cfg.lr_decay}_t{cfg.temp}_eps{cfg.epsilon}_"
                f"lr-eps{cfg.lr_epsilon}_feat{cfg.feat_dim}_"
                f"{'identity_' if cfg.train_on_head else 'head_'}"
                f"alpha{cfg.alpha}_beta{cfg.beta}_lambda{cfg.lambd}_kld{cfg.kld}_" # remove {cfg.dist}_
                f"mlp_lr{cfg.mlp_lr}_mlp_optimizer_{cfg.mlp_optimizer}_"
                f"trial{cfg.trial}")
    tb_dir = os.path.join(cfg.log_dir, run_name)
    cfg.model_class = model.__class__.__name__
    cfg.criterion = infonce
    cfg.optimizer_class = optimizer.__class__.__name__
    cfg.scheduler = scheduler.__class__.__name__ if scheduler is not None else None

    wandb.init(project="product-recognition", config=cfg, name=run_name, sync_tensorboard=True)
    print('Config:', cfg)
    print('Model:', model)
    print('Criterion:', infonce)
    print('Optimizer:', optimizer)
    print('Scheduler:', scheduler)

    writer = torch.utils.tensorboard.writer.SummaryWriter(tb_dir)
    
    def target_loss(projected, labels):
        if cfg.selfsup:
            return cfg.alpha*infonce(projected)
        return cfg.alpha*infonce(projected, labels)
    criterion = lambda projected, feats, logits, labels, bias_labels: target_loss(projected, labels)
            
    if cfg.lambd != 0:
        print("Applying regularization")
        
        def infonce_fairkl(projected, feats, logits, labels, bias_labels):
            feats = F.normalize(feats)
            return cfg.alpha * target_loss(projected, labels) + \
                   cfg.lambd * losses.fairkl(feats, labels, bias_labels, 1.0, kld=cfg.kld)
        
        criterion = infonce_fairkl

    scaler = torch.cuda.amp.GradScaler() if cfg.amp else None
    if cfg.amp:
        print("Using AMP")
    
    output_dir = Path(cfg.output_dir)
    torch.save(cfg, output_dir / "args.pyT")
    
    start_time = time.time()
    best_acc = 0.
    for epoch in range(1, cfg.epochs + 1):
        t1 = time.time()
        loss_train, accuracy_train, batch_time, data_time = train(train_loader, model, criterion, (optimizer, optimizer_fc), cfg, epoch, scaler)
        t2 = time.time()

        writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar("train/loss", loss_train, epoch)
        writer.add_scalar("train/acc@1", accuracy_train, epoch)
        if "auto" in cfg.method:
            writer.add_scalar("train/epsilon", infonce.epsilon, epoch)

        writer.add_scalar("BT", batch_time, epoch)
        writer.add_scalar("DT", data_time, epoch)
        print(f"epoch {epoch}, total time {t2-start_time:.2f}, epoch time {t2-t1:.3f} "
              f"acc {accuracy_train:.2f} loss {loss_train:.4f}")
        
        if scheduler is not None:
            scheduler.step()
        
        if cfg.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if cfg.save_every is not None:
                if epoch % cfg.save_every == 0: checkpoint_paths.append(output_dir / 'checkpoint_{}.pth'.format(epoch))
            for checkpoint_path in checkpoint_paths:
                save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'cfg': cfg,
                }, checkpoint_path)

        if not cfg.full_training and (epoch % cfg.test_freq == 0) or epoch == 1 or epoch == cfg.epochs:
            loss_test, accuracy_test, aligned_sim, conflicting_sim, \
            negative_aligned_sim, negative_conflicting_sim \
                 = test(test_loader, model, criterion, cfg)
            
            writer.add_scalar("test/loss", loss_test, epoch)
            writer.add_scalar("test/acc@1", accuracy_test, epoch)
            print(f"test accuracy {accuracy_test:.2f}")

            print(f"""pos-aligned sim {aligned_sim[1]:.4f}, pos-conflict sim {conflicting_sim[1]:.4f}, """
                  f"""neg-aligned sim {negative_aligned_sim[1]:.4f} neg-conflict sim {negative_conflicting_sim[1]:.4f}""")
            writer.add_scalar('test/aligned_sim_mean', aligned_sim[1], epoch)
            writer.add_scalar('test/conflicting_sim_mean', conflicting_sim[1], epoch)
            writer.add_scalar('test/negative_aligned_sim_mean', negative_aligned_sim[1])
            writer.add_scalar('test/negative_conflicting_sim_mean', negative_conflicting_sim[1])

            try:
                writer.add_histogram('test/aligned_sim', aligned_sim[0], epoch, bins=256, max_bins=512)
                writer.add_histogram('test/conflicting_sim', conflicting_sim[0], epoch, bins=256, max_bins=512)
                writer.add_histogram('test/negative_aligned_sim', negative_aligned_sim[0], epoch, bins=256, max_bins=512)
                writer.add_histogram('test/negative_conflicting_sim', negative_conflicting_sim[0], epoch, bins=256, max_bins=512)
            except:
                pass

            if accuracy_test > best_acc:
                best_acc = accuracy_test

        if not cfg.full_training:
            writer.add_scalar("best_acc@1", best_acc, epoch)
    
    if not cfg.full_training:
        print(f"best accuracy: {best_acc:.2f}")