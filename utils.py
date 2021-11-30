from torch.utils.data import sampler
import torchvision.datasets as dset
import numpy as np
import torch
from torch import nn
import torchvision.transforms as transforms
from loss import *
from tqdm import tqdm
import os
import torchvision.transforms.functional as tvf
import random
import torch.nn.functional as F

def rgb_to_hsv(input, device):
    input = input.transpose(1, 3)
    sh = input.shape
    input = input.reshape(-1, 3)
 
    mx, inmx = torch.max(input, dim=1)
    mn, inmc = torch.min(input, dim=1)
    df = mx - mn
    h = torch.zeros(input.shape[0], 1).to(device)
    # if False: #'xla' not in device.type:
    #     h.to(device)
    ii = [0, 1, 2]
    iid = [[1, 2], [2, 0], [0, 1]]
    shift = [360, 120, 240]
 
    for i, id, s in zip(ii, iid, shift):
        logi = (df != 0) & (inmx == i)
        h[logi, 0] = \
            torch.remainder((60 * (input[logi, id[0]] - input[logi, id[1]]) / df[logi] + s), 360)
 
    s = torch.zeros(input.shape[0], 1).to(device) #
    # if False: #'xla' not in device.type:
    #     s.to(device)
    s[mx != 0, 0] = (df[mx != 0] / mx[mx != 0]) * 100
 
    v = mx.reshape(input.shape[0], 1) * 100
 
    output = torch.cat((h / 360., s / 100., v / 100.), dim=1)
 
    output = output.reshape(sh).transpose(1, 3)
    return output

def hsv_to_rgb(input, device):
    input = input.transpose(1, 3)
    sh = input.shape
    input = input.reshape(-1, 3)
 
    hh = input[:, 0]
    hh = hh * 6
    ihh = torch.floor(hh).type(torch.int32)
    ff = (hh - ihh)[:, None];
    v = input[:, 2][:, None]
    s = input[:, 1][:, None]
    p = v * (1.0 - s)
    q = v * (1.0 - (s * ff))
    t = v * (1.0 - (s * (1.0 - ff)));
 
    output = torch.zeros_like(input).to(device) #.to(device)
    # if False: #'xla' not in device.type:
    #     output.to(device)
    output[ihh == 0, :] = torch.cat((v[ihh == 0], t[ihh == 0], p[ihh == 0]), dim=1)
    output[ihh == 1, :] = torch.cat((q[ihh == 1], v[ihh == 1], p[ihh == 1]), dim=1)
    output[ihh == 2, :] = torch.cat((p[ihh == 2], v[ihh == 2], t[ihh == 2]), dim=1)
    output[ihh == 3, :] = torch.cat((p[ihh == 3], q[ihh == 3], v[ihh == 3]), dim=1)
    output[ihh == 4, :] = torch.cat((t[ihh == 4], p[ihh == 4], v[ihh == 4]), dim=1)
    output[ihh == 5, :] = torch.cat((v[ihh == 5], p[ihh == 5], q[ihh == 5]), dim=1)
 
    output = output.reshape(sh)
    output = output.transpose(1, 3)
    return output

def deform_data(x_in, perturb, trans, s_factor, h_factor, embedd, device):
    h=x_in.shape[2]
    w=x_in.shape[3]
    nn=x_in.shape[0]
    v=((torch.rand(nn, 6) - .5) * perturb).to(device)
    rr = torch.zeros(nn, 6).to(device)
    if not embedd:
        ii = torch.randperm(nn)
        u = torch.zeros(nn, 6).to(device)
        u[ii[0:nn//2]]=v[ii[0:nn//2]]
    else:
        u=v
    # Ammplify the shift part of the
    u[:,[2,5]]*=2
    rr[:, [0,4]] = 1
    if trans=='shift':
      u[:,[0,1,3,4]]=0
    elif trans=='scale':
      u[:,[1,3]]=0
    elif 'rotate' in trans:
      u[:,[0,1,3,4]]*=1.5
      ang=u[:,0]
      v=torch.zeros(nn,6)
      v[:,0]=torch.cos(ang)
      v[:,1]=-torch.sin(ang)
      v[:,4]=torch.cos(ang)
      v[:,3]=torch.sin(ang)
      s=torch.ones(nn)
      if 'scale' in trans:
        s = torch.exp(u[:, 1])
      u[:,[0,1,3,4]]=v[:,[0,1,3,4]]*s.reshape(-1,1).expand(nn,4)
      rr[:,[0,4]]=0
    theta = (u+rr).view(-1, 2, 3)
    grid = F.affine_grid(theta, [nn,1,h,w],align_corners=True)
    x_out=F.grid_sample(x_in,grid,padding_mode='zeros',align_corners=True)

    if x_in.shape[1]==3 and s_factor>0:
        v=torch.rand(nn,2).to(device)
        vv=torch.pow(2,(v[:,0]*s_factor-s_factor/2)).reshape(nn,1,1)
        uu=((v[:,1]-.5)*h_factor).reshape(nn,1,1)
        x_out_hsv=rgb_to_hsv(x_out, device)
        x_out_hsv[:,1,:,:]=torch.clamp(x_out_hsv[:,1,:,:]*vv,0.,1.)
        x_out_hsv[:,0,:,:]=torch.remainder(x_out_hsv[:,0,:,:]+uu,1.)
        x_out=hsv_to_rgb(x_out_hsv, device)

    ii=torch.where(torch.bernoulli(torch.ones(nn)*.5)==1)
    for i in ii:
          x_out[i]=x_out[i].flip(3)
    return x_out

def deform_gaze(x):
    n = x.shape[2]
    x1 = torch.cat((x[:,:,:n//2,:n//2], x[:,:,:n//2,n//2:]), dim=0)
    x2 = torch.cat((x[:,:,n//2:,:n//2], x[:,:,n//2:,n//2:]), dim=0)
    return x1, x2

def deform_gaze2(x, pars):
    bsz = x.size(0)
    patch_size = pars.patch_size
    x_unfold = F.unfold(x, kernel_size=patch_size, stride=patch_size//2) # (bsz, 256, 49)
    all_patches = x_unfold.permute(0,2,1).reshape(bsz*x_unfold.shape[-1], patch_size, patch_size) # (bsz*49, 16, 16)
    output = all_patches.unsqueeze(dim=1) # bsz*49, 1, 16, 16
    return output

def random_rotate(image):
   if random.random() > 0.5:
       return tvf.rotate(image, angle=random.choice((0, 90, 180, 270)))
   return image

def get_scripted_transforms(s=1.0):
    tf = torch.nn.Sequential(
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomRotation(90),
        transforms.RandomApply(torch.nn.ModuleList([
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)
        ]),p=0.8)
    )
    scripted_transforms = torch.jit.script(tf)
    return scripted_transforms

def train_model(train_loader, test_loader, fix, model, pars, ep_loss, ep_acc, expdir, current_layer=-1):
    """
    The training function
    """
    device = pars.device
    dtype = torch.float32

    fix = fix.to(device=device)
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    print(fix)
    print(model)

    if pars.train_unsupervised:
        lr = pars.LR
        opt = pars.OPT
        # select self-supervised losses
        if pars.loss == 'Hinge':
            criterion = ContrastiveHinge(pars.batch_size, pars.thr1, pars.thr2, device=pars.device)
        elif pars.loss == 'HingeNN':
            criterion = ContrastiveHingeNN(pars.batch_size, pars.thr1, pars.thr2, pars.grad_block, device=pars.device)
        elif pars.loss == 'HingeNN2':
            criterion = ContrastiveHingeNN2(pars.batch_size, pars.thr1, pars.thr2, pars.grad_block, device=pars.device)
        elif pars.loss == 'HingeNNFewerNegs':
            criterion = HingeNNFewerNegs(pars.batch_size, pars.thr1, pars.thr2, pars.n_negs, pars.grad_block, device=pars.device)
        elif pars.loss == 'GazeHingeNN':
            criterion = GazeHingeNN(pars)
        elif pars.loss =='CLAPP':
            n_features = model[0].weight.shape[0] if pars.process != 'E2E' else 1024
            criterion = CLAPPHinge(pars, n_features)
        else:
            criterion = SimCLRLoss(pars.batch_size, pars.device)

        if pars.process == 'E2E':
            if pars.loss == 'CLAPP':
                params = list(fix.parameters())+list(model.parameters())+list(criterion.parameters())
            else:
                params = list(fix.parameters())+list(model.parameters())
        else:
            if pars.loss == 'CLAPP':
                params = list(model.parameters())+list(criterion.parameters())
            else:
                params = model.parameters()
    else:
        if pars.unsupervised:
            lr = pars.clf_lr
            loss = pars.clf_loss
            opt = pars.clf_opt
            params = model.parameters()
        else:
            lr = pars.LR
            loss = pars.loss
            opt = pars.OPT
            if pars.process == 'E2E':
                params = list(fix.parameters())+list(model.parameters())
            else:
                params = model.parameters()

        criterion = torch.nn.CrossEntropyLoss()
    print(criterion)
       
    if opt == 'SGD':
        optimizer = torch.optim.SGD(params, lr=lr)
    else:
        optimizer = torch.optim.Adam(params, lr=lr)

    if pars.loadnet and pars.train_unsupervised:
        checkpoint = torch.load(pars.loadnet)
        optimizer.load_state_dict(checkpoint['optimizer'])
    print(optimizer)
    
    start_epoch = 0
    if (pars.unsupervised) and (not pars.train_unsupervised):
        n_epochs = pars.clf_epochs
    else:
        n_epochs = pars.epochs
        if pars.loadnet:
            checkpoint = torch.load(pars.loadnet)
            start_epoch = checkpoint['epoch']

    for e in range(start_epoch, n_epochs):
        running_loss = 0
        bsz_multiplier = 49 if pars.gaze_shift else 2
        num_train = min(pars.num_train, len(train_loader.dataset))
        total_n = bsz_multiplier * num_train if pars.train_unsupervised else num_train
        with tqdm(total=total_n) as progress_bar:
            for batch_idx, (data, targ) in enumerate(train_loader):

                model.train()  # put model to training mode
                # using new data deformation with random resized crop
                if not pars.gaze_shift:
                    if pars.distort == 0 and pars.train_unsupervised:
                        x = [d.to(device, dtype=dtype) for d in data]
                    else:
                        x = data.to(device, dtype=dtype)
                else:
                    x = data.to(device, dtype=dtype)
                    
                
                if pars.train_unsupervised:
                    if pars.gaze_shift:
                        x = deform_gaze2(x, pars)
                    elif pars.distort == 3:
                        x1 = deform_data(x, 0.5, ['aff'], 4, 0.2, False, pars.device)
                        x2 = deform_data(x, 0.5, ['aff'], 4, 0.2, False, pars.device)
                        x = torch.cat((x1,x2), dim=0)
                    elif pars.distort == 0:
                        x = torch.cat(x, dim=0)
                else:
                    y = targ.to(device=device, dtype=torch.long)
                
                
                with torch.no_grad():
                    x1 = fix(x)
                scores = model(x1)

                if pars.train_unsupervised:
                    loss = criterion(scores)
                else:
                    loss = criterion(scores, y)
                running_loss += loss.item()

                progress_bar.set_postfix(loss=loss.item())
                progress_bar.update(x.size(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        running_loss /= len(train_loader)
        ep_loss.append(running_loss)
        if pars.train_unsupervised:
            print('Epoch %d, loss = %.4f' % (e, running_loss))            
        else:
            acc_test = check_accuracy(test_loader, fix, model, pars)
            acc_train = check_accuracy(train_loader, fix, model, pars)
            print('Epoch %d, loss = %.4f, train.acc = %.4f, test.acc = %.4f' % (e, running_loss, acc_train, acc_test))
            ep_acc.append(acc_test)

        if pars.train_unsupervised:
            if (e+1) % pars.log_every == 0:
                torch.save({'epoch': e + 1,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                           }, os.path.join(expdir, f"basenet_epoch_{e+1}_layer_{current_layer}.pth"))


def train_model_rand(train_loader, test_loader, net, classifier, pars, ep_loss, ep_acc):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    device=pars.device
    dtype = torch.float32
    # train_dat=data[0]; train_tar=data[1]
    # val_dat=data_test[2]; val_tar=data_test[3]
    # train_loader, test_loader = get_dataset(data, pars.batch_size, pars.num_train)
    net = net.to(device=device)  # move the model parameters to CPU/GPU
    classifier = classifier.to(device=device)

    if pars.train_unsupervised:
        scripted_transforms = get_scripted_transforms()
        lr = pars.LR
        if pars.loss == 'Hinge':
            criterion = ContrastiveHinge(pars.batch_size, device=pars.device)
        elif pars.loss == 'HingeNN':
            criterion = ContrastiveHingeNN(pars.batch_size, pars.thr1, pars.thr2, device=pars.device)
        elif pars.loss =='Chinge':
            criterion = Chinge_loss(pars.batch_size, pars.device, -1.25, 0)
        else:
            criterion = SimCLRLoss(pars.batch_size, pars.device)      
    else:
        lr = pars.clf_lr
        if pars.unsupervised:
            loss = pars.clf_loss
        else:
            loss = pars.loss
        if loss == 'Hinge':
            criterion = HingeLoss(pars.device)
        else:
            criterion = torch.nn.CrossEntropyLoss()

    opts = []
    for layer in np.arange(pars.NUM_LAYER):
        model = nn.Sequential(
                net[layer],
                classifier[layer]
                )
        if pars.OPT=='SGD':
            opts.append(torch.optim.SGD(model.parameters(), pars.LR))
        else:
            opts.append(torch.optim.Adam(model.parameters(), pars.LR))
    print(opts)

    epochs = pars.epochs * pars.NUM_LAYER
    for e in range(epochs):
        running_loss = 0
        for j in np.arange(0,pars.num_train, pars.batch_size):
            choose_layer = torch.randint(0, pars.NUM_LAYER, (1,)).item()
            fix = net[:choose_layer]
            model = nn.Sequential(
                net[choose_layer],
                classifier[choose_layer]
            )

            optimizer = opts[choose_layer]

            model.train()  # put model to training mode
            batch = next(iter(train_loader))
            x = batch[0].to(device, dtype=dtype)
            # x = torch.from_numpy(train_dat[j:j+pars.batch_size]).to(device=device, dtype=dtype)  # move to device, e.g. GPU
            if pars.train_unsupervised:
                img = x*0.5+0.5
                if pars.distort == 1:
                    x1 = (deform_data(img, 0.5, ['aff'], 4, 0.2, pars.batch_size, pars.device)-0.5)/0.5
                    x2 = (deform_data(img, 0.5, ['aff'], 4, 0.2, pars.batch_size, pars.device)-0.5)/0.5
                elif pars.distort == 2:
                    x1 = scripted_transforms(img)
                    x2 = scripted_transforms(img)
                elif pars.distort == 3:
                    # x1 = deform_data(x, 0.4, ['aff'], 4, 0.2, pars.batch_size, pars.device)
                    # x2 = deform_data(x, 0.4, ['aff'], 4, 0.2, pars.batch_size, pars.device)
                    x1 = deform_data(x, 0.4, ['aff'], 0, 0, pars.batch_size, pars.device)
                    x2 = deform_data(x, 0.4, ['aff'], 0, 0, pars.batch_size, pars.device)
                else:
                    img = x*0.5+0.5
                    x1 = (scripted_transforms(img)-0.5)/0.5
                    x2 = (scripted_transforms(img)-0.5)/0.5
                x = torch.cat((x1,x2), dim=0)
            else:
                # y = torch.from_numpy(train_tar[j:j+pars.batch_size]).to(device=device, dtype=torch.long)
                y = batch[1].to(device=device, dtype=torch.long)
            
            with torch.no_grad():
                x1 = fix(x)
            scores = model(x1)
            if pars.train_unsupervised:
                loss = criterion(scores)
            else:
                loss = criterion(scores, y)
            running_loss += loss.item()
            #print('Layer:{}, loss:{:.4f}'.format(choose_layer, loss.item()))

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

        running_loss /= (pars.num_train/pars.batch_size)
        ep_loss.append(running_loss) 

        if pars.train_unsupervised:
            print('Epoch %d, loss = %.4f' % (e, running_loss))
        else:
            acc = check_accuracy_rand(test_loader, net, classifier, pars)
            ep_acc.append(acc)
            print('Epoch {:d}, loss = {:.4f}, val.acc = {}'.format(e, running_loss, [round(x,4) for x in acc]))

               
def check_accuracy(dataloader, fix, model, pars):
        
    device=pars.device
    # train_loader, test_loader = get_dataset(data, pars.batch_size, pars.num_train)
    
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for batch_idx, (data, targ) in enumerate(dataloader):
            x = data.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
            y = targ.to(device=device, dtype=torch.long)
            x1 = fix(x)
            scores = model(x1)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        #print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc


def check_accuracy_rand(data, net, classifier, pars):

    all_acc = []

    for i in range(0, pars.NUM_LAYER):
        fix = net[:i]
        model = nn.Sequential(
            net[i],
            classifier[i]
        )
        acc = check_accuracy(data, fix, model, pars)
        all_acc.append(acc)

    return all_acc