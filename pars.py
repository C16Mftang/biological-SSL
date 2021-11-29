class PARS:
    def __init__(self, device, datapath, savepath,
                 process='E2E', update='BP', architecture = 'CONV', nonlinear='hardtanh', batchsize=500,
                 num_train=50000, unsupervised = False, gaze_shift=True, headsize = 64, distort=0, thr1=-1.25, thr2=0,
                 dataset = 'Cifar100', loss='SimCLR', optimizer='Adam', lr=0.0001, epochs=800,
                 clf_dataset = 'Cifar10', clf_loss='CE', clf_opt='Adam', clf_lr=0.0002, clf_epochs=400,
                 repeat = 5, loadnet=None, loadclf = None, patch_size=16, n_negs=16, n_patches=7, log_every=20,
                 augment_stl_train=True):
        self.process = process # 'E2E', 'GLL', 'RLL'
        self.update = update # 'BP', 'FA', 'UF', 'US'
        self.architecture = architecture # 'LW', 'CONV'
        self.nonlinear = nonlinear # 'hartanh','tanh', 'relu'
        self.batch_size = batchsize
        self.num_train = num_train

        self.unsupervised = unsupervised # SimCLR
        self.gaze_shift = True
        self.headsize = headsize # head for unsupervised learning
        self.distort = distort # 3 for our old one, 0 for our new one with random resized crop
        self.thr1 = thr1
        self.thr2 = thr2

        self.dataset = dataset # 'Cifar10', 'Cifar100'
        self.loss = loss # 'SimCLR', 'Hinge'
        self.OPT = optimizer # 'SGD', 'Adam', Only SGD with RLL
        self.LR = lr
        self.epochs = epochs # Epochs per layer
        
        self.clf_dataset = clf_dataset # 'Cifar10', 'Cifar100'
        self.clf_loss = clf_loss # 'CE', 'Hinge'
        self.clf_opt = clf_opt
        self.clf_lr = clf_lr
        self.clf_epochs = clf_epochs # epochs for training classifier
              
        self.repeat = repeat
        self.device = device
        self.datapath = datapath
        self.savepath = savepath
        self.loadnet = loadnet
        self.loadclf = loadclf

        self.patch_size = patch_size
        self.n_negs = n_negs
        self.n_patches = n_patches
        self.log_every = log_every

        self.augment_stl_train = augment_stl_train
    
    def __str__(self):
        res = ""
        for key, val in self.__dict__.items():
            if (key != 'loadclf'):
                res += "{}: {}\n".format(key, val)
            else:
                res += "{}: {}\n".format(key, val.keys() if val else val)
        return res