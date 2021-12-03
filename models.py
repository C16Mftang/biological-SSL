import torch.nn as nn
from layers import FALinear, FAConv2d

def setup_net(pars):
    net = nn.Sequential()
    classifier = nn.Sequential()
    head = nn.Sequential()

    if pars.unsupervised:
        if pars.clf_dataset == 'Cifar100':
            NUM_CLASS = 100
        else:
            NUM_CLASS = 10
    else:
        if pars.dataset == 'Cifar100':
            NUM_CLASS = 100
        else:
            NUM_CLASS = 10
                    
    if pars.architecture == 'CONV6':
        HW = 64
        NUM_CHANNEL = 32
        pars.NUM_LAYER = 5

        for i in range(pars.NUM_LAYER):
            layer = nn.Sequential()

            if i==0:
                if pars.update == 'BP':
                    layer.add_module('conv', nn.Conv2d(3,int(NUM_CHANNEL),3,padding=1,bias=False))
                elif pars.update == 'FA':
                    layer.add_module('conv', FAConv2d(3,int(NUM_CHANNEL),3,padding=1,bias=False))
                else:
                    layer.add_module('conv', nn.Conv2d(3,int(NUM_CHANNEL),3,padding=1,bias=False))
                layer.add_module('activation', nn.Hardtanh())
            
            elif (i == 1) or (i == 3):
                if pars.update == 'BP':
                    layer.add_module('conv', nn.Conv2d(int(NUM_CHANNEL),int(NUM_CHANNEL),3,padding=1,bias=False))
                elif pars.update == 'FA':
                    layer.add_module('conv', FAConv2d(int(NUM_CHANNEL),int(NUM_CHANNEL),3,padding=1,bias=False))
                else:
                    layer.add_module('conv', nn.Conv2d(int(NUM_CHANNEL),int(NUM_CHANNEL),3,padding=1,bias=False))
                layer.add_module('maxpool', nn.MaxPool2d(2))
                HW /= 2

            elif i == 2:
                if pars.update == 'BP':
                    layer.add_module('conv', nn.Conv2d(int(NUM_CHANNEL),int(NUM_CHANNEL*2),3,padding=1,bias=False))
                elif pars.update == 'FA':
                    layer.add_module('conv', FAConv2d(int(NUM_CHANNEL),int(NUM_CHANNEL*2),3,padding=1,bias=False))
                else:
                    layer.add_module('conv', nn.Conv2d(int(NUM_CHANNEL),int(NUM_CHANNEL*2),3,padding=1,bias=False))
                layer.add_module('activation', nn.Hardtanh())
                NUM_CHANNEL *= 2

            else:
                if pars.update == 'BP':
                    layer.add_module('conv', nn.Conv2d(int(NUM_CHANNEL),int(NUM_CHANNEL*8),3,padding=1,bias=False))
                elif pars.update == 'FA':
                    layer.add_module('conv', FAConv2d(int(NUM_CHANNEL),int(NUM_CHANNEL*8),3,padding=1,bias=False))
                else:
                    layer.add_module('conv', nn.Conv2d(int(NUM_CHANNEL),int(NUM_CHANNEL*8),3,padding=1,bias=False))
                layer.add_module('maxpool', nn.MaxPool2d(2))
                NUM_CHANNEL *= 8
                HW /= 2
            
      
            aux = nn.Sequential(
                # nn.AvgPool2d(2),
                nn.Flatten(),
                # nn.Dropout(p=0.5),
            )
            if pars.update=='FA':
                aux.add_module('fc', FALinear(8192,NUM_CLASS))
            else:
                aux.add_module('fc', nn.Linear(8192,NUM_CLASS))

            if pars.unsupervised:
                if pars.loss != 'CLAPP' and pars.loss != 'GazeHingeNN':
                    auxhead = nn.Sequential(
                        # nn.AvgPool2d(2),
                        nn.Flatten(),
                    )
                    if pars.update=='FA':
                        auxhead.add_module('fc', FALinear(8192,pars.headsize,bias=False))
                    else:
                        auxhead.add_module('fc', nn.Linear(8192,pars.headsize,bias=False))
                elif pars.loss == 'GazeHingeNN':
                    auxhead = nn.Sequential(
                        # nn.AvgPool2d(2),
                        nn.Flatten(),
                    )
                    if pars.update=='FA':
                        auxhead.add_module('fc', FALinear(int(NUM_CHANNEL*HW*HW),pars.headsize,bias=False))
                    else:
                        auxhead.add_module('fc', nn.Linear(int(NUM_CHANNEL*HW*HW),pars.headsize,bias=False))
            

            net.add_module('layer%d'%i, layer)
            if pars.process != 'E2E':
                classifier.add_module('aux%d'%i, aux)
            elif i==(pars.NUM_LAYER-1):
                classifier.add_module('aux%d'%i, aux)

            if pars.unsupervised:
                if pars.process != 'E2E':
                    head.add_module('auxhead%d'%i, auxhead)
                elif i==(pars.NUM_LAYER-1):
                    head.add_module('auxhead%d'%i, auxhead)

    elif pars.architecture == 'VGG6':
        HW = 64
        NUM_CHANNEL = 128
        CH = 1 if pars.gaze_shift else 3
        pars.NUM_LAYER = 6

        for i in range(pars.NUM_LAYER):
            layer = nn.Sequential()

            if i==0:
                if pars.process != 'E2E':
                    layer.add_module('conv', nn.Conv2d(CH,int(NUM_CHANNEL),3,padding=1,bias=False))
                elif pars.update == 'FA':
                    layer.add_module('conv', FAConv2d(CH,int(NUM_CHANNEL),3,padding=1,bias=False))
                else:
                    layer.add_module('conv', nn.Conv2d(CH,int(NUM_CHANNEL),3,padding=1,bias=False))
                layer.add_module('activation', nn.Hardtanh())
            
            elif (i == 1) or (i == 3) or (i == 4):
                if pars.process != 'E2E':
                    layer.add_module('conv', nn.Conv2d(int(NUM_CHANNEL),int(NUM_CHANNEL)*2,3,padding=1,bias=False))
                elif pars.update == 'FA':
                    layer.add_module('conv', FAConv2d(int(NUM_CHANNEL),int(NUM_CHANNEL)*2,3,padding=1,bias=False))
                else:
                    layer.add_module('conv', nn.Conv2d(int(NUM_CHANNEL),int(NUM_CHANNEL)*2,3,padding=1,bias=False))
                layer.add_module('maxpool', nn.MaxPool2d(2))
                HW /= 2
                NUM_CHANNEL *= 2

            elif i == 2:
                if pars.process != 'E2E':
                    layer.add_module('conv', nn.Conv2d(int(NUM_CHANNEL),int(NUM_CHANNEL),3,padding=1,bias=False))
                elif pars.update == 'FA':
                    layer.add_module('conv', FAConv2d(int(NUM_CHANNEL),int(NUM_CHANNEL),3,padding=1,bias=False))
                else:
                    layer.add_module('conv', nn.Conv2d(int(NUM_CHANNEL),int(NUM_CHANNEL),3,padding=1,bias=False))
                layer.add_module('activation', nn.Hardtanh())

            elif i == 5:
                if pars.process != 'E2E':
                    layer.add_module('conv', nn.Conv2d(int(NUM_CHANNEL),int(NUM_CHANNEL),3,padding=1,bias=False))
                elif pars.update == 'FA':
                    layer.add_module('conv', FAConv2d(int(NUM_CHANNEL),int(NUM_CHANNEL),3,padding=1,bias=False))
                else:
                    layer.add_module('conv', nn.Conv2d(int(NUM_CHANNEL),int(NUM_CHANNEL),3,padding=1,bias=False))
                layer.add_module('maxpool', nn.MaxPool2d(2))
                HW /= 2
            
            aux = nn.Sequential(
                nn.AvgPool2d(2),
                nn.Flatten(),
            )
            if pars.update=='FA':
                aux.add_module('fc', FALinear(int(NUM_CHANNEL*(HW/2)*(HW/2)), NUM_CLASS))
            else:
                aux.add_module('fc', nn.Linear(int(NUM_CHANNEL*(HW/2)*(HW/2)), NUM_CLASS))

            if pars.unsupervised:
                if not pars.gaze_shift:
                    auxhead = nn.Sequential(
                        nn.AvgPool2d(2),
                        nn.Flatten(),
                    )
                    if pars.update=='FA':
                        auxhead.add_module('fc', FALinear(int(NUM_CHANNEL*(HW/2)*(HW/2)),pars.headsize,bias=False))
                    else:
                        auxhead.add_module('fc', nn.Linear(int(NUM_CHANNEL*(HW/2)*(HW/2)),pars.headsize,bias=False))
                elif pars.loss == 'GazeHingeNN':
                    auxhead = nn.Sequential(
                        # nn.AvgPool2d(2),
                        nn.Flatten(),
                    )
                    if pars.update=='FA':
                        auxhead.add_module('fc', FALinear(int(NUM_CHANNEL*HW*HW),pars.headsize,bias=False))
                    else:
                        auxhead.add_module('fc', nn.Linear(int(NUM_CHANNEL*HW*HW),pars.headsize,bias=False))
            

            net.add_module('layer%d'%i, layer)
            if pars.process != 'E2E':
                classifier.add_module('aux%d'%i, aux)
            elif i==(pars.NUM_LAYER-1):
                classifier.add_module('aux%d'%i, aux)

            if pars.unsupervised:
                if pars.loss != 'CLAPP':
                    if pars.process != 'E2E':
                        head.add_module('auxhead%d'%i, auxhead)
                    elif i==(pars.NUM_LAYER-1):
                        head.add_module('auxhead%d'%i, auxhead)
        

    return net, classifier, head