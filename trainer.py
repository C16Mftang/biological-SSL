import os
import torch
from pars import PARS
from utils import *
from models import setup_net
from get_data import *
import time
import json

def main(pars):
    if pars.loadnet == None:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        expdir = os.path.join(pars.savepath, timestr)
        if not os.path.exists(expdir):
            os.makedirs(expdir)
    else:
        expdir = pars.loadnet.rsplit('/',1)[0]

    pars.expdir = expdir
    print(pars.expdir)

    pars.train_unsupervised = pars.unsupervised
    start_epoch = 0
    dtype = torch.float32

    if pars.gaze_shift:
        print('---------------')
        base_train_loader, base_test_loader = get_stl10_unlabeled_patches(pars.datapath, pars.batch_size, pars.num_train)
    else:
        if pars.distort == 0:
            base_train_loader, base_test_loader = get_stl10_unlabeled_deform(pars.datapath, pars.batch_size, pars.num_train)
        else:
            base_train_loader, base_test_loader = get_stl10_unlabeled_vanilla_deform(pars.datapath, pars.batch_size, pars.num_train)

    clf_train_loader, clf_test_loader = get_stl10_labeled(pars.datapath, pars.batch_size, pars)
    
    test_acc_all = []
    for rep in range(pars.repeat):
        print("\nRep {}".format(rep+1))

        net, classifier, head = setup_net(pars)      
        val_loss = []
        val_acc = []
        lw_test_acc = []
        if pars.unsupervised:
            head_loss = []
            if pars.process == 'RLL':
                pars.train_unsupervised = True
                train_model_rand(base_train_loader, base_test_loader, net, head, pars, head_loss, None)
                print('Train classifier')
                pars.train_unsupervised = False
                for i in range(pars.NUM_LAYER):
                    train_model(clf_train_loader, clf_test_loader, net[:(i+1)], classifier[i], pars, val_loss, val_acc, pars.expdir)
                    test_acc = check_accuracy(clf_test_loader, net[:(i+1)], classifier[i], pars)
                    print('Rep: %d, Layer: %d, te.acc = %.4f' % (rep+1, i, test_acc))
                    lw_test_acc.append(test_acc)
            elif pars.process == 'GLL':
                # start from where the previous check point was saved
                if pars.loadnet:
                    if pars.loadnet.startswith('.'):
                        start_layer = pars.loadnet.rsplit('.')[1].rsplit('_')[-1]
                    else:
                        start_layer = pars.loadnet.rsplit('.')[0].rsplit('_')[-1]
                else:
                    start_layer = 0
                for i in range(int(start_layer), pars.NUM_LAYER):
                    print('LAYER:%d'%i)
                    fix = net[:i]
                    print("Fixed part:\n", fix)
                    if pars.loss != 'CLAPP':
                        model = nn.Sequential(
                            net[i],
                            head[i]
                        )        
                    else:
                        model = net[i]

                    if i != int(start_layer):
                        pars.loadnet = None  

                    if pars.loadnet:
                        checkpoint = torch.load(pars.loadnet)
                        model.load_state_dict(checkpoint['state_dict'])
                        timestr = pars.loadnet.split('/')[-2].strip()
                        expdir = os.path.join(pars.savepath, timestr)
                        for pre in range(i):
                            print('Loading from fixed layer ', pre)
                            fix_path = os.path.join(expdir, f'basenet_epoch_{pars.epochs}_layer_{pre}.pth')
                            fix_checkpoint = torch.load(fix_path)
                            loaded_fix_weights = fix_checkpoint['state_dict']
                            new_fix_weights = fix[pre].state_dict()
                            for k in new_fix_weights.keys():
                                if pars.loss != 'CLAPP':
                                    new_fix_weights[k] = loaded_fix_weights['0.'+k]
                                else:
                                    new_fix_weights[k] = loaded_fix_weights[k]
                            fix[pre].load_state_dict(new_fix_weights)
 
                        if os.path.exists(expdir):
                            start_epoch = checkpoint['epoch']
                            print(f'Checkpoint loaded, resuming from {start_epoch}')
                            print(f"Resuming from layer {start_layer}")
                            print(f'Saving to existing path: {expdir}')
                            pars.expdir = expdir
                        
                    if pars.loadclf:
                        classifier.load_state_dict(pars.loadclf)
                    print("Part to train:\n", model)
                    print('Classifier:\n', classifier[i])

                    pars.train_unsupervised = True
                    train_model(base_train_loader, base_test_loader, fix, model, pars, head_loss, None, pars.expdir, current_layer=i)

                    pars.train_unsupervised = False
                    if not pars.classify_whole_net:
                        print('Train Classifier on current layer')
                        train_model(clf_train_loader, clf_test_loader, net[:(i+1)], classifier[i], pars, val_loss, val_acc, pars.expdir)
                        test_acc = check_accuracy(clf_test_loader, net[:(i+1)], classifier[i], pars)
                        print('Rep: %d, Layer: %d, te.acc = %.4f' % (rep+1, i, test_acc))
                        lw_test_acc.append(test_acc)
                        print()
                
                if pars.classify_whole_net:
                    print('Train Classifier for the whole model')
                    pars.train_unsupervised = False
                    train_model(clf_train_loader, clf_test_loader, net, classifier[-1], pars, val_loss, val_acc, pars.expdir)
                    test_acc = check_accuracy(clf_test_loader, net, classifier[-1], pars)
                    print('Rep: %d, te.acc = %.4f' % (rep+1, test_acc))
                    lw_test_acc.append(test_acc)

            else: # 'E2E'
                fix = nn.Sequential()
                model = nn.Sequential(
                    net,
                    head
                )
                
                if pars.loadnet:
                    checkpoint = torch.load(pars.loadnet)
                    model.load_state_dict(checkpoint['state_dict'])
                    timestr = pars.loadnet.split('/')[-2].strip()
                    expdir = os.path.join(pars.savepath, timestr)
                    if os.path.exists(expdir):
                        start_epoch = checkpoint['epoch']
                        print(f'Checkpoint loaded, resuming from {start_epoch}')
                        print(f'Saving to existing path: {expdir}')
                        pars.expdir = expdir
                if pars.loadclf:
                    classifier.load_state_dict(pars.loadclf)
                print("Part to train:\n", model)
                print('Classifier:\n', classifier)

                pars.train_unsupervised = True
                train_model(base_train_loader, base_test_loader, fix, model, pars, head_loss, None, pars.expdir)

                print('Train Classifier')
                pars.train_unsupervised = False
                train_model(clf_train_loader, clf_test_loader, net, classifier, pars, val_loss, val_acc, pars.expdir)
                test_acc = check_accuracy(clf_test_loader, net, classifier, pars)
                print('Rep: %d, te.acc = %.4f' % (rep+1, test_acc))
                lw_test_acc.append(test_acc)
        
        torch.save(net.state_dict(), os.path.join(pars.expdir, "final_basenet.mdlp"))
        torch.save(classifier.state_dict(), os.path.join(pars.expdir, "final_classifier.mdlp"))

        if pars.unsupervised:
            np.save(pars.expdir+'/head_loss_rep_{}_after_epoch{}'.format(rep+1, start_epoch), head_loss)

        np.save(pars.expdir+'/loss_rep_{}_'.format(rep+1), val_loss)
        np.save(pars.expdir+'/val.acc_rep_{}_'.format(rep+1), val_acc)
        np.save(pars.expdir+'/te.acc_rep_{}_'.format(rep+1), lw_test_acc)

        test_acc_all.append(lw_test_acc)

    print('\nAll reps test.acc:')
    for acc in test_acc_all:
        print(acc)
    np.save(pars.expdir+'/te.acc.all_', test_acc_all)


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    
    datapath = './data'
    savepath = './save'
    pars = PARS(device, datapath, savepath)
    pars.process = 'E2E'
    pars.update = 'BP'
    pars.architecture = 'VGG6'
    pars.gaze_shift = False
    pars.loss = 'HingeNN2'
    pars.thr1 = 1
    pars.thr2 = 3
    pars.grad_block = True
    print(pars)
    main(pars)
    with open(os.path.join(pars.expdir, 'configs.json'), 'w') as fp:
       json.dump(pars.__dict__, fp)
