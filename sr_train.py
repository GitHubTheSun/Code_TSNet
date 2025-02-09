from tool import *
from tool import _label_sum, initWeights
import os
import datetime
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data.dataset import Dataset
from torchvision.datasets import ImageFolder
from torch.autograd import Variable

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train_epoch(model, loader, optimizer, epoch, n_epochs, print_freq=125):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # Model on train mode
    model.train()

    end = time.time()
    for batch_idx, datas in enumerate(loader):
        # Create vaiables
        images, labels = datas['images'], datas['labels']
        images = images.view(len(datas['images']) * 2, 1, 256, 256)
        labels = labels.view(len(datas['labels']) * 2)

        images = Variable(images.to(device))
        labels = Variable(labels.to(device))

        # compute output
        outputs = model(images)

        # loss = torch.nn.functional.cross_entropy(outputs, labels)
        output1 = F.log_softmax(outputs, dim=1)
        loss = F.nll_loss(output1, labels)
        pred = outputs.max(1, keepdim=True)[1]

        # measure accuracy and record loss
        batch_size = labels.size(0)
        acc.update(pred.eq(labels.view_as(pred)).sum().item() / batch_size, batch_size)
        losses.update(loss.item() / batch_size, batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val * batch_size, losses.avg),
                'Acc %.4f (%.4f)' % (acc.val, acc.avg),
            ])
            print(res)

    # Return summary statistics
    return batch_time.avg, losses.avg, acc.avg


def test_epoch(model, loader, is_test=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    N = 0  # 正确被分类为载体图像的数目
    P = 0  # 正确被分类为载密图像的数目

    # Model on eval mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for batch_idx, datas in enumerate(loader):
            # Create vaiables
            images, labels = datas

            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            # compute output
            outputs = model(images)

            output1 = F.log_softmax(outputs, dim=1)
            loss = F.nll_loss(output1, labels)
            pred = outputs.max(1, keepdim=True)[1]

            # measure accuracy and record loss
            batch_size = labels.size(0)
            acc.update(pred.eq(labels.view_as(pred)).sum().item() / batch_size, batch_size)
            losses.update(loss.item() / batch_size, batch_size)

            batch_time.update(time.time() - end)
            end = time.time()

            a, b, c = _label_sum(pred, labels)
            N += a
            P += b

        S = len(loader.dataset) / 2  # 待测数据中所有的载密图像个数  具体的数量具体设置，如果载体图像等于载密数量则这样写代码即可
        C = len(loader.dataset) / 2  # 待测数据集中所有载体图像的个数
        FPR = (C - N) / C  # 虚警率 即代表载体图像被误判成载密图像 占所有载体图像的比率
        FNR = (S - P) / S  # 漏检率 即代表载密图像被误判成载体图像 占所有载密图像的比率

        res = '\t'.join([
            'Test:' if is_test else 'Valid:',
            'Time %.3f s' % batch_time.sum,
            'Loss %.4f ' % losses.avg,
            'Acc %.4f ' % acc.avg,
            'FPR %.4f' % FPR,
            'FNR %.4f' % FNR,
        ])
        print(res)

    # Return summary statistics
    return batch_time.avg, losses.avg, acc.avg


def train(model, train_set, valid_set, test_set, save, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    if torch.cuda.is_available():
        model = model.cuda()

    model1 = model
    model_wrapper = model
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model_wrapper = torch.nn.DataParallel(model).cuda()

    # xunet
    '''
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.95)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=90, gamma=0.9)
    '''

    # lwenet
    '''
    n_epochs = 200
    optimizer = torch.optim.SGD(model_wrapper.parameters(), lr=0.01, weight_decay=0.0005,
                                momentum=0.9)
    DECAY_EPOCH = [80, 140, 180]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=DECAY_EPOCH,
                                                     gamma=0.1)
    '''

    # yedroudjnet
    '''
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.95, weight_decay=0.0001)  # 优化程序
    DECAY_EPOCH = [100, 150]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=DECAY_EPOCH, gamma=0.1)
    '''

    # YeNet
    '''
    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.4,) # rho=0.95, eps=1e-8, weight_decay=5e-4)
    DECAY_EPOCH = [100, 150]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=DECAY_EPOCH, gamma=0.1)
    '''

    # SRNet
    # '''
    n_epochs = 571
    optimizer = torch.optim.Adamax(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                                   weight_decay=2e-4)  # SRNet
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=457, gamma=0.1)  # 457次时 调整学习率
    # '''

    # GFSNet
    '''
    def init_weights(model):
        if type(model) == nn.Conv2d:
            if model.weight.requires_grad:
                nn.init.kaiming_normal_(model.weight, mode='fan_in', nonlinearity='leaky_relu')
        if type(model) == nn.Linear:
            nn.init.normal_(model.weight, mean=0, std=0.01)
            # nn.init.constant_(model.bias.data, val=0)

    model.apply(init_weights)

    params = model.parameters()
    params_wd, params_rest = [], []
    for param_item in params:
        if param_item.requires_grad:
            (params_wd if param_item.dim() != 1 else params_rest).append(param_item)  # 卷积层中的bias和FC层中的bias，以及BN层不参与权重衰减
    param_groups = [{'params': params_wd, 'weight_decay': 5e-4}, {'params': params_rest}]
    n_epochs = 160
    optimizer = torch.optim.SGD(param_groups, lr=0.02, momentum=0.9)
    DECAY_EPOCH = [80, 120, 140]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=DECAY_EPOCH, gamma=0.1)
    '''

    # Start log
    with open(os.path.join(save, 'results.csv'), 'w') as f:
        f.write('epoch,train_loss,train_acc,valid_loss,valid_acc,test_acc\n')

    # Train model
    best_acc = 0
    for epoch in range(n_epochs):
        _, train_loss, train_acc = train_epoch(
            model=model_wrapper,
            loader=train_set,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs
        )
        scheduler.step()
        _, valid_loss, valid_acc = test_epoch(
            model=model_wrapper,
            loader=valid_set if valid_set else test_set,
            is_test=(not valid_set)
        )

        # Determine if model is the best
        if valid_set:
            if valid_acc >= best_acc:
                best_acc = valid_acc
                print('New best recently acc: %.4f' % best_acc)
                torch.save(model.state_dict(), os.path.join(save, 'model.dat'))
        else:
            torch.save(model.state_dict(), os.path.join(save, 'model.dat'))
        torch.save(model.state_dict(), os.path.join(save, 'epoch{}model.dat'.format(epoch + 1)))

        # Log results
        with open(os.path.join(save, 'results.csv'), 'a') as f:
            f.write('%03d,%0.6f,%0.6f,%0.5f,%0.5f,\n' % (
                (epoch + 1),
                train_loss,
                train_acc,
                valid_loss,
                valid_acc,
            ))

    # Final test of model on test set
    model1.load_state_dict(torch.load(os.path.join(save, 'model.dat')))
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model1 = torch.nn.DataParallel(model).cuda()
    test_results = test_epoch(
        model=model1,
        loader=test_set,
        is_test=True
    )
    _, _, test_acc = test_results
    with open(os.path.join(save, 'results.csv'), 'a') as f:
        f.write(',,,,,%0.5f\n' % test_acc)
    print('Final test acc: %.4f' % test_acc)


# SRH 213
def demo(batch_size=16, seed=None):

    # train_path = 'D:\shiyao_DataSet\Dataset\BOSSBase_256\S_UNIWARD\\train_boss_BOSW2\SUN-0.1'
    # test_path = 'D:\shiyao_DataSet\Dataset\BOSSBase_256\S_UNIWARD\\test_boss\SUN-0.1'
    # valid_path = 'D:\shiyao_DataSet\Dataset\BOSSBase_256\S_UNIWARD\\valid_boss\SUN-0.1'

    # train_path = 'D:\shiyao_DataSet\Dataset\BOSSBase_256\WOW\\train_boss_BOSW2\WOW-0.2'
    # test_path = 'D:\shiyao_DataSet\Dataset\BOSSBase_256\WOW\\test_boss\WOW-0.2'
    # valid_path = 'D:\shiyao_DataSet\Dataset\BOSSBase_256\WOW\\valid_boss\WOW-0.2'

    # train_path = 'D:\shiyao_DataSet\Dataset\BOSSBase_256\HILL\\train_boss_BOSW2\HILL-0.2'
    # test_path = 'D:\shiyao_DataSet\Dataset\BOSSBase_256\HILL\\test_boss\HILL-0.2'
    # valid_path = 'D:\shiyao_DataSet\Dataset\BOSSBase_256\HILL\\valid_boss\HILL-0.2'

    # train_path = 'D:\shiyao_DataSet\Dataset\BOSSBase_256\Mipod\\train\MIPOD-0.2'
    # test_path = 'D:\shiyao_DataSet\Dataset\BOSSBase_256\Mipod\\test\MIPOD-0.2'
    # valid_path = 'D:\shiyao_DataSet\Dataset\BOSSBase_256\Mipod\\valid\MIPOD-0.2'

    train_path = 'G:\TransportDataset\dataset\\train\wow041'
    test_path = 'G:\TransportDataset\dataset\\test\wow041'
    valid_path = 'G:\TransportDataset\dataset\\valid\wow041'

    print("训练数据集：", train_path)
    print("验证数据集：", valid_path)
    print("测试数据集：", test_path)

    # Data transforms
    train_transform = torchvision.transforms.Compose([AugData(), ToTensor()])
    data_transform = torchvision.transforms.Compose([torchvision.transforms.Grayscale(),
                                                     torchvision.transforms.ToTensor()])

    # Datasets
    Trainfilepath_cover = train_path + '\\0\\'
    Trainfilepath_stego = train_path + '\\1\\'
    Testfilepath = test_path + '\\'
    train_set = torch.utils.data.DataLoader(
        DatasetPair(Trainfilepath_cover, Trainfilepath_stego, train_transform),
        batch_size=batch_size,
        shuffle=True)
    test_set = torch.utils.data.DataLoader(
        ImageFolder(Testfilepath, data_transform),
        batch_size=batch_size * 2,
        shuffle=False)

    if valid_path:
        valid_path = valid_path + '\\'
        valid_set = torch.utils.data.DataLoader(
            ImageFolder(valid_path, data_transform),
            batch_size=batch_size * 2,
            shuffle=False)
    else:
        valid_set = None

    # 常修改区域 网络名称 && log保存路径
    # from Net.LWENet import lwenet as Net
    from Net.SRNet import SRNet as Net
    # save = 'C:\\Users\Lenovo\Desktop\第三个工作实验记录\EXP\\SRNet\\wow02'
    # save = 'E:\\work3exp_SR_ssy\\mipod02_cl'
    save = 'G:\\transportresult\\SRNet\\wow04'
    print("log保存路径：", save)
    assert not os.path.exists(save), print("文件重名，可能覆盖之前数据！")

    model = Net()
    print(model)
    cudnn.benchmark = True  # 设置卷积神经网络加速
    # 初始化参数
    model.apply(initWeights)

    # model_path = "E:\work3exp_SR_ssy\mipod03_cl\\"
    # best_ckpt = os.path.join(model_path, 'model.dat')
    # print("model_Path:", best_ckpt)
    # model.load_state_dict(torch.load(best_ckpt))
    # model = model.cuda()

    # Print number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: ", num_params)

    # Make save directory
    if not os.path.exists(save):
        os.makedirs(save)
    # if not os.path.isdir(save):
    #     raise Exception('%s is not a dir' % save)

    # Train the model
    starttime = datetime.datetime.now()
    train(model=model, train_set=train_set, valid_set=valid_set, test_set=test_set, save=save,
          seed=seed)
    print('Done!')
    endtime = datetime.datetime.now()
    print("==================================================================")
    print("==>Train Dataset：", train_path)
    print("==>Valid Dataset：", valid_path)
    print("==>Test  Dataset：", test_path)
    print("==>The logs path：", save)
    print("==>Total parameters: ", num_params)
    print("==>Programma Strat Time：", starttime)
    print("==>Programma End   Time：", endtime)
    print("==================================================================")


if __name__ == '__main__':
    import time
    time.sleep((4.4+)*3600)
    demo()
