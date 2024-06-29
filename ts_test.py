
from tool import *
from tool import _label_sum
import os
import datetime
import torch
import torchvision
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data.dataset import Dataset
from torchvision.datasets import ImageFolder
from torch.autograd import Variable

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def test_epoch(model, loader, is_test=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    all_time = 0
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
            start = time.time()
            outputs = model(images)
            all_time += time.time() - start

            output1 = F.log_softmax(outputs, dim=1)
            loss = F.nll_loss(output1, labels)
            pred = outputs.max(1, keepdim=True)[1]

            # measure accuracy and record loss
            batch_size = labels.size(0)
            # _, pred = outputs.data.cpu().topk(1, dim=1)
            # acc.update(torch.ne(pred.squeeze(), labels.cpu()).float().sum().item() / batch_size, batch_size)
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
    return batch_time.avg, losses.avg, acc.avg, all_time


def demo(batch_size=4):

    # test_path = "D:\shiyao_DataSet\Dataset\BOSSBase_256\S_UNIWARD\\test_boss\SUN-0.4"
    # test_path = "D:\shiyao_DataSet\Dataset\BOSSBase_256\S_UNIWARD\\test_boss\SUN-0.3"
    # test_path = "D:\shiyao_DataSet\Dataset\BOSSBase_256\S_UNIWARD\\test_boss\SUN-0.2"
    # test_path = "D:\shiyao_DataSet\Dataset\BOSSBase_256\S_UNIWARD\\test_boss\SUN-0.1"
    # test_path = "D:\shiyao_DataSet\Dataset\BOSSBase_256\WOW\\test_boss\WOW-0.4"
    # test_path = "D:\shiyao_DataSet\Dataset\BOSSBase_256\WOW\\test_boss\WOW-0.3"
    # test_path = "D:\shiyao_DataSet\Dataset\BOSSBase_256\WOW\\test_boss\WOW-0.2"
    # test_path = "D:\shiyao_DataSet\Dataset\BOSSBase_256\WOW\\test_boss\WOW-0.1"
    # test_path = "D:\shiyao_DataSet\Dataset\BOSSBase_256\HILL\\test_boss\HILL-0.4"
    # test_path = "D:\shiyao_DataSet\Dataset\BOSSBase_256\HILL\\test_boss\HILL-0.3"
    # test_path = "D:\shiyao_DataSet\Dataset\BOSSBase_256\HILL\\test_boss\HILL-0.2"
    # test_path = "D:\shiyao_DataSet\Dataset\BOSSBase_256\HILL\\test_boss\HILL-0.1"
    # test_path = "D:\shiyao_DataSet\Dataset\BOSSBase_256\Mipod\\test\MIPOD-0.4"
    # test_path = "D:\shiyao_DataSet\Dataset\BOSSBase_256\Mipod\\test\MIPOD-0.3"
    # test_path = "D:\shiyao_DataSet\Dataset\BOSSBase_256\Mipod\\test\MIPOD-0.2"
    # test_path = "D:\shiyao_DataSet\Dataset\BOSSBase_256\Mipod\\test\MIPOD-0.1"

    # test_path = "G:\ALASKA_v2_512_20000_Downsample256\dataset\\test\wow04"
    test_path = "G:\TransportDataset\dataset\\test\wow04"
    # test_path = "G:\ALASKA_v2_512_20000_Downsample256\dataset\\test\wow03"
    # test_path = "G:\ALASKA_v2_512_20000_Downsample256\dataset\\test\wow02"


    print("测试数据集：", test_path)

    # Data transforms
    data_transform = torchvision.transforms.Compose([torchvision.transforms.Grayscale(),
                                                     torchvision.transforms.ToTensor()])

    # Datasets
    Testfilepath = test_path + '\\'
    test_set = torch.utils.data.DataLoader(
        ImageFolder(Testfilepath, data_transform),
        batch_size=batch_size * 2,
        shuffle=False)

    # 常修改区域 网络名称 && log保存路径
    from Net.ISTSNet import ISTSNet as Net
    # from Net.GFSNet import GFS_Net as Net
    # from Net.LWENet import lwenet as Net
    # from Net.AblationNet.ISTSNet_GAP import ISTSNet as Net
    # from Net.AblationNet.ISTSNet32_32 import ISTSNet as Net
    # from Net.AblationNet.ISTSNetSE import ISTSNet as Net
    # from Net.AblationNet.ISTSNet_IP import ISTSNet as Net
    # from Net.AblationNet.ISTSNet_ALS import ISTSNet as Net
    # from Net.AblationNet.ISTSNet_FFS import ISTSNet as Net
    # from Net.AblationNet.ISTSNet_Cat import ISTSNet as Net
    # from Net.SRNet import SRNet as Net
    model = Net()
    print(model)
    cudnn.benchmark = True  # 设置卷积神经网络加速

    # Print number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: ", num_params)

    # 测试程序加载model过程
    # model_path = 'C:\\Users\Lenovo\Desktop\TSNet实验记录备份\实验记录\LWENet\Boss'
    # model_path = 'G:\研三寒假实验\EXP\消融实验\TSNet_GAP\wow04'
    # model_path = 'G:\研三寒假实验\EXP\消融实验\TSNet_32_32\ISTSNet32_32_wow04'
    # model_path = 'G:\研三寒假实验\EXP\消融实验\TSNet_SE\wow04'
    # model_path = 'G:\研三寒假实验\EXP\消融实验\TSNet_IP\ISTSNet_IP_wow04'
    # model_path = 'G:\研三寒假实验\EXP\消融实验\TSNet_ALS\wow04'
    # model_path = 'G:\研三寒假实验\EXP\消融实验\TSNet_FFS\wow04'
    # model_path = 'G:\研三寒假实验\EXP\消融实验\TSNet_Cat\wow04'
    # model_path = 'C:\\Users\Lenovo\Desktop\第三个工作实验记录\EXP\LWENet\wow04'
    # model_path = 'C:\\Users\Lenovo\Desktop\第三个工作实验记录\EXP\GFSNet\wow04'
    model_path = 'C:\\Users\Lenovo\Desktop\第三个工作实验记录\EXP\ISTSNet\wow04'
    # model_path = 'E:\work3exp_SR_ssy\wow04_1205'
    # model_path = 'C:\\Users\Lenovo\Desktop\第三个工作实验记录\EXP\LWENet\mipod01_cl'
    # best_ckpt = os.path.join(model_path, 'lwe_h1cl_model.dat')
    best_ckpt = os.path.join(model_path, 'model.dat')
    print("model_Path:", best_ckpt)
    model.load_state_dict(torch.load(best_ckpt))
    model = model.cuda()

    # Train the model
    starttime = datetime.datetime.now()
    _, _, test_acc, all_time = test_epoch(model, loader=test_set, is_test=True)
    print('Final test acc: %.4f' % test_acc)
    print('Done!')
    endtime = datetime.datetime.now()
    print("==================================================================")
    print("==>Test  Dataset：", test_path)
    print("==>Total parameters: ", num_params)
    print("==>Programma Strat Time：", starttime)
    print("==>Programma End   Time：", endtime)
    print("==>Programma test time：", all_time)
    print("==================================================================")


if __name__ == '__main__':
    import time
    time.sleep(0)
    demo(batch_size=16)
