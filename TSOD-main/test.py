import argparse
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from datasets.ImbalanceCIFAR import IMBALANCECIFAR10, IMBALANCECIFAR100
from datasets.SCOODBenchmarkDataset import SCOODDataset
from datasets.ImbalanceImageNet import LT_Dataset
from models.resnet import ResNet18
from models.resnet_imagenet import ResNet50
from utils.utils import *
from utils.ltr_metrics import *
from utils.ood_metrics import *
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.nn.utils import spectral_norm
import matplotlib

# 使用Agg后端（不依赖GUI）
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 递归函数，对神经网络模型中的所有卷积层应用谱归一化
def apply_spectral_norm_to_conv(module):
    if isinstance(module, torch.nn.Conv2d):
        return spectral_norm(module)
    for child_name, child_module in module.named_children():
        module.add_module(child_name, apply_spectral_norm_to_conv(child_module))
    return module


# 高精度累积和计算函数
def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: ' 'its last element does not correspond to sum')
    return out


# 计算给定召回率下的FPR
def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.
    y_true = (y_true == pos_label)
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    thresholds = y_score[threshold_idxs]
    recall = tps / tps[-1]
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]
    cutoff = np.argmin(np.abs(recall - recall_level))
    return fps[cutoff] / (np.sum(np.logical_not(y_true)))


# 获取OOD检测指标（只保留需要的7个指标）
def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))  # OOD样本
    neg = np.array(_neg[:]).reshape((-1, 1))  # ID样本
    examples = np.squeeze(np.vstack((pos, neg)))

    # 计算标签
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1  # 1表示OOD，0表示ID

    # 计算AUROC
    auroc = roc_auc_score(labels, examples)

    # 计算AUPR（将OOD作为正类）
    aupr = average_precision_score(labels, examples)

    # 计算FPR95
    fpr95 = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr95


def plot_ood_distribution(tail_scores, ood_scores, save_dir=None, score_name="得分函数值", dout_name=""):
    # 确保输入是NumPy数组
    tail_scores = np.array(tail_scores)
    ood_scores = np.array(ood_scores)

    # 数据验证
    if len(tail_scores) == 0 or len(ood_scores) == 0:
        print(f"警告：尾类样本数={len(tail_scores)}，OOD样本数={len(ood_scores)}，跳过绘图")
        return

    # 计算统一的分箱
    all_scores = np.concatenate([tail_scores, ood_scores])
    data_min, data_max = np.min(all_scores), np.max(all_scores)
    range_min = min(data_min, -4)
    range_max = max(data_max, 3)
    bins = np.linspace(range_min, range_max, 51)

    # 绘制直方图
    n_tail, bins, _ = plt.hist(
        tail_scores,
        bins=bins,
        alpha=0.7,
        label='尾类分布内样本',
        color='blue',
        density=False
    )

    n_ood, bins, _ = plt.hist(
        ood_scores,
        bins=bins,
        alpha=0.7,
        label='OOD样本',
        color='orange',
        density=False
    )

    # 添加垂直参考线
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # 设置坐标轴标签
    plt.xlabel(score_name, fontsize=12)
    plt.ylabel('样本数量', fontsize=12)
    plt.xlim(-4, 3)
    max_count = max(np.max(n_tail), np.max(n_ood))
    plt.ylim(bottom=0, top=max_count * 1.1)

    # 添加图例和网格
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # 保存图像
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'distribution_{dout_name}.png')
        plt.savefig(save_path, dpi=300)
        print(f"分布图已保存至: {save_path}")

    plt.close()


# 修改：只返回需要的7个指标
def val_cifar(ood_loader, dout_name, log_file=None):
    model.eval()
    test_acc_meter = AverageMeter()
    score_list = []
    labels_list = []
    pred_list = []
    tail_scores = []  # 收集尾类样本的分数

    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.cuda(), targets.cuda()
            logits, scores = get_scores_fn(model, images)
            probs = F.softmax(logits, dim=1)
            pred = logits.data.max(1)[1]
            acc = pred.eq(targets.data).float().mean()

            # 收集所有样本信息
            score_list.append(scores.detach().cpu().numpy())
            labels_list.append(targets.detach().cpu().numpy())
            pred_list.append(pred.detach().cpu().numpy())
            test_acc_meter.append(acc.item())

            # 筛选尾类样本（样本数最少的33%为尾类）
            targets_np = targets.cpu().numpy()
            scores_np = scores.detach().cpu().numpy()
            for idx, label in enumerate(targets_np):
                if img_num_per_cls[label] <= np.percentile(img_num_per_cls, 33):
                    tail_scores.append(scores_np[idx])

    # 计算ID准确率
    test_acc = test_acc_meter.avg
    in_scores = np.concatenate(score_list, axis=0)
    in_labels = np.concatenate(labels_list, axis=0)
    in_preds = np.concatenate(pred_list, axis=0)
    many_acc, median_acc, low_acc, _ = shot_acc(in_preds, in_labels, img_num_per_cls, acc_per_cls=True)

    # 收集OOD样本分数
    ood_score_list, sc_labels_list = [], []
    with torch.no_grad():
        for images, sc_labels in ood_loader:
            images, sc_labels = images.cuda(), sc_labels.cuda()
            logits, scores = get_scores_fn(model, images)
            ood_score_list.append(scores.detach().cpu().numpy())
            sc_labels_list.append(sc_labels.detach().cpu().numpy())

    ood_scores = np.concatenate(ood_score_list, axis=0)
    sc_labels = np.concatenate(sc_labels_list, axis=0)

    # 处理SCOOD数据集中的伪OOD样本（属于ID分布的其他类）
    fake_ood_scores = ood_scores[sc_labels >= 0]
    real_ood_scores = ood_scores[sc_labels < 0]
    real_in_scores = np.concatenate([in_scores, fake_ood_scores], axis=0)

    # 计算OOD检测指标（只保留需要的7个）
    auroc, aupr, fpr95 = get_measures(real_ood_scores, real_in_scores)

    # 绘制分布图
    plot_ood_distribution(tail_scores, real_ood_scores, save_dir=save_dir, score_name="Energy得分", dout_name=dout_name)

    # 打印该数据集的结果（只包含需要的7个指标）
    result_str = f"\n=== {dout_name} ===\n"
    result_str += f"AUROC: {auroc * 100:.2f}, AUPR: {aupr * 100:.2f}, FPR95: {fpr95 * 100:.2f}\n"
    result_str += f"ACC: {test_acc * 100:.2f}, MANY: {many_acc * 100:.2f}, MEDIUM: {median_acc * 100:.2f}, FEW: {low_acc * 100:.2f}\n"

    print(result_str)

    # 将结果写入日志文件
    if log_file:
        with open(log_file, 'a') as f:
            f.write(result_str)

    return auroc, aupr, fpr95, test_acc, many_acc, median_acc, low_acc


def val_imagenet(log_file=None):
    model.eval()
    test_acc_meter = AverageMeter()
    score_list = []
    labels_list = []
    pred_list = []
    tail_scores = []

    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.cuda(), targets.cuda()
            logits, scores = get_scores_fn(model, images)
            probs = F.softmax(logits, dim=1)
            pred = logits.data.max(1)[1]
            acc = pred.eq(targets.data).float().mean()

            score_list.append(scores.detach().cpu().numpy())
            labels_list.append(targets.detach().cpu().numpy())
            pred_list.append(pred.detach().cpu().numpy())
            test_acc_meter.append(acc.item())

            targets_np = targets.cpu().numpy()
            scores_np = scores.detach().cpu().numpy()
            for idx, label in enumerate(targets_np):
                if img_num_per_cls[label] <= np.percentile(img_num_per_cls, 33):
                    tail_scores.append(scores_np[idx])

    test_acc = test_acc_meter.avg
    in_scores = np.concatenate(score_list, axis=0)
    in_labels = np.concatenate(labels_list, axis=0)
    in_preds = np.concatenate(pred_list, axis=0)
    many_acc, median_acc, low_acc = shot_acc(in_preds, in_labels, img_num_per_cls, acc_per_cls=False)

    ood_score_list = []
    with torch.no_grad():
        for images, _ in ood_loader:
            images = images.cuda()
            logits, scores = get_scores_fn(model, images)
            ood_score_list.append(scores.detach().cpu().numpy())

    ood_scores = np.concatenate(ood_score_list, axis=0)
    auroc, aupr, fpr95 = get_measures(ood_scores, in_scores)

    # 只输出需要的7个指标
    result_str = f"AUROC: {auroc * 100:.2f}, AUPR: {aupr * 100:.2f}, FPR95: {fpr95 * 100:.2f}, ACC: {test_acc * 100:.2f}, "
    result_str += f"MANY: {many_acc * 100:.2f}, MEDIUM: {median_acc * 100:.2f}, FEW: {low_acc * 100:.2f}\n"
    print(result_str)

    # 将结果写入日志文件
    if log_file:
        with torch.no_grad():
            f.write(result_str)

    return auroc, aupr, fpr95, test_acc, many_acc, median_acc, low_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a CIFAR Classifier')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--dataset', '--ds', default='cifar100', choices=['cifar10', 'cifar100', 'imagenet'],
                        help='which dataset to use')
    parser.add_argument('--data_root_path', '--drp', default='../data', help='Where you save all your datasets.')
    # 支持多个OOD数据集输入
    parser.add_argument('--dout', nargs='+', default=['texture', 'svhn', 'cifar', 'tin', 'lsun', 'places365'],
                        choices=['texture', 'svhn', 'cifar', 'tin', 'lsun', 'places365'],
                        help='which dout to use')
    parser.add_argument('--model', '--md', default='ResNet18', choices=['ResNet18', 'ResNet50'],
                        help='which model to use')
    parser.add_argument('--imbalance_ratio', '--rho', default=0.01, type=float)
    parser.add_argument('--test_batch_size', '--tb', type=int, default=256)
    parser.add_argument('--metric', default='energy', help='OOD detection metric')
    parser.add_argument('--ckpt_path',
                        default=r'D:\PyCharmWorkSpace\TSOD-main\logs\cifar100-0.01-OOD300000\ResNet18\e901-b128-adam-lr0.001-wd0.0005-cos')
    parser.add_argument('--ckpt', default='latest', choices=['latest', 'epoch'])
    parser.add_argument('--feature_dim', default=512, type=int)
    # 新增：保存结果的文件路径参数
    parser.add_argument('--result_file', default=None, help='Path to save the test results')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    save_dir = os.path.join(args.ckpt_path, 'ood_results')
    os.makedirs(save_dir, exist_ok=True)

    # 确定结果保存路径
    if args.result_file:
        result_file = args.result_file
    else:
        # 如果未指定结果文件，则使用默认路径
        result_file = os.path.join(args.ckpt_path, 'normal', 'test_results.txt')

    # 创建保存目录
    result_dir = os.path.dirname(result_file)
    os.makedirs(result_dir, exist_ok=True)

    # 初始化结果文件
    # with open(result_file, 'w') as f:
    #     f.write(f"测试配置:\n")
    #     f.write(f"数据集: {args.dataset}, 不平衡比例: {args.imbalance_ratio}, 模型: {args.model}, 指标: {args.metric}\n")
    #     f.write(f"OOD数据集: {args.dout}\n")
    #     f.write(f"{'=' * 50}\n")
    #
    # print(f"结果将保存到: {result_file}")

    # 数据预处理
    if args.dataset in ['cifar10', 'cifar100']:
        train_transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), ])
        test_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), ])
    elif args.dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        train_transform = transforms.Compose(
            [transforms.RandomResizedCrop(224, scale=(0.2, 1.0)), transforms.RandomHorizontalFlip(),
             transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
             transforms.RandomGrayscale(p=0.2), transforms.ToTensor(), transforms.Normalize(mean, std)])
        test_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                             transforms.Normalize(mean, std)])
    else:
        raise NotImplementedError()

    # 加载ID数据集
    if args.dataset == 'cifar10':
        num_classes = 10
        train_set = IMBALANCECIFAR10(train=True, transform=train_transform, imbalance_ratio=args.imbalance_ratio,
                                     root=args.data_root_path)
        test_set = IMBALANCECIFAR10(train=False, transform=test_transform, imbalance_ratio=1, root=args.data_root_path)
    elif args.dataset == 'cifar100':
        num_classes = 100
        train_set = IMBALANCECIFAR100(train=True, transform=train_transform, imbalance_ratio=args.imbalance_ratio,
                                      root=args.data_root_path)
        test_set = IMBALANCECIFAR100(train=False, transform=test_transform, imbalance_ratio=1, root=args.data_root_path)
    elif args.dataset == 'imagenet':
        num_classes = 1000
        train_set = LT_Dataset(os.path.join(args.data_root_path, 'imagenet'),
                               './datasets/ImageNet_LT/ImageNet_LT_train.txt', transform=train_transform,
                               subset_class_idx=np.arange(0, num_classes))
        test_set = LT_Dataset(os.path.join(args.data_root_path, 'imagenet'),
                              './datasets/ImageNet_LT/ImageNet_LT_test.txt', transform=test_transform,
                              subset_class_idx=np.arange(0, num_classes))
    else:
        raise NotImplementedError()

    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers,
                             drop_last=False, pin_memory=True)
    print(f'Din is {args.dataset} with {len(test_set)} images')

    # 处理CIFAR数据集的特殊替换（cifar10<->cifar100）
    if args.dataset == 'cifar10' and 'cifar' in args.dout:
        args.dout[args.dout.index('cifar')] = 'cifar100'
    elif args.dataset == 'cifar100' and 'cifar' in args.dout:
        args.dout[args.dout.index('cifar')] = 'cifar10'

    img_num_per_cls = np.array(train_set.img_num_per_cls)

    # 加载模型
    if args.model == 'ResNet18':
        device = 'cuda'
        orig_resnet = ResNet18(num_classes=num_classes, feature_dim=args.feature_dim)
        sn_resnet = apply_spectral_norm_to_conv(orig_resnet)
        model = sn_resnet.to(device)
    elif args.model == 'ResNet50':
        model = ResNet50(num_classes=num_classes).cuda()
    else:
        raise NotImplementedError()

    # 加载权重
    ckpt = torch.load(os.path.join(args.ckpt_path, 'latest.pth'), map_location="cuda:0", weights_only=False)['model']
    model.load_state_dict(ckpt, strict=False)
    model.requires_grad_(False)

    # 选择OOD检测指标函数
    if args.metric == 'msp':
        get_scores_fn = get_msp_scores
    elif args.metric == 'rep_norm':
        get_scores_fn = get_rep_norm_scores
    elif args.metric == 'energy':
        get_scores_fn = get_energy_scores
    else:
        raise NotImplementedError("The score metric is NOT IMPLEMENTED!")

    # 初始化累加变量，循环处理所有OOD数据集
    all_auroc, all_aupr, all_fpr95, all_acc, all_many, all_medium, all_few = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    num_douts = len(args.dout)

    if args.dataset in ['cifar10', 'cifar100']:
        for dout in args.dout:
            # 创建当前OOD数据集的加载器
            ood_set = SCOODDataset(
                os.path.join(args.data_root_path, 'SCOOD'),
                id_name=args.dataset,
                ood_name=dout,
                transform=test_transform
            )
            ood_loader = DataLoader(
                ood_set,
                batch_size=args.test_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                drop_last=False,
                pin_memory=True
            )
            print(f'\n加载OOD数据集: {dout}，共{len(ood_set)}张图像')

            # 评估并累加指标
            auroc, aupr, fpr95, acc, many, medium, few = val_cifar(ood_loader, dout, result_file)
            all_auroc += auroc
            all_aupr += aupr
            all_fpr95 += fpr95
            all_acc += acc
            all_many += many
            all_medium += medium
            all_few += few

        # 计算并打印平均值
        mean_results = f"\n{'=' * 50}\n"
        mean_results += f"***mean_auroc: {all_auroc / num_douts * 100:.2f}, mean_aupr: {all_aupr / num_douts * 100:.2f}, "
        mean_results += f"mean_fpr95: {all_fpr95 / num_douts * 100:.2f}\n"
        mean_results += f"mean_acc: {all_acc / num_douts * 100:.2f}, mean_many: {all_many / num_douts * 100:.2f}, "
        mean_results += f"mean_medium: {all_medium / num_douts * 100:.2f}, mean_few: {all_few / num_douts * 100:.2f}***\n"
        mean_results += f"{'=' * 50}\n"

        print(mean_results)

        # 将平均值写入结果文件
        with open(result_file, 'a') as f:
            f.write(mean_results)

    elif args.dataset == 'imagenet':
        # 处理imagenet数据集
        val_imagenet(result_file)