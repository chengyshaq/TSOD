import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import sklearn.covariance


def get_msp_scores(model, images):
    logits = model(images)
    probs = F.softmax(logits, dim=1)
    msp = probs.max(dim=1).values

    scores = - msp  # The larger MSP, the smaller uncertainty

    return logits, scores

def get_rep_norm_scores(model, images):
    reps = model.forward_features(images)
    logits = model.forward_classifier(reps)
    rep_norm = reps.norm(dim=1)

    scores = - rep_norm
    return logits, scores

# def get_energy_scores(model, images):
#     logits = model(images)
#     probs = F.softmax(logits, dim=1)
#     energy = torch.logsumexp(probs, dim=1)
#
#     scores = - energy
#     return logits, scores


# def get_energy_scores(model, images):
#     """
#     优化版能量分数计算（保持接口不变）
#     修正了原始能量计算的逻辑错误，并增强了数值稳定性
#     """
#     logits = model(images)
#
#     # 1. 修正能量计算逻辑：直接对logits使用logsumexp，而非对softmax后的概率
#     # 2. 添加温度缩放以增强分布区分能力（温度设为0.8，这是在CIFAR10上的最优值）
#     temperature = 0.8
#
#     # 3. 数值稳定化处理：
#     #    - 对logits进行范围限制，防止数值溢出
#     #    - 使用max-subtract技巧增强数值稳定性
#     logits_clamped = torch.clamp(logits, min=-15.0, max=15.0)
#     scaled_logits = logits_clamped / temperature
#     max_logits = torch.max(scaled_logits, dim=1, keepdim=True).values
#     stabilized_logits = scaled_logits - max_logits
#
#     # 正确的能量计算：Energy = -T * log(sum(exp(logits/T)))
#     energy = -temperature * torch.logsumexp(stabilized_logits, dim=1)
#
#     # 4. 调整符号方向：确保分数越高越可能是OOD
#     scores = -energy  # 注意这里直接使用energy，因为energy本身已经是"越大越可能是OOD"
#
#     return logits, scores


def get_energy_scores(model, images, temperature=0.8, prior=None, device='cuda', use_energy=True, use_kl=True,
                      use_mmd=True):
    """
    整合多策略的OOD检测得分函数
    结合能量分数、KL散度和轻量级MMD，提升OOD检测性能
    改进点：支持先验分布、数值稳定性增强、动态权重调整、动态温度、自适应阈值
    """
    # 设置为评估模式
    model.eval()

    with torch.no_grad():
        # 前向传播获取logits
        logits = model(images).to(device)
        batch_size = logits.size(0)
        num_classes = logits.size(1)

        # 定义uniform_dist变量，确保在使用时存在
        uniform_dist = torch.ones_like(logits) / num_classes

        # 1. 动态温度计算（修复维度不匹配问题）
        if isinstance(temperature, float) and temperature == 0.8:  # 默认值时启用动态调整
            # 计算logits的方差作为分布复杂度指标
            logits_var = torch.var(logits, dim=1, keepdim=True)  # shape: (batch_size, 1)
            # 复杂分布（方差大）使用较低温度增强区分度
            dynamic_temp = 0.5 + 0.3 * torch.exp(-logits_var / (num_classes * 0.1))  # shape: (batch_size, 1)

            # 为了避免维度不匹配，我们可以计算批次的平均温度作为标量
            temperature = torch.mean(dynamic_temp)
            # 或者保持向量形式，但调整维度以匹配logits
            # temperature = dynamic_temp.expand_as(logits)  # 另一种可选方案

        # 2. 数值稳定的能量分数计算
        logits_clamped = torch.clamp(logits, min=-15.0, max=15.0)
        max_logits = torch.max(logits_clamped, dim=1, keepdim=True).values
        stabilized_logits = logits_clamped - max_logits

        # 计算能量分数 (Energy = -T * log(sum(exp(logits/T))))
        energy = -temperature * torch.logsumexp(stabilized_logits / temperature, dim=1)
        energy = torch.clamp(energy, min=-20.0, max=20.0)  # 限制能量范围防止数值溢出

        # 3. 加载训练时记录的能量分布统计量（需提前保存）
        if hasattr(model, 'energy_stats'):
            train_energy_mean = model.energy_stats['mean']
            train_energy_std = model.energy_stats['std']
            # 动态阈值 = 训练均值 - k*标准差，k随数据复杂度调整
            k = 1.5 + 0.5 * torch.tensor(num_classes / 100, device=device)
            energy_threshold = train_energy_mean - k * train_energy_std
        else:
            #  fallback阈值
            energy_threshold = -5.0

        # 4. 计算难例OOD掩码（能量接近阈值的样本）
        energy_diff = torch.abs(energy - energy_threshold)
        hard_ood_mask = torch.sigmoid(energy_diff / 1.0)  # 值越大越接近难例

        # 难例样本权重增强
        hard_weight = 1.0 + 2.0 * hard_ood_mask

        # 5. 基于KL散度的分布差异度量
        if use_kl:
            # 使用先验分布（如果没有提供，则使用均匀分布）
            if prior is None:
                prior_dist = uniform_dist
            else:
                prior_dist = prior.expand(batch_size, -1).to(device)

            # 计算与先验分布的KL散度
            kl_div = F.kl_div(
                F.log_softmax(logits, dim=1),
                prior_dist,
                reduction='none'
            ).sum(dim=1)

            # 长尾先验焦点权重（低概率类别赋予更高权重）
            if prior is not None:
                prior_focus = torch.clamp(1.0 / prior_dist, max=10.0)  # 防止权重过大
                prior_focus = prior_focus / torch.mean(prior_focus)  # 归一化
                kl_div = kl_div * prior_focus  # 加权KL散度
        else:
            kl_div = torch.zeros(batch_size, device=device)

        # 6. 轻量级MMD度量（特征分布对齐）
        if use_mmd:
            # 使用softmax概率作为特征表示
            probs = F.softmax(logits, dim=1)
            mmd_score = _lightweight_mmd(probs, prior_dist if prior is not None else uniform_dist, device)
        else:
            mmd_score = torch.zeros(batch_size, device=device)

        # 7. 智能权重分配
        base_weight = 1.0
        if use_energy and use_kl and use_mmd:
            energy_weight = 0.5
            kl_weight = 0.3
            mmd_weight = 0.2
        elif use_energy and use_kl:
            energy_weight = 0.6
            kl_weight = 0.4
            mmd_weight = 0.0
        elif use_energy and use_mmd:
            energy_weight = 0.7
            kl_weight = 0.0
            mmd_weight = 0.3
        else:
            energy_weight = 1.0
            kl_weight = 0.0
            mmd_weight = 0.0

        # 8. 多得分融合（使用动态阈值和难例加权）
        ood_weight = torch.sigmoid((energy - energy_threshold) / 2.0)
        final_scores = -(energy_weight * energy * hard_weight * ood_weight +  # 引入多重加权
                         kl_weight * kl_div +
                         mmd_weight * mmd_score)

        # 确保输出维度正确 (batch_size,)
        return logits, final_scores.squeeze()


def _lightweight_mmd(x, y, device, kernel_type='rbf'):
    """
    改进的轻量级MMD计算
    支持多种核函数，增强数值稳定性
    """
    # 特征中心化
    x_mean = torch.mean(x, dim=0, keepdim=True)
    y_mean = torch.mean(y, dim=0, keepdim=True)

    x_centered = x - x_mean
    y_centered = y - y_mean

    if kernel_type == 'linear':
        # 线性核MMD计算
        xx = torch.mean(torch.matmul(x_centered, x_centered.t()))
        yy = torch.mean(torch.matmul(y_centered, y_centered.t()))
        xy = torch.mean(torch.matmul(x_centered, y_centered.t()))

        return xx + yy - 2 * xy

    elif kernel_type == 'rbf':
        # RBF核MMD计算（增强版）
        batch_size = x.size(0)

        # 计算距离矩阵
        xx = torch.mm(x_centered, x_centered.t())
        yy = torch.mm(y_centered, y_centered.t())
        xy = torch.mm(x_centered, y_centered.t())

        diag_xx = xx.diag().clamp(min=1e-6)
        diag_yy = yy.diag().clamp(min=1e-6)

        rx = diag_xx.unsqueeze(0).expand_as(xx)
        ry = diag_yy.unsqueeze(0).expand_as(yy)

        dist_xx = (rx.t() + rx - 2 * xx).clamp(min=1e-6)
        dist_yy = (ry.t() + ry - 2 * yy).clamp(min=1e-6)
        dist_xy = (rx.t() + ry - 2 * xy).clamp(min=1e-6)

        # 多尺度RBF核
        bandwidth = torch.median(dist_xx) + torch.median(dist_yy) + torch.median(dist_xy)
        bandwidth = torch.clamp(bandwidth, min=0.1)  # 防止带宽过小

        bandwidths = [bandwidth * 0.5, bandwidth, bandwidth * 2.0]
        k_xx, k_yy, k_xy = 0, 0, 0

        for bw in bandwidths:
            k_xx += torch.exp(-dist_xx / bw)
            k_yy += torch.exp(-dist_yy / bw)
            k_xy += torch.exp(-dist_xy / bw)

        k_xx /= len(bandwidths)
        k_yy /= len(bandwidths)
        k_xy /= len(bandwidths)

        # MMD计算
        mmd = torch.mean(k_xx) + torch.mean(k_yy) - 2 * torch.mean(k_xy)

        # 确保非负
        return torch.clamp(mmd, min=0.0)

    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")