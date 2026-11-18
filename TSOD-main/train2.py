    # 修复的强数据增强函数（解决设备不匹配问题）
    def strong_augment(x, p=0.5):
        """
        针对少数类的强数据增强组合
        包含：随机水平翻转、随机裁剪、随机擦除、亮度/对比度调整
        x: 输入图像张量 (C, H, W)
        """
        # 获取输入张量的设备，确保所有新生成的张量都在同一设备上
        device = x.device

        # 随机水平翻转
        if torch.rand(1, device=device) < p:  # 在同一设备上生成随机数
            x = torch.flip(x, dims=[2])

        # 随机裁剪（带填充）
        if torch.rand(1, device=device) < p:
            h, w = x.shape[1], x.shape[2]
            # 在同一设备上生成随机数
            crop_scale = 0.8 + torch.rand(1, device=device) * 0.2
            crop_size = int(min(h, w) * crop_scale.item())  # 80%-100%
            x = F.interpolate(
                F.adaptive_max_pool2d(x.unsqueeze(0), (crop_size, crop_size)),
                size=(h, w)
            ).squeeze(0)

        # 随机擦除
        if torch.rand(1, device=device) < p:
            h, w = x.shape[1], x.shape[2]
            # 在同一设备上生成随机数
            erase_area = (torch.rand(1, device=device) * 0.15 + 0.05).item()  # 5%-20%面积
            ratio = 0.5 + torch.rand(1, device=device) * 0.5
            erase_h = int(h * np.sqrt(erase_area) * ratio.item())
            erase_w = int(w * np.sqrt(erase_area) / ratio.item())
            x1 = max(0, int(torch.rand(1, device=device) * (h - erase_h)))
            y1 = max(0, int(torch.rand(1, device=device) * (w - erase_w)))
            # 确保填充的随机数张量与x在同一设备
            x[x1:x1 + erase_h, y1:y1 + erase_w] = torch.randn_like(
                x[x1:x1 + erase_h, y1:y1 + erase_w],
                device=device
            ) * 0.1

        # 亮度/对比度调整（核心修复部分）
        if torch.rand(1, device=device) < p:
            # 在同一设备上生成亮度和对比度参数
            brightness = 0.8 + torch.rand(1, device=device) * 0.4  # 0.8-1.2
            contrast = 0.8 + torch.rand(1, device=device) * 0.4  # 0.8-1.2
            # 确保所有运算都在同一设备上进行
            x = x * brightness - 0.5 * (brightness - 1)
            x = (x - x.mean()) * contrast + x.mean()
            x = torch.clamp(x, 0, 1)  # 保持像素值范围

        return x

    # 补充2：生成类别语义相似度矩阵（示例实现）
    def generate_semantic_similarity(num_classes, seed=42):
        """
        生成示例类别语义相似度矩阵
        实际应用中应替换为基于真实语义（如词向量）的相似度计算
        """
        np.random.seed(seed)
        # 生成随机对称矩阵作为示例（实际应使用真实语义关系）
        sem_sim = np.random.rand(num_classes, num_classes)
        sem_sim = (sem_sim + sem_sim.T) / 2  # 保证对称性
        np.fill_diagonal(sem_sim, 1.0)  # 自身相似度为1
        return torch.tensor(sem_sim, dtype=torch.float32)

    # 改进的三元组损失函数
    def batch_hard_triplet_loss(embeddings, labels, logits,
                                margin=1.0, squared=False, use_angular=False,
                                min_samples_per_class=2):
        batch_size = embeddings.size(0)
        if batch_size < 4 or len(torch.unique(labels)) < 2:
            return torch.tensor(0.0, device=embeddings.device)

        probs = F.softmax(logits, dim=1)
        confidences = torch.gather(probs, 1, labels.unsqueeze(1)).squeeze()
        confidences = torch.clamp(confidences, min=0.01, max=0.99)

        # 多区间权重分配
        mask_low = (confidences > 0.1) & (confidences <= 0.2)
        mask_mid = (confidences > 0.2) & (confidences < 0.6)
        mask_high = (confidences >= 0.6) & (confidences < 0.8)
        weights = torch.where(
            mask_low, 1.8 - confidences,
            torch.where(
                mask_mid, 1.5 - confidences,
                torch.where(
                    mask_high, 1.2 - (confidences - 0.6) / 0.2 * 0.2,
                    torch.tensor(1.0, device=embeddings.device)
                )
            )
        )
        weights = torch.clamp(weights, 0.8, 1.8)

        # 距离矩阵计算
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        dist_matrix = torch.clamp(dist_matrix, min=1e-8)
        if squared:
            dist_matrix = dist_matrix ** 2

        # 标签匹配矩阵
        labels_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)
        mask_pos = labels_matrix.float() - torch.eye(batch_size, device=embeddings.device)

        # 单样本类别处理
        class_sample_counts = torch.bincount(labels, minlength=torch.max(labels) + 1).to(embeddings.device)
        single_sample_classes = (class_sample_counts < min_samples_per_class)

        if torch.any(single_sample_classes):
            pos_dist_matrix = torch.zeros_like(dist_matrix)
            for i in range(batch_size):
                c = labels[i]
                if single_sample_classes[c] and torch.sum(labels != c) > 0:
                    other_class_mask = (labels != c).float()
                    pos_dist_matrix[i] = torch.min(dist_matrix[i] * other_class_mask + 1e9 * (1 - other_class_mask))
                else:
                    pos_dist_matrix[i] = dist_matrix[i]
            hardest_positive_dist = torch.max(pos_dist_matrix * mask_pos - 1e9 * (1 - mask_pos), dim=1)[0]
        else:
            hardest_positive_dist = torch.max(dist_matrix * mask_pos - 1e9 * (1 - mask_pos), dim=1)[0]

        # 难负样本计算
        mask_neg = (1 - labels_matrix.float())
        hardest_negative_dist = torch.min(dist_matrix * mask_neg + 1e9 * labels_matrix.float(), dim=1)[0]

        # 角度损失与动态margin
        if use_angular:
            dot_product = torch.matmul(embeddings, embeddings.transpose(0, 1))
            norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
            norms = torch.clamp(norms, min=1e-8)
            cos_sim = dot_product / (norms * norms.transpose(0, 1) + 1e-8)
            cos_sim = torch.clamp(cos_sim, -1 + 1e-6, 1 - 1e-6)
            angles = torch.acos(cos_sim)

            hard_ratio = torch.mean((hardest_positive_dist > torch.mean(hardest_positive_dist)).float())
            angular_margin = 0.65 - 0.15 * min(1.0, torch.mean(hardest_positive_dist) / 3.5)
            angular_margin = angular_margin + 0.1 * hard_ratio

            pos_angles = torch.max(angles * mask_pos - 1e9 * (1 - mask_pos), dim=1)[0]
            neg_angles = torch.min(angles * mask_neg + 1e9 * labels_matrix.float(), dim=1)[0]
            angular_loss = torch.clamp(pos_angles - neg_angles + angular_margin, min=0.0)
            angular_loss = torch.mean(angular_loss * weights)

            triplet_base = torch.clamp(hardest_positive_dist - hardest_negative_dist + margin, min=0.0)
            triplet_base = torch.mean(triplet_base * weights)
            triplet_loss = 0.5 * triplet_base + 0.5 * angular_loss
        else:
            adaptive_margin = margin + 0.2 * torch.mean(hardest_positive_dist - hardest_negative_dist)
            triplet_base = torch.clamp(hardest_positive_dist - hardest_negative_dist + adaptive_margin, min=0.0)
            triplet_loss = torch.mean(triplet_base * weights)

        # 正则化
        reg_loss = torch.mean(torch.norm(embeddings, p=2, dim=1))
        triplet_loss = triplet_loss + 0.0001 * reg_loss

        return triplet_loss

    # 改进的类别权重计算
    def compute_cb_class_weights(train_set, num_classes=100, beta=0.999, device='cuda', eps=1e-8,
                                 use_relation=True,
                                 relation_scale=0.1,
                                 max_weight_scale=2.2,
                                 transition_epoch=40, current_epoch=0, adaptive_amplify=True,
                                 focus_threshold=0.1, seed=42, cache_dir=None, sem_sim=None):
        torch.manual_seed(seed)

        # 缓存逻辑
        if cache_dir is not None:
            cache_path = os.path.join(cache_dir, f"class_counts_{len(train_set)}_{num_classes}.pkl")
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        class_counts = pickle.load(f)
                    print(f"Loaded class counts from cache: {cache_path}")
                except:
                    print("Failed to load cache, recomputing class counts")
                    class_counts = _compute_class_counts(train_set, num_classes, device)
                    _save_class_counts(class_counts, cache_path)
            else:
                class_counts = _compute_class_counts(train_set, num_classes, device)
                _save_class_counts(class_counts, cache_path)
        else:
            class_counts = _compute_class_counts(train_set, num_classes, device)

        # 零样本处理
        zero_mask = (class_counts == 0)
        if torch.any(zero_mask):
            print(f"Warning: {zero_mask.sum().item()} classes have zero samples")
            class_counts = class_counts.masked_fill(zero_mask, eps)

        # 分布分析
        q1 = torch.quantile(class_counts, 0.25)
        distribution_entropy = -torch.sum(
            class_counts / class_counts.sum() * torch.log(class_counts / class_counts.sum() + eps))

        # 动态beta参数
        entropy_ratio = distribution_entropy / np.log(num_classes)
        dynamic_beta = beta + (0.9995 - beta) * (1 - entropy_ratio)
        class_counts_log = torch.log(class_counts + 1) + 0.5
        effective_num = 1.0 - torch.pow(dynamic_beta, class_counts_log)
        weights = (1.0 - dynamic_beta) / effective_num

        # 自适应放大机制
        if adaptive_amplify:
            dynamic_threshold = max(0.05, focus_threshold * (1.7 - min(1.0, current_epoch / transition_epoch)))
            tail_index = torch.exp(-class_counts / q1)

            if current_epoch < transition_epoch:
                amplification_factor = torch.clamp(
                    1.4 * torch.log(2.8 / tail_index),
                    min=1.0, max=max_weight_scale * (1.0 + 0.5 * (1 - distribution_entropy / np.log(num_classes)))
                )
            else:
                amplification_factor = torch.clamp(
                    1.2 * torch.log(2.5 / tail_index),
                    min=1.0,
                    max=max_weight_scale * 0.95 * (1.0 + 0.4 * (1 - distribution_entropy / np.log(num_classes)))
                )

            focus_mask = (class_counts < q1 * dynamic_threshold).float()
            extreme_tail_mask = (class_counts < q1 * dynamic_threshold * 0.5).float()
            focus_boost = 1.0 + focus_mask * 0.7 * (1.0 - min(1.0, current_epoch / (transition_epoch * 1.5)))
            extreme_boost = 1.0 + extreme_tail_mask * 1.0 * (1.0 - min(1.0, current_epoch / transition_epoch))
            amplification_factor = amplification_factor * focus_boost * extreme_boost
            weights = weights * amplification_factor

        # 类间关系处理
        if use_relation and class_counts.shape[0] > 1:
            if current_epoch > transition_epoch * 0.5:
                relation_matrix = torch.log(
                    torch.clamp(class_counts.unsqueeze(1) / class_counts.unsqueeze(0), min=0.1, max=10.0)
                )
                relation_strength = torch.mean(torch.abs(relation_matrix), dim=1)
                relation_progress = min(1.0, (current_epoch - transition_epoch * 0.5) / (transition_epoch * 0.5))
                relation_weights = 1.0 + relation_scale * relation_strength * (1.0 - relation_progress)
                weights = weights * relation_weights
            else:
                relation_weights = torch.ones_like(weights)

        # 平滑过渡
        if transition_epoch > 0 and current_epoch < transition_epoch:
            progress = current_epoch / transition_epoch
            original_weights = (1.0 - dynamic_beta) / (1.0 - torch.pow(dynamic_beta, class_counts))
            smooth_factor = torch.sigmoid(torch.tensor(progress * 5 - 2.5, device=device))
            weights = torch.lerp(original_weights * 0.85 + weights * 0.15, weights, smooth_factor)

        # 权重裁剪与零样本处理
        max_weight = 12.0 * (1.0 - dynamic_beta) / (1.0 - torch.pow(dynamic_beta, torch.max(class_counts_log)))
        weights = torch.clamp(weights, min=1e-4, max=max_weight)
        weights = weights / torch.sum(weights) * num_classes

        # 零样本类别权重修正
        if torch.any(zero_mask) and sem_sim is not None:
            zero_classes = torch.where(zero_mask)[0]
            for c in zero_classes:
                similar_classes = torch.argsort(sem_sim[c])[-3:]
                similar_weights = weights[similar_classes[~zero_mask[similar_classes]]]
                if len(similar_weights) > 0:
                    weights[c] = torch.mean(similar_weights) * 0.8
                else:
                    weights[c] = torch.mean(weights) * 0.5
        elif torch.any(zero_mask):
            weights = weights.masked_fill(zero_mask, 1e-4)

        return weights.to(device)

    # 辅助函数：计算类别样本数（假设之前未实现）
    def _compute_class_counts(dataset, num_classes, device):
        """计算每个类别的样本数量"""
        counts = torch.zeros(num_classes, device=device)
        # 假设dataset的targets属性包含所有标签
        for label in dataset.targets:
            counts[label] += 1
        return counts

    # 辅助函数：保存类别计数（假设之前未实现）
    def _save_class_counts(counts, path):
        """保存类别计数到缓存文件"""
        with open(path, 'wb') as f:
            pickle.dump(counts.cpu(), f)

    # 动态权重计算函数（假设之前未实现）
    def compute_dynamic_weight(current_step, total_steps, initial_weight=0.0, final_weight=1.0, curve='cosine'):
        """计算动态权重，支持余弦曲线和线性曲线"""
        progress = min(1.0, current_step / total_steps)
        if curve == 'cosine':
            # 余弦曲线：前期增长慢，中期快，后期平缓
            return initial_weight + (final_weight - initial_weight) * (1 - np.cos(progress * np.pi / 2))
        else:
            # 线性曲线
            return initial_weight + (final_weight - initial_weight) * progress





    def ood_detection_loss_improved(ood_logits, prior, temperature=1.0, use_energy=True, use_mmd=True,
                                    epoch=0, max_epochs=200, device='cuda', alpha=0.8, gamma=0.7, seed=42,
                                    ood_cache=None, cache_threshold=0.9, class_centers=None, features=None,
                                    labels=None):
        """增强版OOD检测损失函数，优化稳定性、检测能力和计算效率"""
        torch.manual_seed(seed)

        batch_size = ood_logits.size(0)
        num_classes = ood_logits.size(1)

        # 1. 自适应温度调节
        temperature = _adaptive_temperature(epoch, max_epochs, temperature)

        # 2. 增强型能量损失
        if use_energy:
            # 对数几率稳定化
            logits_clamped = torch.clamp(ood_logits, min=-20.0, max=20.0)
            max_logits = torch.max(logits_clamped, dim=1, keepdim=True).values
            stabilized_logits = logits_clamped - max_logits

            # 能量计算与范围限制
            energy = -temperature * torch.logsumexp(stabilized_logits / temperature, dim=1)
            energy = torch.clamp(energy, min=-50.0, max=0.0)

            # 动态能量阈值
            energy_threshold = _dynamic_energy_threshold(epoch, max_epochs, device, ood_cache, cache_threshold)

            # 双曲正切焦点权重
            energy_diff = energy_threshold - energy
            focus_weight = 1.0 + gamma * torch.tanh(energy_diff / temperature * 1.5) * 1.8

            # FPR95优化
            edge_mask = ((energy > energy_threshold - 2.0) & (energy < energy_threshold)).float()
            edge_boost = 1.0 + 3.5 * (1.0 - (energy - (energy_threshold - 2.0)) / 2.0)
            focus_weight = focus_weight * (1.0 + 0.8 * edge_mask * edge_boost)

            # 类别平衡能量损失
            if prior is not None:
                class_weights = 1.0 / (prior + 1e-6)
                class_weights = class_weights / torch.sum(class_weights) * num_classes
                pred_classes = torch.argmax(ood_logits, dim=1)
                sample_weights = class_weights[pred_classes]
                sample_weights = torch.clamp(sample_weights, min=0.5, max=2.5)
                energy_loss = torch.mean(
                    focus_weight * energy * (1 + 0.9 * (energy < energy_threshold).float()) * sample_weights)
            else:
                energy_loss = torch.mean(focus_weight * energy * (1 + 0.9 * (energy < energy_threshold).float()))
        else:
            energy_loss = torch.tensor(0.0, device=device)

        # 3. 双重KL散度损失
        ood_probs = F.softmax(ood_logits, dim=1)
        uniform_dist = torch.ones_like(ood_probs) / num_classes

        # 原始KL散度
        log_ood_probs = F.log_softmax(ood_logits, dim=1)
        kl_div_uniform = F.kl_div(log_ood_probs, uniform_dist, reduction='batchmean')

        # 平滑处理先验分布
        prior_smooth = (prior + 1e-8) / (1 + num_classes * 1e-8)
        prior_dist = prior_smooth.expand(batch_size, -1)

        # 长尾感知KL散度
        kl_div_prior = F.kl_div(log_ood_probs, prior_dist, reduction='batchmean')

        # 动态融合两种KL散度
        kl_weight = min(1.0, epoch / (max_epochs * 0.6))
        kl_div = (1 - kl_weight) * kl_div_uniform + kl_weight * kl_div_prior

        # 4. 增强型先验分布匹配
        entropy = -torch.sum(ood_probs * torch.log(ood_probs + 1e-6), dim=1)

        # 动态置信度阈值
        confident_threshold = 2.9 - 0.9 * min(1.0, epoch / (max_epochs * 0.75))
        confident_mask = (entropy < confident_threshold).float().unsqueeze(1)

        # 焦点先验匹配
        prior_match = -torch.sum(ood_probs * confident_mask * torch.clamp(prior_dist.log(), min=-15.0), dim=1)

        # 改进的指数函数稳定性
        exp_term = torch.exp(alpha * (confident_threshold - entropy))
        exp_term = torch.clamp(exp_term, max=50.0)
        prior_match = torch.mean(exp_term * prior_match)

        # 熵正则化
        entropy_reg = torch.mean(entropy)

        # 5. 改进型MMD损失
        if use_mmd and batch_size > 10:
            if batch_size > 256:
                indices = torch.randperm(batch_size)[:256]
                ood_probs_sampled = ood_probs[indices]
                hard_ood_mask_sampled = (energy < energy_threshold).float()[indices] if use_energy else None
            else:
                ood_probs_sampled = ood_probs
                hard_ood_mask_sampled = (energy < energy_threshold).float() if use_energy else None

            mmd_loss = _compute_mmd_stable_advanced_optimized(
                ood_probs_sampled, prior_dist[:len(ood_probs_sampled)], num_classes, device,
                epoch=epoch, max_epochs=max_epochs,
                hard_ood_mask=hard_ood_mask_sampled,
                kl_div=kl_div, seed=seed
            )
        else:
            mmd_loss = torch.tensor(0.0, device=device)

        # 6. 对比学习损失
        if class_centers is not None and features is not None and labels is not None:
            ood_mask = (energy < energy_threshold).float() if use_energy else torch.zeros(batch_size, device=device)
            contrastive_loss = contrastive_ood_loss(features, labels, class_centers, ood_mask)
        else:
            contrastive_loss = torch.tensor(0.0, device=device)

        # 7. 自适应动态权重调整
        phase = min(4, int(epoch / (max_epochs * 0.25)))
        weights = _get_phase_weights(phase)

        ood_kl_weight, ood_prior_weight, ood_energy_weight, ood_mmd_weight, entropy_reg_weight = weights

        # 8. 组合损失
        total_loss = (ood_kl_weight * kl_div +
                      ood_prior_weight * (prior_match - entropy_reg_weight * entropy_reg) +
                      ood_energy_weight * energy_loss +
                      ood_mmd_weight * mmd_loss +
                      0.1 * contrastive_loss)  # 对比损失权重

        # 9. 智能NaN保护
        if torch.isnan(total_loss):
            fallback_weights = [0.4, 0.3, 0.3, 0.0, 0.0]
            return (fallback_weights[0] * kl_div +
                    fallback_weights[1] * prior_match +
                    fallback_weights[2] * energy_loss)

        # 10. 分类-OOD平衡项
        if epoch > max_epochs * 0.3:
            pred_confidence = torch.max(ood_probs, dim=1)[0]
            balance_term = -0.01 * torch.mean(torch.log(pred_confidence + 1e-6))
            total_loss = total_loss + balance_term

        return total_loss

    def _adaptive_temperature(epoch, max_epochs, initial_temp=1.0):
        """增强版自适应温度调节，优化OOD分布学习"""
        progress = min(1.0, epoch / (max_epochs * 0.95))
        temp = initial_temp * (0.5 + 0.5 * np.cos(np.pi * progress))
        return max(0.3, min(temp, 2.0))

    def _dynamic_energy_threshold(epoch, max_epochs, device, ood_cache=None, cache_threshold=0.9):
        """增强版动态能量阈值，基于训练进度和OOD样本分布自适应调整"""
        if ood_cache is not None and len(ood_cache) > 100:
            ood_energies = torch.tensor([e for e in ood_cache if e < 0], device=device)
            if len(ood_energies) > 50:
                percentile_90 = torch.quantile(ood_energies, 0.9)
                base_threshold = percentile_90
            else:
                base_threshold = -7.0
        else:
            base_threshold = -7.0

        # 基于训练阶段微调阈值
        if epoch < max_epochs * 0.2:
            return torch.tensor(base_threshold * 0.8, device=device)
        elif epoch < max_epochs * 0.5:
            return torch.tensor(base_threshold, device=device)
        elif epoch < max_epochs * 0.8:
            return torch.tensor(base_threshold * 1.1, device=device)
        else:
            return torch.tensor(base_threshold * 1.2, device=device)  # 提高阈值以减少FPR95

    def _get_phase_weights(phase):
        """增强版五阶段训练权重策略，优化OOD检测与分类的平衡"""
        if phase == 0:  # 阶段1：预热
            return [0.45, 0.1, 0.35, 0.1, 0.0]
        elif phase == 1:  # 阶段2：分布对齐
            return [0.35, 0.2, 0.3, 0.15, 0.05]
        elif phase == 2:  # 阶段3：能量优化
            return [0.1, 0.15, 0.65, 0.1, 0.0]
        elif phase == 3:  # 阶段4：精细调整
            return [0.15, 0.35, 0.35, 0.15, 0.0]
        else:  # 阶段5：收敛
            return [0.25, 0.4, 0.25, 0.1, 0.0]

    def _compute_mmd_stable_advanced_optimized(x, y, num_classes, device, kernel_mul=2.0, kernel_num=3, min_samples=10,
                                               epoch=0, max_epochs=200, hard_ood_mask=None, kl_div=None, seed=42):
        """增强版MMD计算，优化分布匹配能力、数值稳定性和计算效率"""
        torch.manual_seed(seed)

        batch_size = x.size(0)
        if batch_size < min_samples:
            return torch.tensor(0.0, device=device)

        # 1. 特征白化
        x_centered = x - torch.mean(x, dim=0, keepdim=True)
        y_centered = y - torch.mean(y, dim=0, keepdim=True)

        # 计算协方差矩阵并白化
        cov = torch.mm(x_centered.t(), x_centered) / (batch_size - 1)

        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(cov, UPLO='U')
            eigenvalues = torch.clamp(eigenvalues, min=1e-10)
            whitening_matrix = torch.mm(eigenvectors, torch.diag(1.0 / torch.sqrt(eigenvalues)))
            x_whitened = torch.mm(x_centered, whitening_matrix)
            y_whitened = torch.mm(y_centered, whitening_matrix)
        except:
            x_whitened = F.normalize(x_centered, p=2, dim=1)
            y_whitened = F.normalize(y_centered, p=2, dim=1)

        # 2. 难例样本加权
        if hard_ood_mask is not None:
            x_weights = torch.ones(batch_size, device=device)
            y_weights = hard_ood_mask + 1.5  # 增加OOD样本权重
            x_weights = x_weights / torch.sum(x_weights)
            y_weights = y_weights / torch.sum(y_weights)
        else:
            x_weights = y_weights = torch.ones(batch_size, device=device) / batch_size

        # 3. 安全的欧氏距离计算
        xx = torch.mm(x_whitened, x_whitened.t())
        yy = torch.mm(y_whitened, y_whitened.t())
        xy = torch.mm(x_whitened, y_whitened.t())

        diag_xx = xx.diag().clamp(min=1e-10)
        diag_yy = yy.diag().clamp(min=1e-10)

        rx = diag_xx.unsqueeze(0).expand_as(xx)
        ry = diag_yy.unsqueeze(0).expand_as(yy)

        dist_xx = (rx.t() + rx - 2 * xx).clamp(min=1e-10)
        dist_yy = (ry.t() + ry - 2 * yy).clamp(min=1e-10)
        dist_xy = (rx.t() + ry - 2 * xy).clamp(min=1e-10)

        # 4. 自适应带宽计算
        all_dists = torch.cat([dist_xx.view(-1), dist_yy.view(-1), dist_xy.view(-1)])
        q1 = torch.quantile(all_dists, 0.2)
        q3 = torch.quantile(all_dists, 0.8)
        bandwidth = (q3 - q1) / 1.2
        bandwidth = torch.clamp(bandwidth, min=0.2, max=num_classes * 0.3)

        # 5. 核函数优化
        kernel_list = ['rbf', 'linear']
        kernel_val_xx, kernel_val_yy, kernel_val_xy = [], [], []

        for kernel_type in kernel_list:
            if kernel_type == 'rbf':
                # 多尺度RBF核
                bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
                rbf_kernels_xx = []
                rbf_kernels_yy = []
                rbf_kernels_xy = []

                for bw in bandwidth_list:
                    scaled_dist_xx = dist_xx / bw
                    scaled_dist_yy = dist_yy / bw
                    scaled_dist_xy = dist_xy / bw

                    # 指数运算安全限制
                    capped_xx = torch.clamp(scaled_dist_xx, max=40.0)
                    capped_yy = torch.clamp(scaled_dist_yy, max=40.0)
                    capped_xy = torch.clamp(scaled_dist_xy, max=40.0)

                    k_xx = torch.exp(-capped_xx)
                    k_yy = torch.exp(-capped_yy)
                    k_xy = torch.exp(-capped_xy)

                    rbf_kernels_xx.append(k_xx)
                    rbf_kernels_yy.append(k_yy)
                    rbf_kernels_xy.append(k_xy)

                # 多尺度RBF核合并
                k_xx = sum(rbf_kernels_xx) / len(rbf_kernels_xx)
                k_yy = sum(rbf_kernels_yy) / len(rbf_kernels_yy)
                k_xy = sum(rbf_kernels_xy) / len(rbf_kernels_xy)

                # 应用样本权重
                k_xx = k_xx * x_weights.unsqueeze(1) * x_weights.unsqueeze(0)
                k_yy = k_yy * y_weights.unsqueeze(1) * y_weights.unsqueeze(0)
                k_xy = k_xy * x_weights.unsqueeze(1) * y_weights.unsqueeze(0)

            elif kernel_type == 'linear':
                k_xx = torch.mm(x_whitened, x_whitened.t()) * x_weights.unsqueeze(1) * x_weights.unsqueeze(0)
                k_yy = torch.mm(y_whitened, y_whitened.t()) * y_weights.unsqueeze(1) * y_weights.unsqueeze(0)
                k_xy = torch.mm(x_whitened, y_whitened.t()) * x_weights.unsqueeze(1) * y_weights.unsqueeze(0)

            kernel_val_xx.append(k_xx)
            kernel_val_yy.append(k_yy)
            kernel_val_xy.append(k_xy)

        # 6. 核权重动态计算
        kernel_importances = _get_kernel_importance_gradient(kernel_val_xx, kernel_val_yy, kernel_val_xy, kl_div, epoch,
                                                             max_epochs)

        # 7. MMD计算
        kernel_xx = sum(k * w for k, w in zip(kernel_val_xx, kernel_importances)) / len(kernel_importances)
        kernel_yy = sum(k * w for k, w in zip(kernel_val_yy, kernel_importances)) / len(kernel_importances)
        kernel_xy = sum(k * w for k, w in zip(kernel_val_xy, kernel_importances)) / len(kernel_importances)

        mmd_loss = (torch.sum(kernel_xx) +
                    torch.sum(kernel_yy) -
                    2 * torch.sum(kernel_xy))

        # 8. 动态范围约束
        mmd_clamp_min = 0.0
        mmd_clamp_max = max(8.0, num_classes * 0.35)

        return torch.clamp(mmd_loss, min=mmd_clamp_min, max=mmd_clamp_max)

    def _get_kernel_importance_gradient(kernel_val_xx, kernel_val_yy, kernel_val_xy, kl_div, epoch, max_epochs):
        """增强版核权重计算，基于训练阶段和分布特性"""
        progress = min(1.0, epoch / (max_epochs * 0.9))

        # 动态调整各核权重
        linear_weight = 1.3 - progress
        rbf_weight = 0.3 + 0.7 * progress

        # 归一化权重
        total = rbf_weight + linear_weight
        return [rbf_weight / total, linear_weight / total]

    def center_loss(features, labels, centers, alpha=0.5):
        """计算中心损失并更新类中心"""
        batch_size = features.size(0)
        features_dim = features.size(1)
        labels_expand = labels.unsqueeze(1).expand(batch_size, features_dim)
        centers_batch = centers.gather(0, labels_expand)
        loss = F.mse_loss(features, centers_batch)

        # 更新类中心
        diff = centers_batch - features
        unique_label, unique_idx, unique_count = np.unique(labels.cpu().numpy(), return_inverse=True,
                                                           return_counts=True)
        appear_times = torch.from_numpy(unique_count).gather(0, torch.from_numpy(unique_idx)).float().unsqueeze(1).to(
            features.device)
        diff = diff / (1 + appear_times)
        diff = alpha * diff
        centers.scatter_add_(0, labels_expand, -diff)

        return loss

    def contrastive_ood_loss(features, labels, class_centers, ood_mask, temperature=0.1):
        """对比学习损失，拉近ID样本与类中心，推远OOD样本"""
        # 计算样本与类中心的相似度
        sim_matrix = torch.matmul(features, class_centers.t()) / temperature
        sim_matrix = torch.exp(sim_matrix)

        # 对ID样本，正样本为同类中心；对OOD样本，正样本为所有中心的负平均
        pos_mask = torch.zeros_like(sim_matrix)
        for i in range(len(labels)):
            if ood_mask[i] == 0:  # ID样本
                pos_mask[i, labels[i]] = 1.0
            else:  # OOD样本
                pos_mask[i] = -1.0 / class_centers.size(0)

        # 计算InfoNCE损失
        pos_sim = torch.sum(sim_matrix * pos_mask, dim=1)
        neg_sim = torch.sum(sim_matrix * (1 - pos_mask), dim=1)
        loss = -torch.mean(torch.log(pos_sim / (pos_sim + neg_sim) + 1e-8))

        return loss

    def focal_loss(logits, labels, alpha=0.25, gamma=2.0):
        """Focal Loss，降低对易分类样本的关注"""
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()

    def label_smoothing_loss(logits, labels, epsilon=0.1):
        """标签平滑损失，增强泛化能力"""
        n_classes = logits.size(1)
        one_hot = torch.zeros_like(logits).scatter(1, labels.unsqueeze(1), 1)
        smoothed_label = one_hot * (1 - epsilon) + epsilon / n_classes
        loss = -(smoothed_label * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        return loss

    MAX_OOD_WEIGHT = 0.15
    MIN_DYNAMIC_EPOCHS = 5

    # 训练主循环
    for epoch in range(args.epochs):
        if args.ddp:
            train_sampler.set_epoch(epoch)

        model.train()

        training_loss_sum = 0.0
        training_loss_LA_sum = 0.0
        training_loss_triplet_sum = 0.0
        training_loss_ood_sum = 0.0
        batch_count = 0

        current_lr = scheduler.get_last_lr()[0]

        # ------------------- 损失权重动态调度（对齐旧成功版本思路） -------------------
        # 三元组延后启动，改用linear慢速上升，降低峰值
        triplet_start_epoch = int(args.epochs * 0.4)
        max_triplet_epochs = max(MIN_DYNAMIC_EPOCHS, int(args.epochs * 0.25))
        if epoch < triplet_start_epoch:
            triplet_weight = 0.0
        elif epoch < triplet_start_epoch + max_triplet_epochs:
            triplet_weight = compute_dynamic_weight(
                epoch - triplet_start_epoch, max_triplet_epochs,
                initial_weight=0.0, final_weight=0.4,
                curve='linear'  # 关键：从 exponential → linear，避免权重突然冲上去
            )
        else:
            triplet_weight = 0.4

        # OOD保持原设置不变
        ood_start_epoch = int(args.epochs * 0.5)
        max_ood_epochs = max(MIN_DYNAMIC_EPOCHS, int(args.epochs * 0.3))
        if epoch < ood_start_epoch:
            ood_weight = 0.0
        elif epoch < ood_start_epoch + max_ood_epochs:
            ood_weight = compute_dynamic_weight(
                epoch - ood_start_epoch, max_ood_epochs,
                initial_weight=0.0, final_weight=MAX_OOD_WEIGHT,
                curve='linear'
            )
        else:
            ood_weight = MAX_OOD_WEIGHT

        for batch_idx, ((in_data, labels), (ood_data, _)) in enumerate(zip(train_loader, ood_loader)):
            in_data, labels = in_data.to(device), labels.to(device)
            ood_data = ood_data.to(device)

            N_in = len(labels)
            all_data = torch.cat([in_data, ood_data], dim=0)

            all_logits, all_reps = model(all_data)
            in_logits = all_logits[:N_in]
            in_reps = all_reps[:N_in]
            ood_logits = all_logits[N_in:]

            adjusted_in_logits = in_logits + args.tau * prior.log()[None, :]
            LA_loss = F.cross_entropy(adjusted_in_logits, labels)

            # ========== 重点改动：margin固定，不再随epoch放大 ==========
            if len(in_reps) > 1 and len(torch.unique(labels)) > 1:
                margin = args.margin
                triplet_loss = batch_hard_triplet_loss(
                    in_reps, labels, logits=in_logits, margin=margin
                )
            else:
                triplet_loss = torch.tensor(0.0, device=device)

            ood_loss = ood_detection_loss_improved(ood_logits, prior, temperature=args.temperature)

            loss = LA_loss + triplet_weight * triplet_loss + ood_weight * ood_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss_sum += loss.item()
            training_loss_LA_sum += LA_loss.item()
            training_loss_triplet_sum += triplet_loss.item()
            training_loss_ood_sum += ood_loss.item()
            batch_count += 1

            if batch_idx % 100 == 0:
                train_str = 'epoch %d batch %d (train): ' \
                            'loss %.4f (LA: %.4f, Triplet: %.4f, OOD: %.4f) | ' \
                            'lr %.6f, triplet_w: %.4f, ood_w: %.4f' % (
                                epoch, batch_idx,
                                training_loss_sum / batch_count,
                                training_loss_LA_sum / batch_count,
                                training_loss_triplet_sum / batch_count,
                                training_loss_ood_sum / batch_count,
                                current_lr, triplet_weight, ood_weight
                            )
                print(train_str)
                if fp:
                    fp.write(train_str + '\n')
                    fp.flush()

        scheduler.step()

        epoch_avg_loss = training_loss_sum / batch_count
        epoch_avg_LA_loss = training_loss_LA_sum / batch_count
        epoch_avg_triplet_loss = training_loss_triplet_sum / batch_count
        epoch_avg_ood_loss = training_loss_ood_sum / batch_count
        epoch_str = f'Epoch {epoch} completed: Avg Loss = {epoch_avg_loss:.4f}, ' \
                    f'LA Loss = {epoch_avg_LA_loss:.4f}, ' \
                    f'Triplet Loss = {epoch_avg_triplet_loss:.4f}, ' \
                    f'OOD Loss = {epoch_avg_ood_loss:.4f}'
        print(epoch_str)






        """
        这段代码是用于在训练过程中定期评估模型的性能。
        每经过一定数量的 epoch，就会对验证集（clean set）进行评估，计算并输出模型的准确率和损失。
        总结：
            每隔一定的 epoch（由 args.eval_period 控制），模型会进行一次评估，计算测试集上的准确率和损失。
            在评估过程中，模型被设置为 eval() 模式，所有梯度计算被禁用。
            评估结果会在终端打印出来，并写入到文件中，便于跟踪模型的性能变化。
        """
        # 检查当前 epoch 是否是评估的时机。如果 (epoch + 1) % args.eval_period == 0，
        # 则每隔 args.eval_period 个 epoch 进行一次评估。args.eval_period 是一个超参数，控制评估的频率。
        if (epoch + 1) % args.eval_period == 0:
            # eval on clean set:
            # 将模型设置为评估模式。
            # 在评估模式下，模型会禁用一些特性，如 dropout 和 batch normalization，这些特性在训练时需要启用，但在评估时不使用。
            model.eval()
            # 创建两个 AverageMeter 实例，分别用于计算测试集上的平均准确率（test_acc_meter）和平均损失（test_loss_meter）。
            # AverageMeter 是一个自定义的类，用于累积和计算平均值。
            test_acc_meter, test_loss_meter = AverageMeter(), AverageMeter()
            # 初始化两个空的列表，preds_list 用于存储模型的预测结果，labels_list 用于存储真实标签。这些将用于计算最终的准确率。
            preds_list, labels_list = [], []
            # 使用 torch.no_grad() 上下文管理器来禁用梯度计算。在评估过程中不需要计算梯度，从而节省内存和计算资源。
            with torch.no_grad():
                # 遍历测试集的 test_loader。test_loader 是一个迭代器，它按批次加载测试数据。每个批次包含 data 和 labels。
                for data, labels in test_loader:
                    # 将数据 data 和标签 labels 移动到计算设备（如 GPU）。这样确保数据和模型在同一设备上进行计算。
                    data, labels = data.to(device), labels.to(device)
                    # 将输入数据 data 输入模型，得到预测的 logits（未经激活的输出）。
                    # 在这里假设模型输出了两个值，第一个是 logits，第二个是我们不需要的输出，所以用 _ 来忽略它。
                    logits, _ = model(data)
                    # 从 logits 中获取预测结果。logits.argmax(dim=1) 会返回每个样本的最大值索引，即模型认为最可能的类别。
                    # dim=1 表示在类别维度（通常是第二维）上选择最大值。keepdim=True 保持输出张量的维度。
                    pred = logits.argmax(dim=1, keepdim=True)
                    # 计算当前批次的交叉熵损失（cross_entropy）。
                    # 它将 logits 和真实标签 labels 作为输入，计算模型预测与真实标签之间的差异。
                    loss = F.cross_entropy(logits, labels)
                    # 计算当前批次的准确率并将其添加到 test_acc_meter 中。
                    # (logits.argmax(1) == labels) 返回一个布尔值张量，表示每个样本的预测是否正确。
                    # .float() 将布尔值转换为浮动值（1 或 0），.mean() 计算批次的平均准确率，.item() 提取标量值。
                    test_acc_meter.append((logits.argmax(1) == labels).float().mean().item())
                    # 将当前批次的损失值 loss.item() 添加到 test_loss_meter 中，方便计算整个测试集的平均损失。
                    test_loss_meter.append(loss.item())
                    # 将当前批次的预测结果 pred 添加到 preds_list 中。
                    preds_list.append(pred)
                    # 将当前批次的真实标签 labels 添加到 labels_list 中。
                    labels_list.append(labels)
            # 将所有批次的预测结果拼接成一个大的张量 preds，然后将其从计算图中分离出来（detach()），移动到 CPU（cpu()），
            # 转换为 NumPy 数组（numpy()），最后使用 .squeeze() 去掉多余的维度。
            preds = torch.cat(preds_list, dim=0).detach().cpu().numpy().squeeze()
            # 将所有批次的真实标签拼接成一个大的张量 labels，然后进行类似的处理：分离计算图，移动到 CPU，转换为 NumPy 数组。
            labels = torch.cat(labels_list, dim=0).detach().cpu().numpy()
            # 计算整体准确率。通过比较 preds 和 labels 是否相等，得到一个布尔值张量，sum() 计算正确预测的数量。
            # 除以总标签数 len(labels)，得到最终的准确率。
            overall_acc = (preds == labels).sum().item() / len(labels)
            # 将当前 epoch 的测试集平均损失（test_loss_meter.avg）添加到 test_clean_losses 列表中，以便记录和追踪损失的变化。
            test_clean_losses.append(test_loss_meter.avg)
            # 将当前 epoch 的整体准确率（overall_acc）添加到 overall_accs 列表中，便于记录和追踪准确率的变化。
            overall_accs.append(overall_acc)
            # 格式化一个字符串 val_str，用于输出当前 epoch 在测试集上的准确率。
            val_str = 'epoch %d (test): ACC %.4f ' % (epoch, overall_acc)
            # 打印评估结果字符串 val_str，显示当前 epoch 的准确率。
            print(val_str)
            # 将评估结果 val_str 写入到文件 fp_val 中，并在字符串后添加换行符。
            fp_val.write(val_str + '\n')
            # 刷新文件缓冲区，确保评估结果立即写入磁盘。
            fp_val.flush()
        """
        这段代码的目的是在训练过程中保存模型的状态以及相关信息。根据args.ddp的值（即是否启用分布式数据并行，DDP），保存的内容略有不同。
        总结：
            根据 args.ddp 的值，代码保存了模型的训练状态（包括模型参数、优化器状态、调度器状态、训练和测试损失、各类准确率等）。
            如果使用分布式数据并行（DDP），模型参数存储在 model.module.state_dict() 中；
            否则，直接存储在 model.state_dict() 中。保存的文件路径是 save_dir 目录下的 'latest.pth'。
        """
        # save pth:
        # 检查args.ddp是否为 True，如果是，表示模型在使用分布式数据并行（Distributed Data Parallel, DDP）进行训练。
        # 在DDP中，每个训练节点（如多个 GPU）都有一个模型副本，因此需要保存模型的特定部分。
        if args.ddp:
            # 使用 torch.save() 函数将一个字典（包含训练状态）保存到磁盘上。字典中的键值对包括模型、优化器、学习率调度器等信息。
            torch.save({
                # 如果启用了 DDP，模型的状态字典应该从 model.module 中获取，而不是直接从 model。
                # 这是因为在 DDP 中，model 是一个 DataParallel 或 DistributedDataParallel 对象，
                # 而实际的模型位于 model.module 中。state_dict() 获取模型的参数字典，它是保存模型时所需的内容。
                'model': model.module.state_dict(),
                # 保存优化器的状态字典。这允许在恢复模型时从上次的优化器状态继续训练，比如学习率、动量等参数。
                'optimizer': optimizer.state_dict(),
                # 保存学习率调度器的状态字典。调度器用于控制学习率的变化，
                # 通过保存调度器的状态，可以确保在恢复训练时继续从当前的学习率状态开始。
                'scheduler': scheduler.state_dict(),
                # 保存当前训练的 epoch 数量。这是为了在恢复训练时知道从哪一轮开始。
                'epoch': epoch,
                # 保存训练损失的历史记录（如每个 epoch 或批次的损失），这些可以用来分析训练过程中的表现。
                'training_losses': training_losses,
                # 保存测试集的损失记录。与训练损失类似，它用于记录每个 epoch 或测试阶段在测试集上的损失。
                'test_clean_losses': test_clean_losses,
                # 保存 F1 分数的历史记录。F1 分数是评估分类模型性能的常用指标，保存它可以用来评估模型在不同阶段的分类效果。
                'f1s': f1s,
                # 保存每个 epoch 或阶段的总体准确率（overall accuracy）。这有助于跟踪模型的性能。
                'overall_accs': overall_accs,
                # 保存某些特定类别或条件下的准确率，many_accs 可能是某些特定类别或样本的准确度。
                'many_accs': many_accs,
                # 保存每个 epoch 或阶段的中位数准确率，通常用于衡量模型在不同类别之间的均衡表现。
                'median_accs': median_accs,
                # 保存准确率较低的类别或样本的准确率。这可能有助于分析哪些部分的模型表现不佳。
                'low_accs': low_accs,
            },
                # 将上述字典保存到指定目录save_dir下的'latest.pth'文件中。os.path.join()用于确保路径在不同操作系统上正确拼接。
                os.path.join(save_dir, 'latest.pth'))
        # 如果 args.ddp 为 False，即没有启用分布式数据并行，则直接保存模型，而不需要从 model.module 获取状态字典。
        else:
            # 同样使用 torch.save() 保存一个字典，字典内容与启用 DDP 时相同，但这时模型直接通过 model.state_dict() 获取。
            torch.save({
                # 直接保存模型的状态字典，因为没有使用分布式数据并行。
                'model': model.state_dict(),
                # 保存优化器的状态字典。
                'optimizer': optimizer.state_dict(),
                # 保存学习率调度器的状态字典。
                'scheduler': scheduler.state_dict(),
                # 保存当前的 epoch 数量。
                'epoch': epoch,
                # 保存训练损失的历史记录。
                'training_losses': training_losses,
                # 保存测试集的损失记录。
                'test_clean_losses': test_clean_losses,
                # 保存 F1 分数的历史记录。
                'f1s': f1s,
                # 保存每个 epoch 的总体准确率。
                'overall_accs': overall_accs,
                # 保存某些特定类别或条件下的准确率。
                'many_accs': many_accs,
                # 保存每个 epoch 的中位数准确率。
                'median_accs': median_accs,
                # 保存准确率较低的类别或样本的准确率。
                'low_accs': low_accs,
            },
                # 将模型及其他状态字典保存到 save_dir 目录下的 'latest.pth' 文件中。
                os.path.join(save_dir, 'latest.pth'))

    """
    这段代码的作用是清理与分布式数据并行（DDP）相关的资源。
    总结：
        这段代码的目的是，在使用分布式数据并行训练完成后，
        调用 cleanup() 函数来释放 DDP 占用的资源，确保训练过程的清理工作能够正确完成。
    """
    # 这是一个注释，说明接下来的代码是用于清理 DDP（分布式数据并行）相关的操作或资源
    # Clean up ddp:
    # 这行代码检查 args.ddp 的值。如果 args.ddp 为 True，表示启用了分布式数据并行（DDP）。
    # 在这种情况下，需要执行清理操作。args.ddp 通常是在命令行参数中传递的，控制程序是否使用 DDP 来并行训练模型。
    if args.ddp:
        # 如果 args.ddp 为 True，则调用 cleanup() 函数来清理与 DDP 相关的资源。
        # 这个函数通常用于释放 GPU、清理进程、关闭通信通道等操作，以确保在分布式训练结束后，相关的资源能够被正确释放和回收。
        # cleanup() 这个函数是 PyTorch 的分布式训练中常用的函数。它主要执行以下任务（具体取决于 PyTorch 版本和自定义实现）：
        # 释放 DDP 使用的进程通信资源（如 NCCL 或 Gloo 后端）。清理并关闭通信的后台进程。
        # 可能会做一些与分布式训练相关的其他清理工作。
        cleanup()



















"""
    总结：
    这段代码根据命令行参数启动训练过程。如果启用了分布式训练（args.ddp），
    则使用 torch.multiprocessing.spawn() 启动多个进程进行分布式训练；否则，它会直接调用 train 函数进行单机单卡训练。
    os.environ['CUDA_VISIBLE_DEVICES'] 设置了要使用的 GPU，确保模型训练时只使用指定的设备。
"""
if __name__ == '__main__':
    # get args:
    # 调用 get_args_parser() 函数来获取命令行参数或者配置文件中的参数，并将返回值存储在 args 变量中。
    # 通常，get_args_parser() 函数会解析参数并返回一个包含所有配置选项的对象（如 argparse.Namespace）。
    args = get_args_parser()

    # mkdirs:
    # 调用 create_save_path() 函数来创建保存目录，并将返回的目录路径存储在 save_dir 变量中。
    # 通常，这个函数会根据参数生成一个路径，并确保该路径存在。
    save_dir = create_save_path()
    # 将 save_dir（即创建的目录路径）赋值给 args.save_dir。这意味着 args.save_dir 中现在存储了用于保存训练结果或模型的目录路径。
    args.save_dir = save_dir

    # set CUDA:
    # 通过设置 CUDA_VISIBLE_DEVICES 环境变量，指定要使用的 GPU。
    # args.gpu 应该包含一个或多个 GPU 的设备编号，通常以逗号分隔。这样设置后，PyTorch 将只会使用指定的 GPU，而忽略其他设备。
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # 判断 args.ddp 是否为真。args.ddp 可能是一个命令行参数，用于指定是否启用分布式数据并行（Distributed Data Parallel，DDP）。
    # 如果为真，则进行以下的 DDP 配置；如果为假，则使用普通训练。
    if args.ddp:
        # 调用 torch.cuda.device_count() 获取当前机器上可用的 GPU 数量，并将其存储在 ngpus_per_node 中。
        # 这个值通常用于设置分布式训练的进程数。
        ngpus_per_node = torch.cuda.device_count()
        # 调用 torch.multiprocessing.spawn() 启动多进程训练。这个函数会创建多个进程来进行分布式训练。
        # train：表示训练的主函数，将在每个进程中运行。
        # args=(ngpus_per_node, args)：将 ngpus_per_node 和 args 作为参数传递给 train 函数。
        # nprocs=ngpus_per_node：指定要启动的进程数量。通常情况下，进程数量等于 GPU 的数量。
        # join=True：表示等待所有子进程完成后再继续执行。这意味着主进程会等待所有训练进程结束。
        torch.multiprocessing.spawn(train, args=(ngpus_per_node, args), nprocs=ngpus_per_node, join=True)
    else:
        # 调用 train 函数进行训练。传入的参数 0, 0, args 表示在没有分布式训练的情况下，train 函数将以普通的方式运行。
        # 通常，0 表示第一个 GPU（即使用单卡训练）以及其他训练参数（在 args 中传递）。
        train(0, 0, args)

