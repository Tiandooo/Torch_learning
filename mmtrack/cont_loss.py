    def loss_ctr(self, tokens: torch.Tensor, pred_tks: torch.Tensor, pred_box: torch.Tensor = None, cross: bool = False) -> torch.Tensor:
        """
        loss for contrastive learning
        Args:
            tokens: [batch, max_objs], tracking ids

            pred_tk: tensor(batch, max_objs, Dim)
            pred_box: tensor(batch, max_objs, 10) pred bbox
            cross: whether to compute the contrastive loss for different classes

        Returns:

        """
        if pred_box is not None:
            box_dim = pred_box.shape[-1]
            tk_dim = pred_tks.shape[-1]

            pred_box = pred_box
            pred_box = pred_box.reshape(-1, box_dim)
            # tensor(batch, max_objs, Dim) -> tensor(num_objs, Dim)
            pred_tks = pred_tks.reshape(-1, tk_dim)

            pred_tks = self.feature_fusion(pred_box, pred_tks)
        else:
            tk_dim = pred_tks.shape[-1]
            pred_tks = pred_tks.reshape(-1, tk_dim)

        if cross is False:
            # tensor(batch, max_objs) -> tensor(num_objs, 1)

            tokens = torch.reshape(tokens, (-1, 1))
            # delete empty object
            object_mask = torch.where(tokens!=0)
            tokens = tokens[object_mask].reshape(-1, 1)
            nums = tokens.shape[0]

            pred_tks = pred_tks[object_mask[0], :]

            assert pred_tks.shape[0] == nums, "shape of tracking embeddings and tracking ids are not paired!"

        else:
            # [task_idx, batch, max_objs] -> [task_idx*batch*max_objs, 1]
            tasks, batchs, max_objs = len(tokens), tokens[0].shape[0], tokens[0].shape[1]
            nums = tasks * batchs * max_objs
            tokens = torch.stack(tokens, dim=0).reshape(-1, 1)
            # end
            dim = pred_tks[0].shape[-1]
            pred_tks = torch.stack(pred_tks, dim=0).reshape(-1, dim)
            pred_tks = pred_tks.reshape(-1, dim)
            assert pred_tks.shape[0] == nums, "shape of tracking embeddings and tracking ids are not paired!"

        # positive pairs, a[i][j] means i and j is a positive pair
        mask = torch.eq(tokens, tokens.T).float().to('cuda')
        logits_mask = torch.ones_like(mask) - torch.eye(nums).cuda()
        pos_mask = mask * logits_mask
        if pos_mask.sum().item() == 0:
            return
        neg_mask = 1. - mask

        temperature = 0.5
        # anchor_dot_contrast = torch.div(
        #     torch.matmul(pred_tks, pred_tks.T),
        #     temperature)  # 计算两两样本间点乘相似度

        # 归一化
        pred_tks = F.normalize(pred_tks)

        """
            contr loss from web        
        """
        # anchor_dot_contrast = torch.div(
        #     torch.matmul(pred_tks, pred_tks.T),
        #     temperature)  # 计算两两样本间点乘相似度
        # # for numerical stability
        # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # logits = anchor_dot_contrast - logits_max.detach()
        # exp_logits = torch.exp(logits)
        #
        # num_positives_per_row = torch.sum(pos_mask, dim=1)  # 除了自己之外，正样本的个数  [2 0 2 2]
        # denominator = torch.sum(
        #     exp_logits * neg_mask, dim=1, keepdim=True) + torch.sum(
        #     exp_logits * pos_mask, dim=1, keepdim=True)
        #
        # log_probs = logits - torch.log(denominator)
        # log_probs = logits
        # if torch.any(torch.isnan(log_probs)):
        #     raise ValueError("Log_prob has nan!")
        #
        # log_probs = torch.sum(
        #     log_probs * pos_mask, dim=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]
        #
        # '''
        # 计算正样本平均的log-likelihood
        # 考虑到一个类别可能只有一个样本，就没有正样本了 比如我们labels的第二个类别 labels[1,2,1,1]
        # 所以这里只计算正样本个数>0的
        # '''
        # # loss
        # loss = -log_probs
        # # if self.scale_by_temperature:
        # #     loss *= self.temperature
        # loss *= temperature
        #
        # # loss = loss.sum()
        # # loss = loss / len
        #
        # loss = loss.mean()


        """
        contr loss from claude and equation
        """
        # 计算两两样本间相似度
        anchor_sim = torch.matmul(pred_tks, pred_tks.T) / temperature

        # 将自身相似度设为很小的值,避免在后面的运算中影响
        diag_mask = torch.eye(anchor_sim.size(0)).bool()
        anchor_sim[diag_mask] = -1e6

        # 计算分子部分(正样本指数相似度)
        pos_numerators = torch.exp(anchor_sim) * pos_mask.float()
        # 计算负样本的指数相似度之和
        neg_denominators = torch.sum(torch.exp(anchor_sim) * neg_mask.float(), dim=1)

        # 计算分母部分 (负样本之和 + 单独正样本)
        denominators = neg_denominators + pos_numerators

        # 防止分母为0
        denominators = torch.clamp(denominators, min=1e-8)

        # 计算每个正样本对的对数似然比
        log_probs = pos_numerators / denominators

        log_probs = log_probs[log_probs != 0]

        log_probs = torch.log(log_probs)

        # 平均loss
        losses = -log_probs

        # 对损失求平均
        total_loss = losses.mean()
        return total_loss