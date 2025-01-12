import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# from pytorchltr.loss import PairwiseHingeLoss  # maybe no need it anymore
# from fast_soft_sort.pytorch_ops import soft_rank

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def listMLE(y_pred, y_true, n=None, eps=1e-8):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    # shuffle for randomised tie resolution
    random_indices = torch.randperm(y_pred.shape[-1])
    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)

    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)
    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values
    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max
    return torch.mean(torch.sum(observation_loss, dim=1))


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False, adapter_targets=None):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(
        self,
        image_features,
        text_features,
        logits,
        labels,
        logit_scale,
        output_dict=False
    ):
        clip_loss = torch.tensor(0)

        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {
                "contrastive_loss": clip_loss,
                "caption_loss": caption_loss,
            }
        return clip_loss, caption_loss


class CoCaWithAdapterLoss(CoCaLoss):
    """
    A RankNetLoss
    """
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            caption_loss_weight=caption_loss_weight,
            clip_loss_weight=clip_loss_weight,
            pad_id=pad_id,
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )
        # self.max_margin_loss = nn.MarginRankingLoss()
        self.adapter_loss_object = PairwiseHingeLoss()
        # self.adapter_loss_object = listMLE

    def forward(
        self,
        image_features,
        text_features,
        logits,
        labels,
        logit_scale,
        adapter_logits=None,  # [N, 1]
        adapter_targets=None,  # [N, ]
        adapter_logits_scale: float = 1.0,
        adapter_paired_logits=None,
        output_dict: bool = False,
    ):
        """NOTE
        adapter_logits_scale: Use to make the logits become more sensitive
        """
        clip_loss, caption_loss = super().forward(
            image_features,
            text_features,
            logits,
            labels,
            logit_scale,
        )
        ## RankNetLoss
        # predicted_risks = adapter_logits[:, 0].contiguous()
        # survival_statuses = torch.zeros(
        #     adapter_targets.size(0),
        #     device=adapter_logits.device,
        # )  # all zeros, no censor
        # last_observed_times = adapter_targets.contiguous()
        # pairwise_higher_risk_than = torch.logical_and(
        #     (survival_statuses == 0)[:, np.newaxis],
        #     (last_observed_times[:, np.newaxis] >= last_observed_times[np.newaxis, :]),
        # )
        # --- EOF of RankNet ---
        # predicted_risks = -predicted_risks  # larger yt have larger yp
        # pairwise_higher_risk_than = torch.logical_and(
        #     (survival_statuses == 0)[:, np.newaxis],
        #     (last_observed_times[:, np.newaxis] <= last_observed_times[np.newaxis, :]),
        # )
        # pairwise_negative_log_likelihood = -torch.nn.functional.logsigmoid(
        #     (predicted_risks[:, np.newaxis] - predicted_risks[np.newaxis, :]),
        # )
        # pairwise_ranknet_loss = torch.where(
        #     pairwise_higher_risk_than,
        #     pairwise_negative_log_likelihood,
        #     torch.zeros_like(
        #         pairwise_negative_log_likelihood,
        #         device=pairwise_negative_log_likelihood.device,
        #     ),
        # )
        # adapter_loss = torch.mean(pairwise_ranknet_loss)  # original implement is risk
        # --- EOF of RankNet ---

        # --- EOF of max margin ---
        # yt_pairwise_diff = last_observed_times[:, np.newaxis] - last_observed_times[np.newaxis, :]
        # pairwise_diff = predicted_risks[:, np.newaxis] - predicted_risks[np.newaxis, :]

        # y_trues = pairwise_higher_risk_than.view(-1)
        # y_trues = torch.where(y_trues, 1, -1)
        # y_preds = pairwise_diff.view(-1)
        # adapter_loss = self.max_margin_loss(
        #     yt_pairwise_diff.view(-1),
        #     y_preds,
        #     y_trues,
        # )
        # --- EOF of max_margin ---

        # Loss for ranking from other implementation
        y_trues = adapter_targets.contiguous().view((1, -1))
        ns = y_trues.size(-1)
        y_preds = adapter_logits[:, 0].contiguous().view((1, -1))

        # Apply entropy loss for in/out-classes guidence
        label_bce_loss = F.binary_cross_entropy_with_logits(
            y_preds.view((-1)),
            torch.where(y_trues == 0, 0, 1).view((-1)).to(y_preds.dtype),
        )

        regress_loss = (y_trues - y_preds).abs()  # For regression, use MAE

        adapter_loss = self.adapter_loss_object(
            y_preds,
            y_trues,
            torch.tensor([ns]).to(y_trues.device),
        )
        # adapter_loss = adapter_loss.mean() * 0.01 + label_bce_loss + regress_loss
        adapter_loss = label_bce_loss + regress_loss  # no ranking loss version

        ## Embedding loss (SmoothL1 distance)
        # if adapter_paired_logits is not None:
        #     y_for_paired = y_trues.view(-1)
        #     y_for_paired = y_for_paired[:, None] - y_for_paired[None, :]
        #     y_for_paired = y_for_paired.view((-1, 1))  # [N*N, 1]
        #     index_ = torch.where(
        #         (y_for_paired >= 0) &
        #         (y_trues.view(-1).repeat(y_trues.size(1)).view(-1, 1) > 0) &
        #         (~torch.eye(y_trues.size(1), dtype=torch.bool).view(-1, 1).to(y_trues.device)),
        #         1,
        #         0,
        #     )  # exclude non-target and self
        #     if sum(index_) > 0:
        #         y_for_paired = torch.log(y_for_paired.abs() + 1) * index_
        #         y_paried_logits = adapter_paired_logits * index_
        #         paired_loss = F.smooth_l1_loss(
        #             y_paried_logits,
        #             y_for_paired,
        #             reduction='sum',
        #         ) / sum(index_)
        #         adapter_loss = adapter_loss + paired_loss
        ## Embedding loss form1A (TripletLoss)
        if adapter_paired_logits is not None:
            yt = y_trues.view(-1)
            candidates = torch.where(yt > 0)[0]
            if len(candidates) > 3:
                anchor_id = np.random.choice(candidates.cpu(), 1)
                pos_embeddings, neg_embeddings = [], []
                for i in range(len(candidates) - 1):
                    for j in range(i + 1, len(candidates)):
                        if (i == anchor_id) or (j == anchor_id):
                            continue
                        if (yt[i] - yt[anchor_id]) > (yt[j] - yt[anchor_id]):
                            # distance to i > distance to j (j is closer)
                            pos_embeddings.append(adapter_paired_logits[j])
                            neg_embeddings.append(adapter_paired_logits[i])
                        elif (yt[i] - yt[anchor_id]) < (yt[j] - yt[anchor_id]):
                            pos_embeddings.append(adapter_paired_logits[i])
                            neg_embeddings.append(adapter_paired_logits[j])
                        else:
                            # D_a_i = D_a_j: skip
                            continue
                if len(pos_embeddings) > 0:
                    pos_embeddings = torch.stack(pos_embeddings)
                    neg_embeddings = torch.stack(neg_embeddings)
                    anchor_embeddings = adapter_paired_logits[anchor_id].repeat(
                        (pos_embeddings.size(0), 1)
                    )
                    paired_loss = F.triplet_margin_loss(
                        anchor_embeddings,
                        pos_embeddings,
                        neg_embeddings,
                    )
                    adapter_loss = adapter_loss + paired_loss

        ## smoothL1 w/ softrank
        # y_trues = soft_rank(adapter_targets.contiguous().view(1, -1))
        # y_trues = y_trues.view(-1, 1)
        # y_preds = adapter_logits[:, 0].contiguous().view(-1, 1)
        # adapter_loss = F.smooth_l1_loss(y_trues, y_preds)

        # DEBUGGER
        if torch.isnan(adapter_loss):
            print(f'NaN DEBUG, {y_trues=}')
            print(f'NaN DEBUG, {y_preds=}')
            # force adapter loss to 0?

        if output_dict:
            return {
                "contrastive_loss": clip_loss,
                "caption_loss": caption_loss,
                "adapter_loss": adapter_loss,
            }
        return clip_loss, caption_loss, adapter_loss


class ClipWithAdapterLoss(ClipLoss):
    """
    Use for single head.
    one with targets (N, 1)
    """
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            loss_components: list[str] = ['rank'],
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod,
        )
        # self.adapter_loss_object = PairwiseHingeLoss()
        print(f'V1 Loss components: {loss_components}')
        self.loss_components = loss_components

    def forward(
        self,
        image_features,
        text_features,
        logit_scale,
        adapter_logits=None,  # [N, 1]
        adapter_targets=None,  # [N, ]
        output_dict=False,
        adapter_paired_logits=None,  # backward competibility
    ):
        device = image_features.device
        # clip_loss = super().forward(image_features, text_features, logit_scale)
        clip_loss = torch.tensor(0.).to(device)  # no tune when training adapter

        y_trues = adapter_targets.contiguous().view((1, -1))
        ns = y_trues.size(-1)
        y_preds = adapter_logits[:, 0].contiguous().view((1, -1))

        ### BCE loss
        if 'bce' in self.loss_components:
            label_bce_loss = F.binary_cross_entropy_with_logits(
                y_preds.view((-1)),
                torch.where(y_trues == 0, 0, 1).view((-1)).to(y_preds.dtype),
            )
        else:
            label_bce_loss = torch.tensor(0.).to(device)

        ### pairwise-ranking
        if 'rank' in self.loss_components:
            ## (Use pytorch-ltr)
            # pair_rank_loss = self.adapter_loss_object(
            #     y_preds,
            #     y_trues,
            #     torch.tensor([ns]).to(y_trues.device),
            # )
            # pair_rank_loss = pair_rank_loss.mean()

            ## Use pairwise diff
            y_pair_trues = y_trues.view(-1).unsqueeze(1) - y_trues.view(-1).unsqueeze(0)
            y_pair_trues = y_pair_trues.view(-1, 1)
            y_pair_preds = adapter_logits.view(-1).unsqueeze(1) - adapter_logits.view(-1).unsqueeze(0)
            pair_rank_loss = F.binary_cross_entropy_with_logits(
                y_pair_preds.view(-1, 1),
                torch.where(y_pair_trues > 0., 1., 0.).to(y_preds.dtype),
            )
        else:
            pair_rank_loss = torch.tensor(0.).to(device)

        # regression Loss
        if 'regression' in self.loss_components:
            regression_loss = F.smooth_l1_loss(
                y_preds.view(-1, 1),
                y_trues.view(-1, 1),
            )
            regression_loss = torch.log(regression_loss + 1.)  # 1 ~ inf
        else:
            regression_loss = torch.tensor(0.).to(device)

        bce_weight = 1.0
        reg_weight = 0.2
        rnk_weight = 1.0
        adapter_loss = bce_weight * label_bce_loss + rnk_weight * pair_rank_loss + reg_weight * regression_loss

        if torch.isnan(adapter_loss):
            print(f'NaN DEBUG, {y_trues=}')
            print(f'NaN DEBUG, {y_preds=}')

        if output_dict:
            return {
                "contrastive_loss": clip_loss,
                "adapter_loss": adapter_loss,
            }

        return clip_loss, adapter_loss


class ClipAdapterLossV2(ClipLoss):
    """
    Use for dual head,
    one with regression targets (N, 1)
    one with rank targets (N**2, 1)
    """
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            loss_components: list[str] = ['rank'],
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod,
        )
        self.loss_components = loss_components
        print(f'Use ClipAdapterLossV2 with {loss_components}')

    def forward(
        self,
        image_features,
        text_features,
        logit_scale,
        adapter_logits=None,  # [N, 1]
        adapter_targets=None,  # [N, ]
        output_dict=False,
        adapter_paired_logits=None,  # [N*N, 1]
    ):
        device = image_features.device
        # Freeze backbone, ignore clip_loss
        clip_loss = torch.tensor(0.).to(device)

        y_trues = adapter_targets.contiguous().view(-1)
        y_preds = adapter_logits[:, 0].contiguous().view(-1)
        ns = len(y_trues)

        # loss-component: BCE Loss for absolute zero
        if 'bce' in self.loss_components:
            label_bce_loss = F.binary_cross_entropy_with_logits(
                y_preds,
                torch.where(y_trues == 0., 0., 1.).to(y_preds.dtype),
            )
        else:
            label_bce_loss = torch.tensor(0.).to(device)

        # loss-component: Ranking related loss
        if 'rank' in self.loss_components:
            y_pair_trues = y_trues.unsqueeze(1) - y_trues.unsqueeze(0)
            y_pair_trues = y_pair_trues.view(-1, 1)
            # default
            pair_rank_loss = F.binary_cross_entropy_with_logits(
                adapter_paired_logits,
                torch.where(
                    y_pair_trues > 0, 1., 0.,
                ).to(y_preds.dtype),
            )
            # experiment
            # pair_rank_loss = pairwise_paired_hinge_loss(
            #     adapter_paired_logits,
            #     y_pair_trues,
            # )
        else:
            pair_rank_loss = torch.tensor(0.).to(device)

        # loss-component: Regression target
        if 'regression' in self.loss_components:
            # default
            target_regress_loss = F.smooth_l1_loss(y_preds, y_trues)
            target_regress_loss = torch.log(target_regress_loss + 1.0)  # scaling

            # exp -- outcome: no big change (maybe weaker)
            # target_regress_loss = F.mse_loss(y_preds, y_trues)
        else:
            target_regress_loss = torch.tensor(0.).to(device)

        bce_weight = 1.0
        reg_weight = 0.2  # default: 0.2
        rnk_weight = 1.0
        adapter_loss = bce_weight * label_bce_loss + reg_weight * target_regress_loss + rnk_weight * pair_rank_loss

        if torch.isnan(adapter_loss):
            print(f'NaN DEBUG, {y_trues=}')
            print(f'NaN DEBUG, {y_preds=}')

        if output_dict:
            return {
                "contrastive_loss": clip_loss,
                "adapter_loss": adapter_loss,
            }

        return clip_loss, adapter_loss


class ClipDirectRankLoss(ClipLoss):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod,
        )
        self.loss_fn = PairwiseHingeLoss()

    def forward(
        self,
        image_features,
        text_features,
        logit_scale,
        adapter_targets=None,  # [N, ]
        output_dict=False,
    ):
        device = image_features.device
        logits_per_image, _ = self.get_logits(
            image_features,
            text_features,
            logit_scale,
        )  # logit_per_image will be same as logit_per_text when only single cap
        logits_per_image = logits_per_image[:, 0]  # only take first, other is the same
        y_trues = adapter_targets.contiguous().view((1, -1))
        ns = y_trues.size(-1)
        y_preds = logits_per_image.view((1, -1))
        clip_loss = self.loss_fn(
            y_preds,
            y_trues,
            torch.tensor([ns]).to(device),
        )
        if output_dict:
            return {
                "contrastive_loss": clip_loss,
            }
        return clip_loss


def spearmanr_loss(logits, targets, n=None):
    # n not used, to make it competible to others
    # Turn out to be failed
    logits = logits.view(1, -1)
    targets = targets.view(1, -1)
    logits = soft_rank(
        logits.to(torch.float32),
        regularization_strength=0.1
    )
    targets = soft_rank(
        targets.to(torch.float32),
        regularization_strength=0.1,
    )
    logits = logits - logits.mean()
    logits = logits / logits.norm()
    targets = targets - targets.mean()
    targets = targets / targets.norm()
    return 1. - (logits * targets).sum()


def pairwise_paired_hinge_loss(
    score_pair_diffs,
    rel_pair_diffs,
    margin=1.0,
):
    """
    score_pair_diffs: matrix of prediction difference
    rel_pair_diffs: matrix of relevance score difference
    """
    # dont normalize both, it will become too small
    # score_pair_diffs = torch.nn.functional.normalize(score_pair_diffs, dim=0)
    rel_pair_diffs = torch.nn.functional.normalize(rel_pair_diffs, dim=0)

    loss = torch.max(torch.zeros_like(score_pair_diffs), margin - score_pair_diffs * rel_pair_diffs)
    loss = torch.mean(loss)
    return loss


class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    def __init__(
            self,
            cache_labels=False,
            rank=0,
            world_size=1,
            bidir=True,
            use_horovod=False,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        assert not use_horovod  # FIXME need to look at hvd ops for ring transfers
        self.use_horovod = use_horovod
        self.bidir = bidir

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(self, image_features, text_features, logit_scale, logit_bias, output_dict=False):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        if self.world_size > 1:
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size
            if self.bidir:
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )

                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left

        return {"contrastive_loss": loss} if output_dict else loss


def approxNDCGLoss(y_pred, y_true, eps=1e-10, padded_value_indicator=-1, alpha=1.):
    """
    # https://github.com/allegro/allRank/blob/master/allrank/models/losses/approxNDCG.py
    Loss based on approximate NDCG introduced in "A General Approximation Framework for Direct Optimization of
    Information Retrieval Measures". Please note that this method does not implement any kind of truncation.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param alpha: score difference weight used in the sigmoid function
    :return: loss value, a torch.Tensor
    """
    device = y_pred.device
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    padded_mask = y_true == padded_value_indicator
    y_pred[padded_mask] = float("-inf")
    y_true[padded_mask] = float("-inf")

    # Here we sort the true and predicted relevancy scores.
    y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
    y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

    # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
    true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
    true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
    padded_pairs_mask = torch.isfinite(true_diffs)
    padded_pairs_mask.diagonal(dim1=-2, dim2=-1).zero_()

    # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
    true_sorted_by_preds.clamp_(min=0.)
    y_true_sorted.clamp_(min=0.)

    # Here we find the gains, discounts and ideal DCGs per slate.
    pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
    D = torch.log2(1. + pos_idxs.float())[None, :]
    maxDCGs = torch.sum((torch.pow(2, y_true_sorted) - 1) / D, dim=-1).clamp(min=eps)
    G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

    # Here we approximate the ranking positions according to Eqs 19-20 and later approximate NDCG (Eq 21)
    scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :])
    scores_diffs[~padded_pairs_mask] = 0.
    approx_pos = 1. + torch.sum(padded_pairs_mask.float() * (torch.sigmoid(-alpha * scores_diffs).clamp(min=eps)), dim=-1)
    approx_D = torch.log2(1. + approx_pos)
    approx_NDCG = torch.sum((G / approx_D), dim=-1)

    return -torch.mean(approx_NDCG)
