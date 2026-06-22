import os

import torch
from torch import nn
import torch.nn.functional as F
from fast_pytorch_kmeans import KMeans


class RSRM(nn.Module):
    def __init__(self, backbone):
        super(RSRM, self).__init__()
        
        # Load backbone
        if backbone == 'dinov3':
            self.backbone = torch.hub.load("./dinov3", 'dinov3_vitb16', source='local', weights=os.path.expanduser("~/.cache/torch/hub/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"))
            self.layers = 12
        
        for param in self.backbone.parameters():
            param.requires_grad = False


    def forward(self, img_s_list, mask_s_list, img_q):
        h, w = img_q.shape[-2:]

        # extract multi-layer support features
        feature_s_last_list = []
        feature_s_attn_list = []
        resize_mask_s_list = []
        for k in range(len(img_s_list)):
            features_sum, features_attn = self.backbone.get_intermediate_layers(x=img_s_list[k], n=range(0,self.layers), reshape=True)
            feature_s_attn_list.append(torch.stack(features_attn, dim=1))  # [B, L, C, H, W]
            feature_s_last_list.append(features_sum[-1])

            resize_mask_s = F.interpolate(mask_s_list[k].unsqueeze(1).float(), size=features_attn[-1].shape[-2:], mode="bilinear", align_corners=True)
            resize_mask_s = (resize_mask_s > 0.5).long().squeeze(1)
            resize_mask_s_list.append(resize_mask_s)

        # extract multi-layer query features
        features_sum, features_attn = self.backbone.get_intermediate_layers(x=img_q, n=range(0,self.layers), reshape=True)
        feature_q_attn = torch.stack(features_attn, dim=1)
        feature_q_last = features_sum[-1]


        # ---------- Semantic-aware Feature Re-fusion (SAFR) ------------
        # weights
        channel_weights = self.features_refusion(feature_s_last_list, feature_s_attn_list, resize_mask_s_list)  # [B, L]
        # fusion features
        feature_q = (feature_q_attn * channel_weights[:,:,None,None,None] / feature_q_attn.std(dim=(2,3,4), keepdim=True)).sum(dim=1)
        feature_s_list = []
        for k in range(len(img_s_list)):
            feature_s_list.append((feature_s_attn_list[k] * channel_weights[:,:,None,None,None] / feature_s_attn_list[k].std(dim=(2,3,4), keepdim=True)).sum(dim=1))


        # ---------- Adaptive Support Enhancement (ASE) ------------
        # foreground(target class) and background prototypes pooled from K support features
        feature_fg_list = []    # [K,B,C]
        feature_bg_list = []    # [K,B,C]
        fg_thresh_ls = []
        bg_thresh_ls = []
        for k in range(len(img_s_list)):
            # [1,B,C]
            feature_fg = self.masked_average_pooling(feature_s_list[k],
                                                     (resize_mask_s_list[k] == 1).float())[None, :]
            feature_bg = self.masked_average_pooling(feature_s_list[k],
                                                     (resize_mask_s_list[k] == 0).float())[None, :]
            feature_fg_list.append(feature_fg)
            feature_bg_list.append(feature_bg)

            # robust threshold estimation
            support_sim = self.similarity_func(feature_s_list[k], feature_fg[0].unsqueeze(-1).unsqueeze(-1), feature_bg[0].unsqueeze(-1).unsqueeze(-1))
            support_prob = support_sim.softmax(1)

            fg_thresh = self.find_robust_threshold_iou(support_prob[:,1], resize_mask_s_list[k])
            bg_thresh = self.find_robust_threshold_iou(support_prob[:,0], 1-resize_mask_s_list[k])
            # print(fg_thresh, bg_thresh)
            fg_thresh_ls.append(fg_thresh)
            bg_thresh_ls.append(bg_thresh)

        fg_thresh = torch.tensor(fg_thresh_ls).mean().item()
        bg_thresh = torch.tensor(bg_thresh_ls).mean().item()

        # average K foreground prototypes and K background prototypes，[B,C,1,1]
        FP = torch.mean(torch.cat(feature_fg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)
        BP = torch.mean(torch.cat(feature_bg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)
        # measure the similarity of query features to fg/bg prototypes
        prior_sim = self.similarity_func(feature_q, FP, BP)
        # prior mask generation
        prior_mask = self.generate_prior_mask(prior_sim.softmax(1), fg_threshold=fg_thresh, bg_threshold=bg_thresh)
        
        # support feature enhancement
        cross_feat_s_list = []
        for k in range(len(feature_s_list)):
            cross_feat_s = self.support_enhance(feature_q, prior_mask, feature_s_list[k], resize_mask_s_list[k])
            cross_feat_s_list.append((cross_feat_s+feature_s_list[k])/2)
        cross_feat_q = feature_q

        
        # ---------- Hybrid Prototype Matching (HPM) ------------
        fg_sim_ls = []
        bg_sim_ls = []
        for k in range(len(img_s_list)):
            # [B,H,W]
            fg_sim, bg_sim = self.multi_scale_similarity(cross_feat_q, cross_feat_s_list[k], resize_mask_s_list[k])
            fg_sim_ls.append(fg_sim.unsqueeze(1))
            bg_sim_ls.append(bg_sim.unsqueeze(1))

        fg_sim = torch.mean(torch.cat(fg_sim_ls, dim=1), dim=1, keepdim=True)
        bg_sim = torch.mean(torch.cat(bg_sim_ls, dim=1), dim=1, keepdim=True)
        
        out_0 = torch.cat((bg_sim, fg_sim), dim=1)
        out_0 = F.interpolate(out_0, size=(h, w), mode="bilinear", align_corners=True)

        out_ls = [out_0]
        return out_ls
    
    
    # calculate feature weights for re-fusion
    def features_refusion(self, feature_s_last_list, feature_s_attn_list, mask_s_list):
        fusion_weights = 0

        for k in range(len(feature_s_attn_list)):
            B,L,C,H,W = feature_s_attn_list[k].shape
            mask_s = mask_s_list[k]

            # [B,n]
            channel_num = round(C/200)
            final_feature = feature_s_last_list[k]
            channel_std = final_feature.var(dim=(-1,-2))
            top_indices = torch.topk(channel_std, channel_num, dim=1).indices

            # variance proportions
            layer_var = feature_s_attn_list[k].var(dim=(-1,-2))  # [B,L,C]
            batch_indices = torch.arange(B, device=feature_s_attn_list[k].device).view(B, 1, 1).expand(B, L, channel_num)
            layer_indices = torch.arange(L, device=feature_s_attn_list[k].device).view(1, L, 1).expand(B, L, channel_num)
            channel_indices = top_indices.unsqueeze(1).expand(B, L, channel_num)
            # [B, L, n]
            selected_variances = layer_var[batch_indices, layer_indices, channel_indices]
            # [B, L]
            selected_sum = selected_variances.sum(dim=2)
            # [B, L]
            total_sum = layer_var.sum(dim=2)
            var_rate = selected_sum / (total_sum + 1e-8)

            # find representative semantic-aware layer
            indicator = self.find_best_semantic_layer(var_rate)

            # calculate FDR
            fdr_ls = []
            for l in range(L):
                if indicator[l]:
                    feat_s = feature_s_attn_list[k][:, l, :, :, :]
                    fdr_ls.append(self.fisher_ratio(feat_s, mask_s).unsqueeze(-1))
                else:
                    fdr_ls.append(torch.zeros(B, 1, device=feature_s_attn_list[k].device))

            fdr_ls = torch.cat(fdr_ls, dim=-1)
            fusion_weight = (fdr_ls * total_sum) / (selected_sum + 1e-8)
            fusion_weights += fusion_weight
        
        fusion_weights = fusion_weights / len(feature_s_attn_list)
        fusion_weights = fusion_weights / fusion_weights.sum(dim=1, keepdim=True)
        
        return fusion_weights
    

    # find representative semantic-aware layer
    def find_best_semantic_layer(self, var_rate):
        B, L = var_rate.shape

        mean_rate = var_rate.mean(dim=1, keepdim=True)
        is_location = var_rate > mean_rate  # [B, L] → bool

        indicator = torch.zeros_like(var_rate, dtype=torch.bool)
        for b in range(B):
            location_idx = torch.nonzero(is_location[b], as_tuple=False).flatten()
            
            for i in range(location_idx.size(0)):
                l_idx = location_idx[i].item()
                start = l_idx + 1
                if start >= L:
                    continue

                end = L
                if i + 1 < location_idx.size(0):
                    end = location_idx[i+1].item()
                
                if end <= start:
                    continue

                candidate_range = var_rate[b, start:end]               # shape: [end - start]
                min_offset = torch.argmin(candidate_range).item()
                target_idx = start + min_offset
                indicator[b, target_idx] = True

        return indicator.any(dim=0)  # [L]


    # FDR
    def fisher_ratio(self, feature, mask):
        B, C, H, W = feature.shape
        
        scores = []
        for i in range(B):
            feat_flat = feature[i].view(C, -1).T
            mask_flat = mask[i].view(-1)
            
            fg_feat = feat_flat[mask_flat == 1]
            bg_feat = feat_flat[mask_flat == 0]
            
            if len(fg_feat) < 2 or len(bg_feat) < 2:
                continue
                
            # class centers
            fg_center = fg_feat.mean(dim=0)
            bg_center = bg_feat.mean(dim=0)
            
            # between-class scatter matrix
            between_scatter = torch.outer(fg_center - bg_center, fg_center - bg_center)
            # within-class scatter matrix
            fg_cov = torch.cov(fg_feat.T)
            bg_cov = torch.cov(bg_feat.T)
            within_scatter = fg_cov + bg_cov
            
            # Fisher ratio
            try:
                fisher_score = torch.trace(between_scatter) / torch.trace((within_scatter + 1e-8))
                scores.append(fisher_score)
            except:
                continue
        scores = torch.stack(scores)
        
        return scores
    

    def similarity_func(self, feature_q, fg_proto, bg_proto):
        similarity_fg = F.cosine_similarity(feature_q, fg_proto, dim=1)
        similarity_bg = F.cosine_similarity(feature_q, bg_proto, dim=1)
        # [B,2,H,W]
        out = torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1) * 10.0
        return out

    # MAP，[B,C,H,W]-->[B,C]
    def masked_average_pooling(self, feature, mask):
        mask = F.interpolate(mask.unsqueeze(1), size=feature.shape[-2:], mode='bilinear', align_corners=True)
        masked_feature = torch.sum(feature * mask, dim=(2, 3)) \
                         / (mask.sum(dim=(2, 3)) + 1e-5)
        return masked_feature
    

    def find_robust_threshold_iou(self, prob_map, true_mask, num_thresholds=21):

        def compute_iou(pred, target, eps=1e-6):
            intersection = (pred * target).sum(dim=(1, 2))
            union = (pred + target - pred * target).sum(dim=(1, 2))
            iou = (intersection + eps) / (union + eps)
            return iou

        device = prob_map.device
        thresholds = torch.linspace(0.0, 1.0, steps=num_thresholds).to(device)  # [T]
        batch_size = prob_map.size(0)

        best_ious = torch.zeros(batch_size, device=device)
        best_thresholds = torch.zeros(batch_size, device=device)
        
        all_ious = []

        for t in thresholds:
            pred_binary = (prob_map >= t).float()
            ious = compute_iou(pred_binary, true_mask)
            all_ious.append(ious.unsqueeze(0))  # [1, B]
            
            better = ious > best_ious
            best_ious = torch.where(better, ious, best_ious)
            best_thresholds = torch.where(better, t, best_thresholds)

        iou_curve = torch.cat(all_ious, dim=0)  # [T, B]

        # IoU > 0.75 * best_iou
        threshold_mask = iou_curve >= (0.75 * best_ious.unsqueeze(0))  # [T, B]
        robust_thresholds = thresholds.unsqueeze(1).expand_as(iou_curve)[threshold_mask]  # [N_valid_values]

        return robust_thresholds.mean().item()
    

    def generate_prior_mask(self, prob_map, fg_threshold=0.7, bg_threshold=0.6, topk=12):
        device = prob_map.device
        B, _, H, W = prob_map.shape
        
        fg_mask = torch.zeros_like(prob_map[:, 1, :, :], dtype=torch.bool, device=device)
        bg_mask = torch.zeros_like(prob_map[:, 0, :, :], dtype=torch.bool, device=device)
        
        fg_prob = prob_map[:, 1, :, :]  # [B, H, W]
        bg_prob = prob_map[:, 0, :, :]  # [B, H, W]
        
        for i in range(B):
            fg_high_prob = fg_prob[i] > fg_threshold
            if fg_high_prob.any():
                fg_mask[i] = fg_high_prob
            else:
                flat_fg = fg_prob[i].flatten()
                _, indices = torch.topk(flat_fg, min(topk, flat_fg.numel()))
                temp_mask = torch.zeros_like(flat_fg, dtype=torch.bool, device=device)
                temp_mask[indices] = True
                fg_mask[i] = temp_mask.reshape(H, W)
            
            bg_high_prob = bg_prob[i] > bg_threshold
            if bg_high_prob.any():
                bg_mask[i] = bg_high_prob
            else:
                flat_bg = bg_prob[i].flatten()
                _, indices = torch.topk(flat_bg, min(topk, flat_bg.numel()))
                temp_mask = torch.zeros_like(flat_bg, dtype=torch.bool, device=device)
                temp_mask[indices] = True
                bg_mask[i] = temp_mask.reshape(H, W)
        
        mask = torch.stack([bg_mask, fg_mask], dim=1).float()
        
        return mask
    
    
    def support_enhance(self, x_q, mask_q, x_s, mask_s):
        B, C, H, W = x_q.shape

        feat_q = x_q.view(B, C, H*W)
        feat_s = x_s.view(B, C, H*W)
        fg_mask_q = mask_q[:, 1].view(B, -1)
        bg_mask_q = mask_q[:, 0].view(B, -1)
        fg_mask_s = mask_s.view(B, -1)
        bg_mask_s = 1 - fg_mask_s

        large_negative = -1e9
        # s2q mask
        sq_mask = torch.logical_or(torch.logical_and(fg_mask_q.unsqueeze(1)>0.5, fg_mask_s.unsqueeze(2)>0.5), torch.logical_and(bg_mask_q.unsqueeze(1)>0.5, bg_mask_s.unsqueeze(2)>0.5))  # [B, N, N]
        sq_mask = 1 - sq_mask.float()
        sq_mask = sq_mask * large_negative

        # masked attention
        attn_s = torch.bmm(feat_s.transpose(1, 2), feat_q)  # [B, N, N]
        attn_s = attn_s / (C ** 0.5)
        attn_s = attn_s + sq_mask
        attn_s = F.softmax(attn_s, dim=-1)
        cross_x_s = torch.bmm(attn_s, feat_q.transpose(1, 2)).transpose(1,2)
        cross_x_s = cross_x_s.view(B, C, H, W)

        return cross_x_s
    

    # multi-scale prototype generation
    def multi_scale_similarity(self, feature_q, feature_s, mask_s):
        B, C, H, W = feature_q.shape
        fg_scales = [1, 9, -1]
        bg_scales = [1, 18, -1]

        fg_sim_ls = []
        bg_sim_ls = []
        for b in range(B):
            cur_feature = feature_s[b].view(C, -1).permute(1, 0)  # [H*W, C]
            cur_mask = mask_s[b].view(-1)
            cur_feature_q = feature_q[b].view(C, -1).permute(1, 0)  # [H*W, C]

            fg_idx = torch.where(cur_mask > 0.5)[0]
            bg_idx = torch.where(cur_mask <= 0.5)[0]

            fg_feature = cur_feature[fg_idx]  # [N, C]
            bg_feature = cur_feature[bg_idx]  # [M, C]

            fg_sim_maps = []
            bg_sim_maps = []
            confs = []
            for fg_scale, bg_scale in zip(fg_scales, bg_scales):
                # pixel-wise
                if fg_scale == -1 and bg_scale == -1:
                    sim_matrix_fg = torch.mm(F.normalize(fg_feature, p=2, dim=1), F.normalize(cur_feature_q, p=2, dim=1).permute(1, 0))  # [N, H*W]
                    sim_matrix_bg = torch.mm(F.normalize(bg_feature, p=2, dim=1), F.normalize(cur_feature_q, p=2, dim=1).permute(1, 0))  # [M, H*W]
                    fg_sim_map, _ = torch.max(sim_matrix_fg, dim=0)
                    bg_sim_map, _ = torch.max(sim_matrix_bg, dim=0)
                    fg_sim_map = fg_sim_map.view(1, H, W)
                    bg_sim_map = bg_sim_map.view(1, H, W)

                    sim_map = torch.cat([bg_sim_map, fg_sim_map], dim=0)
                    fg_sim_maps.append(sim_map[1:])
                    bg_sim_maps.append(sim_map[0:1])
                # regional or global
                else:
                    fg_scale = min(fg_scale, fg_feature.shape[0])
                    bg_scale = min(bg_scale, bg_feature.shape[0])
                    
                    fg_sim_map = self.region_similarity(fg_feature, feature_q[b].unsqueeze(0), fg_scale)
                    bg_sim_map = self.region_similarity(bg_feature, feature_q[b].unsqueeze(0), bg_scale)

                    sim_map = torch.cat([bg_sim_map, fg_sim_map], dim=0)
                    fg_sim_maps.append(sim_map[1:])
                    bg_sim_maps.append(sim_map[0:1])

            fg_sim_maps = torch.cat(fg_sim_maps, dim=0)
            bg_sim_maps = torch.cat(bg_sim_maps, dim=0)

            # weights
            conf_map = (fg_sim_maps-bg_sim_maps)
            mask = (conf_map > 0).float()
            fg_sim_maps_norm = (fg_sim_maps - fg_sim_maps.min()) / (fg_sim_maps.max() - fg_sim_maps.min() + 1e-5)
            fg_confs = self.masked_average_pooling(fg_sim_maps_norm.unsqueeze(1), mask)
            bg_confs = self.masked_average_pooling(fg_sim_maps_norm.unsqueeze(1), (1-mask))
            confs = (fg_confs - bg_confs).unsqueeze(-1)
            confs = (confs/fg_sim_maps.abs().std()).softmax(0)

            # fg
            fg_sim_maps = torch.mean(confs*fg_sim_maps, dim=0, keepdim=True)
            fg_sim_ls.append(fg_sim_maps)
            # bg
            bg_sim_maps = torch.mean(confs*bg_sim_maps, dim=0, keepdim=True)
            bg_sim_ls.append(bg_sim_maps)

        # [B,H,W]
        fg_sim_maps = torch.cat(fg_sim_ls, dim=0)
        bg_sim_maps = torch.cat(bg_sim_ls, dim=0)

        return fg_sim_maps, bg_sim_maps
    

    def region_similarity(self, feature_s, feature_q, n_clusters):
        B, C, H, W = feature_q.shape
        N, C = feature_s.shape

        # K-Means
        kmeans = KMeans(n_clusters=n_clusters, init_method='kmeans++', verbose=0)
        labels = kmeans.fit_predict(feature_s)
        sim_maps = []
        for region_id in range(n_clusters):
            region_mask = (labels == region_id)
            region_feature = feature_s[region_mask]
            fg_proto = region_feature.mean(dim=0, keepdim=True)
            fg_sim_map = F.cosine_similarity(feature_q, fg_proto.unsqueeze(-1).unsqueeze(-1), dim=1)
            sim_maps.append(fg_sim_map)
        sim_maps = torch.cat(sim_maps, dim=0)
        sim_maps, _ = torch.max(sim_maps, dim=0, keepdim=True)
        # sim_maps = torch.mean(sim_maps, dim=0, keepdim=True)

        return sim_maps
