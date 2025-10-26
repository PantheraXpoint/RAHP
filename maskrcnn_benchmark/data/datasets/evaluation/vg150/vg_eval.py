import os
import torch
import numpy as np
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from maskrcnn_benchmark.data.datasets.evaluation.vg150.sgg_eval import SGRecall, SGNoGraphConstraintRecall, SGZeroShotRecall, SGNGZeroShotRecall, SGPairAccuracy, SGMeanRecall, SGNGMeanRecall, SGAccumulateRecall, SGRelVecRecall, SGStagewiseRecall
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
import logging
logger = logging.getLogger("maskrcnn_benchmark." + __name__)
import torch.distributed as dist

def do_vg_evaluation(
    cfg,
    dataset,
    predictions,
    output_folder,
    iou_types,
):
    # get zeroshot triplet
    zeroshot_triplet = torch.load("maskrcnn_benchmark/data/datasets/evaluation/vg150/zeroshot_triplet.pytorch", map_location=torch.device("cpu")).long().numpy()
    predicates_categories = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind', 'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for', 'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on', 'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over', 'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on', 'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']

    attribute_on = cfg.MODEL.ATTRIBUTE_ON
    num_attributes = cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
    # extract evaluation settings from cfg
    # mode = cfg.TEST.RELATION.EVAL_MODE
    if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            mode = 'predcls'
        else:
            mode = 'sgcls'
    else:
        mode = 'sgdet'

    num_rel_category = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
    multiple_preds = cfg.TEST.RELATION.MULTIPLE_PREDS
    iou_thres = cfg.TEST.RELATION.IOU_THRESHOLD
    assert mode in {'predcls', 'sgdet', 'sgcls', 'phrdet', 'preddet'}

    result = []
    groundtruths = []
    for image_id, prediction in enumerate(predictions):
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        # recover original size which is before transform
        predictions[image_id] = prediction.resize((image_width, image_height))

        gt = dataset.get_groundtruth(image_id, evaluation=True)
        groundtruths.append(gt)

        # # assign pred_box with GLIP label in proposal (zero-shot prediction)
        if not cfg.MODEL.RELATION_ON or not cfg.MODEL.DYHEAD.RELATION_CONSISTENCY_ON:
            predictions[image_id].add_field('pred_scores', predictions[image_id].get_field('scores'))
            predictions[image_id].add_field('pred_labels', predictions[image_id].get_field('labels'))

        # # assign pred_box with max_iou GT label
        # match_quality_matrix = boxlist_iou(gt, predictions[image_id])
        # vals, inds = match_quality_matrix.max(dim=0)
        # predictions[image_id].add_field('pred_labels', gt.get_field('labels')[inds])
        # predictions[image_id].get_field('pred_scores')[vals > 0.5] = 1 # for ranking

    save_output(output_folder, groundtruths, predictions, dataset)
    result_str = '\n' + '=' * 100 + '\n'
    if "bbox" in iou_types:
        # create a Coco-like object that we can use to evaluate detection!
        anns = []
        for image_id, gt in enumerate(groundtruths):
            labels = gt.get_field('labels').tolist() # integer
            boxes = gt.bbox.tolist() # xyxy
            for cls, box in zip(labels, boxes):
                anns.append({
                    'area': (box[3] - box[1] + 1) * (box[2] - box[0] + 1),
                    'bbox': [box[0], box[1], box[2] - box[0] + 1, box[3] - box[1] + 1], # xywh
                    'category_id': cls,
                    'id': len(anns),
                    'image_id': image_id,
                    'iscrowd': 0,
                })
        fauxcoco = COCO()
        fauxcoco.dataset = {
            'info': {'description': 'use coco script for vg detection evaluation'},
            'images': [{'id': i} for i in range(len(groundtruths))],
            'categories': [
                {'supercategory': 'person', 'id': i, 'name': name} 
                for i, name in enumerate(dataset.ind_to_classes) if name != '__background__'
                ],
            'annotations': anns,
        }
        fauxcoco.createIndex()

        # format predictions to coco-like
        cocolike_predictions = []
        for image_id, prediction in enumerate(predictions):
            box = prediction.convert('xywh').bbox.detach().cpu().numpy() # xywh
            score = prediction.get_field('pred_scores').detach().cpu().numpy() # (#objs,)
            label = prediction.get_field('pred_labels').detach().cpu().numpy() # (#objs,)
            # for predcls, we set label and score to groundtruth
            if mode == 'predcls':
                label = prediction.get_field('labels').detach().cpu().numpy()
                score = np.ones(label.shape[0])
                assert len(label) == len(box)
            image_id = np.asarray([image_id]*len(box))
            cocolike_predictions.append(
                np.column_stack((image_id, box, score, label))
            )
            # print(cocolike_predictions)
        cocolike_predictions = np.concatenate(cocolike_predictions, 0)
        # evaluate via coco API
        res = fauxcoco.loadRes(cocolike_predictions)
        coco_eval = COCOeval(fauxcoco, res, 'bbox')
        coco_eval.params.imgIds = list(range(len(groundtruths)))
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        summarize_per_category(coco_eval, os.path.join(output_folder, 'bbox_per_cats.csv'))
        mAp = coco_eval.stats[1]

        result_str += 'Detection evaluation mAp=%.4f\n' % mAp
        result_str += '=' * 100 + '\n'

    if "relations" in iou_types or cfg.MODEL.DYHEAD.RELATION_CONSISTENCY_ON:
        result_dict = {}
        evaluator = {}
        # tradictional Recall@K
        eval_recall = SGRecall(result_dict, num_rel_category)
        eval_recall.register_container(mode)
        evaluator['eval_recall'] = eval_recall

        # no graphical constraint
        eval_nog_recall = SGNoGraphConstraintRecall(result_dict)
        eval_nog_recall.register_container(mode)
        evaluator['eval_nog_recall'] = eval_nog_recall

        # test on different distribution
        eval_zeroshot_recall = SGZeroShotRecall(result_dict, cfg.MODEL.DYHEAD.OV.ZS_OBJ_PREDICATES, cfg.MODEL.DYHEAD.OV.ZS_PREDICATES)
        eval_zeroshot_recall.register_container(mode)
        evaluator['eval_zeroshot_recall'] = eval_zeroshot_recall

        # test on no graph constraint zero-shot recall
        eval_ng_zeroshot_recall = SGNGZeroShotRecall(result_dict)
        eval_ng_zeroshot_recall.register_container(mode)
        evaluator['eval_ng_zeroshot_recall'] = eval_ng_zeroshot_recall
        
        # used by https://github.com/NVIDIA/ContrastiveLosses4VRD for sgcls and predcls
        eval_pair_accuracy = SGPairAccuracy(result_dict)
        eval_pair_accuracy.register_container(mode)
        evaluator['eval_pair_accuracy'] = eval_pair_accuracy

        # used for meanRecall@K
        eval_mean_recall = SGMeanRecall(result_dict, num_rel_category, dataset.ind_to_predicates, cfg.MODEL.DYHEAD.OV.ZS_PREDICATES, print_detail=True)
        eval_mean_recall.register_container(mode)
        evaluator['eval_mean_recall'] = eval_mean_recall

        # used for no graph constraint mean Recall@K
        eval_ng_mean_recall = SGNGMeanRecall(result_dict, num_rel_category, dataset.ind_to_predicates, print_detail=True)
        eval_ng_mean_recall.register_container(mode)
        evaluator['eval_ng_mean_recall'] = eval_ng_mean_recall

        eval_stagewise_recall = SGStagewiseRecall(cfg, predicates_categories, result_dict)
        eval_stagewise_recall.register_container(mode)
        evaluator['eval_stagewise_recall'] = eval_stagewise_recall

        # eval_rel_vec_recall = SGRelVecRecall(cfg, result_dict, predicates_categories)
        # eval_rel_vec_recall.register_container(mode)
        # evaluator['eval_rel_vec_recall'] = eval_rel_vec_recall

        # prepare all inputs
        global_container = {}
        global_container['zeroshot_triplet'] = zeroshot_triplet
        global_container['result_dict'] = result_dict
        global_container['mode'] = mode
        global_container['multiple_preds'] = multiple_preds
        global_container['num_rel_category'] = num_rel_category
        global_container['iou_thres'] = iou_thres
        global_container['attribute_on'] = attribute_on
        global_container['num_attributes'] = num_attributes
        
        for groundtruth, prediction in zip(groundtruths, predictions):
            evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator)
        
        # calculate mean recall
        eval_mean_recall.calculate_mean_recall(mode)
        eval_ng_mean_recall.calculate_mean_recall(mode)

        # print result
        result_str += eval_recall.generate_print_string(mode)
        result_str += eval_nog_recall.generate_print_string(mode)
        result_str += eval_zeroshot_recall.generate_print_string(mode)
        result_str += eval_ng_zeroshot_recall.generate_print_string(mode)
        result_str += eval_mean_recall.generate_print_string(mode)
        result_str += eval_ng_mean_recall.generate_print_string(mode)
        # stagewise
        result_str += eval_stagewise_recall.generate_print_string(mode)
        # result_str += eval_rel_vec_recall.generate_print_string(mode)

        """add for seen and unseen performance"""
        # calculate pre-class recall
        # eval_recall.calculate_recall_category(mode)

        def unseen_eval(cfg, evaluator, mode):
            unseen_marker = cfg.MODEL.DYHEAD.OV.ZS_PREDICATES
            if isinstance(unseen_marker[0], int):
                unseen_marker = [False for _ in range(cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES)]
                for each in cfg.MODEL.DYHEAD.OV.ZS_PREDICATES:
                    unseen_marker[each + 1] = True

            # assert "mean_recall" in evaluator.type or "recall" in evaluator.type
            res_dict = {}
            res_str = f"\nOpenworld {evaluator.type}:\n"
            if "mean_recall" in evaluator.type:
                for topk, cate_rec_list in evaluator.result_dict[f'{mode}_{evaluator.type}_list'].items():
                    part_recall = {"seen": [], "unseen": [] }
                    for idx, each_cat_recall in enumerate(cate_rec_list):
                        if unseen_marker[idx + 1]:
                            part_recall['unseen'].append(each_cat_recall)
                        else:
                            part_recall['seen'].append(each_cat_recall)
                    res_dict[f"sgdet_seen_recall/top{topk}/seen"] = np.mean(part_recall['seen'])
                    res_dict[f"sgdet_unseen_recall/top{topk}/unseen"] = np.mean(part_recall['unseen'])
                    
                    res_str += f"Top{topk:4}: unseen: {np.mean(part_recall['unseen']):.4f} " \
                            f"seen: {np.mean(part_recall['seen']):.4f} \n"


            return res_dict, res_str

        def unseen_stagewise_eval(evaluator, mode):
            unseen_marker = cfg.MODEL.DYHEAD.OV.ZS_PREDICATES
            if  isinstance(unseen_marker[0], int):
                unseen_marker = [False for _ in range(cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES)]
                for each in cfg.MODEL.DYHEAD.OV.ZS_PREDICATES:
                    unseen_marker[each + 1] = True

            res_dict = {}
            res_str = "\nStagewise Openworld part recall:\n"
            for hit_type, stat in evaluator.relation_per_cls_hit_recall.items():
                stat = stat[-1]
                recall_score = (stat[:, 0] / (stat[:, 1] + 1e-5))[1:].tolist()
                part_recall = {"seen": [], "unseen": [] }
                for idx, each_cat_recall in enumerate(recall_score):
                    if unseen_marker[idx + 1]:
                        part_recall['unseen'].append(each_cat_recall)
                    else:
                        part_recall['seen'].append(each_cat_recall)

                res_dict[f"sgdet_stagewise_openworld_recall/{hit_type}/top100/seen"] = np.mean(part_recall['seen'])
                res_dict[f"sgdet_stagewise_openworld_part_recall/{hit_type}/top100/unseen"] = np.mean(part_recall['unseen'])
                res_str += f"{hit_type}: seen: {np.mean(part_recall['seen']):.4f} " \
                        f"unseen: {np.mean(part_recall['unseen']):.4f}\n"
            res_str += '\n'

            res_str += "\nStagewise Openworld part recall Pair!!!:\n"
            for hit_type, stat in evaluator.relation_per_cls_hit_recall_pair.items():
                stat = stat[-1]
                recall_score = (stat[:, 0] / (stat[:, 1] + 1e-5))[1:].tolist()
                part_recall = {"base": [], "novel": [] }
                for idx, each_cat_recall in enumerate(recall_score):
                    if unseen_marker[idx + 1]:
                        part_recall['novel'].append(each_cat_recall)
                    else:
                        part_recall['base'].append(each_cat_recall)

                res_dict[f"sgdet_stagewise_openworld_recall_pair/{hit_type}/top100/base"] = np.mean(part_recall['base'])
                res_dict[f"sgdet_stagewise_openworld_part_recall_pair/{hit_type}/top100/novel"] = np.mean(part_recall['novel'])
                res_str += f"{hit_type}: base: {np.mean(part_recall['base']):.4f} " \
                        f"novel: {np.mean(part_recall['novel']):.4f}\n"
            res_str += '\n'

            return res_dict, res_str

        if cfg.MODEL.DYHEAD.OV.ENABLED:
            unseen_res_dict_all, unseen_res_str_all = unseen_eval(cfg, eval_recall, mode)
            unseen_res_dict, unseen_res_str = unseen_eval(cfg, eval_mean_recall, mode)
            stgw_unseen_res_dict, stgw_unseen_res_str = unseen_stagewise_eval(eval_stagewise_recall, mode)

            result_str += unseen_res_str_all
            result_str += unseen_res_str
            result_str += stgw_unseen_res_str
        """add for seen and unseen performance"""
        
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            result_str += eval_pair_accuracy.generate_print_string(mode)
        result_str += '=' * 100 + '\n'

    logger.info(result_str)
    # print(result_str)
    if output_folder:
        with open(os.path.join(output_folder, "evaluation_res.txt"), 'a') as f:
            f.write(result_str)
    if "relations" in iou_types:
        if output_folder:
            torch.save(result_dict, os.path.join(output_folder, 'result_dict.pytorch'))
        return float(np.mean(result_dict[mode + '_recall'][100]))
    elif "bbox" in iou_types:
        return float(mAp)
    else:
        return -1


def save_output(output_folder, groundtruths, predictions, dataset):
    if output_folder:
        torch.save({'groundtruths':groundtruths, 'predictions':predictions}, os.path.join(output_folder, "eval_results.pytorch"))

        #with open(os.path.join(output_folder, "result.txt"), "w") as f:
        #    f.write(result_str)
        # visualization information
        visual_info = []
        for image_id, (groundtruth, prediction) in enumerate(zip(groundtruths, predictions)):
            img_file = os.path.abspath(dataset.filenames[image_id])
            groundtruth = [
                [b[0], b[1], b[2], b[3], dataset.categories_with_bg[l]] # xyxy, str
                for b, l in zip(groundtruth.bbox.tolist(), groundtruth.get_field('labels').tolist())
                ]
            prediction = [
                [b[0], b[1], b[2], b[3], dataset.categories_with_bg[l]] # xyxy, str
                for b, l in zip(prediction.bbox.tolist(), prediction.get_field('pred_labels').tolist())
                ]
            visual_info.append({
                'img_file': img_file,
                'groundtruth': groundtruth,
                'prediction': prediction
                })
        with open(os.path.join(output_folder, "visual_info.json"), "w") as f:
            json.dump(visual_info, f)



def evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator):
    """
    Returns:
        pred_to_gt: Matching from predicate to GT
        pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
        pred_triplet_scores: [cls_0score, relscore, cls1_score]
    """
    #unpack all inputs
    mode = global_container['mode']

    local_container = {}
    local_container['gt_rels'] = groundtruth.get_field('relation_tuple').long().detach().cpu().numpy()

    # if there is no gt relations for current image, then skip it
    if len(local_container['gt_rels']) == 0:
        return

    local_container['gt_boxes'] = groundtruth.convert('xyxy').bbox.detach().cpu().numpy()                   # (#gt_objs, 4)
    local_container['gt_classes'] = groundtruth.get_field('labels').long().detach().cpu().numpy()           # (#gt_objs, )

    # about relations
    local_container['pred_rel_inds'] = prediction.get_field('rel_pair_idxs').long().detach().cpu().numpy()  # (#pred_rels, 2)
    local_container['rel_scores'] = prediction.get_field('pred_rel_scores').detach().cpu().numpy()          # (#pred_rels, num_pred_class)
    local_container['pred_rel_labels'] = prediction.get_field('pred_rel_labels').detach().cpu().numpy() 
    entity = local_container['gt_classes'][local_container['gt_rels'][:,:2]]
    local_container['gt_rel_triplets'] = np.hstack((entity, local_container['gt_rels'][:, 2].reshape(-1, 1)))

    # stagewise add
    local_container['rel_dist'] = prediction.get_field(
        'pred_rel_scores')[:, 1:].detach().cpu().numpy()  # (#pred_rels, num_pred_class) 是所有标签的预测概率 (xx, 50)
    local_container['rel_cls'] = torch.argmax(prediction.get_field(
        'pred_rel_scores')[:, 1:], dim=1).detach().cpu().numpy()  # (#pred_rels, num_pred_class) 是所有标签的预测概率 (xx,)
    if prediction.has_field('rel_vec'):
        local_container['rel_vec'] = prediction.get_field('rel_vec').detach().cpu().numpy() # (#pred_rels, 4) box, 

    # about objects
    local_container['pred_boxes'] = prediction.convert('xyxy').bbox.detach().cpu().numpy()                  # (#pred_objs, 4)
    local_container['pred_classes'] = prediction.get_field('pred_labels').long().detach().cpu().numpy()     # (#pred_objs, )
    local_container['obj_scores'] = prediction.get_field('pred_scores').detach().cpu().numpy()              # (#pred_objs, )


    # ========== CRITICAL FIX: Filter out invalid relationship indices ==========
    num_pred_objs = len(local_container['pred_classes'])
    pred_rel_inds = local_container['pred_rel_inds']

    # Find valid relationships (both subject and object indices must be within bounds)
    valid_mask = (pred_rel_inds[:, 0] < num_pred_objs) & (pred_rel_inds[:, 1] < num_pred_objs)

    if not valid_mask.all():
        num_invalid = (~valid_mask).sum()
        print(f"⚠️  Warning: Filtered {num_invalid}/{len(pred_rel_inds)} invalid relationships (indices out of bounds)")
        
        # Apply the filter to ALL relationship-related fields
        local_container['pred_rel_inds'] = pred_rel_inds[valid_mask]
        local_container['rel_scores'] = local_container['rel_scores'][valid_mask]
        local_container['pred_rel_labels'] = local_container['pred_rel_labels'][valid_mask]
        local_container['rel_dist'] = local_container['rel_dist'][valid_mask]
        local_container['rel_cls'] = local_container['rel_cls'][valid_mask]
        
        # Filter rel_vec only if it exists
        if 'rel_vec' in local_container:
            local_container['rel_vec'] = local_container['rel_vec'][valid_mask]
    # ========== END OF FIX ==========

    # to calculate accuracy, only consider those gt pairs
    # This metric is used by "Graphical Contrastive Losses for Scene Graph Parsing" 
    # for sgcls and predcls
    if mode != 'sgdet':
        evaluator['eval_pair_accuracy'].prepare_gtpair(local_container)

    # to calculate the prior label based on statistics
    evaluator['eval_zeroshot_recall'].prepare_zeroshot(global_container, local_container)
    evaluator['eval_ng_zeroshot_recall'].prepare_zeroshot(global_container, local_container)

    if mode == 'predcls':
        local_container['pred_boxes'] = local_container['gt_boxes']
        local_container['pred_classes'] = local_container['gt_classes']
        local_container['obj_scores'] = np.ones(local_container['gt_classes'].shape[0])

    elif mode == 'sgcls':
        if local_container['gt_boxes'].shape[0] != local_container['pred_boxes'].shape[0]:
            print('Num of GT boxes is not matching with num of pred boxes in SGCLS')
    elif mode == 'sgdet' or mode == 'phrdet':
        pass
    else:
        raise ValueError('invalid mode')
    """
    elif mode == 'preddet':
        # Only extract the indices that appear in GT
        prc = intersect_2d(pred_rel_inds, gt_rels[:, :2])
        if prc.size == 0:
            for k in result_dict[mode + '_recall']:
                result_dict[mode + '_recall'][k].append(0.0)
            return None, None, None
        pred_inds_per_gt = prc.argmax(0)
        pred_rel_inds = pred_rel_inds[pred_inds_per_gt]
        rel_scores = rel_scores[pred_inds_per_gt]

        # Now sort the matching ones
        rel_scores_sorted = argsort_desc(rel_scores[:,1:])
        rel_scores_sorted[:,1] += 1
        rel_scores_sorted = np.column_stack((pred_rel_inds[rel_scores_sorted[:,0]], rel_scores_sorted[:,1]))

        matches = intersect_2d(rel_scores_sorted, gt_rels)
        for k in result_dict[mode + '_recall']:
            rec_i = float(matches[:k].any(0).sum()) / float(gt_rels.shape[0])
            result_dict[mode + '_recall'][k].append(rec_i)
        return None, None, None
    """

    if local_container['pred_rel_inds'].shape[0] == 0:
        return

    # Traditional Metric with Graph Constraint
    # NOTE: this is the MAIN evaluation function, it must be run first (several important variables need to be update)
    local_container = evaluator['eval_recall'].calculate_recall(global_container, local_container, mode)

    # No Graph Constraint
    evaluator['eval_nog_recall'].calculate_recall(global_container, local_container, mode)
    # GT Pair Accuracy
    evaluator['eval_pair_accuracy'].calculate_recall(global_container, local_container, mode)
    # Mean Recall
    evaluator['eval_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)
    # No Graph Constraint Mean Recall
    evaluator['eval_ng_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)
    # Zero shot Recall
    evaluator['eval_zeroshot_recall'].calculate_recall(global_container, local_container, mode)
    # No Graph Constraint Zero-Shot Recall
    evaluator['eval_ng_zeroshot_recall'].calculate_recall(global_container, local_container, mode)
    
    # if evaluator.get('eval_rel_vec_recall') is not None:
    #     evaluator['eval_rel_vec_recall'].calculate_recall(global_container, local_container, mode)

    # stage wise recall
    if evaluator.get("eval_stagewise_recall") is not None:
        evaluator['eval_stagewise_recall'] \
            .calculate_recall(mode, global_container,
                              gt_boxlist=groundtruth.convert('xyxy').to("cpu"),
                              gt_relations=groundtruth.get_field('relation_tuple').long().detach().cpu(),
                              gt_triplets=local_container['gt_rel_triplets'],
                              pred_boxlist=prediction.convert('xyxy').to("cpu"),
                              pred_rel_pair_idx=prediction.get_field('rel_pair_idxs').long().detach().cpu(),
                              pred_rel_dist=prediction.get_field('pred_rel_scores').detach().cpu())

    return



def convert_relation_matrix_to_triplets(relation):
    triplets = []
    for i in range(len(relation)):
        for j in range(len(relation)):
            if relation[i, j] > 0:
                triplets.append((i, j, relation[i, j]))
    return torch.LongTensor(triplets) # (num_rel, 3)


def generate_attributes_target(attributes, num_attributes):
        """
        from list of attribute indexs to [1,0,1,0,...,0,1] form
        """
        max_att = attributes.shape[1]
        num_obj = attributes.shape[0]

        with_attri_idx = (attributes.sum(-1) > 0).long()
        without_attri_idx = 1 - with_attri_idx
        num_pos = int(with_attri_idx.sum())
        num_neg = int(without_attri_idx.sum())
        assert num_pos + num_neg == num_obj

        attribute_targets = torch.zeros((num_obj, num_attributes), device=attributes.device).float()

        for idx in torch.nonzero(with_attri_idx).squeeze(1).tolist():
            for k in range(max_att):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1

        return attribute_targets

def summarize_per_category(coco_eval, csv_output=None):
    '''
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    '''

    def _summarize(iouThr=None, areaRng='all', maxDets=100):
        p = coco_eval.params
        titleStr = 'Average Precision'
        typeStr = '(AP)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)
        result_str = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ], '. \
            format(titleStr, typeStr, iouStr, areaRng, maxDets)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        # dimension of precision: [TxRxKxAxM]
        s = coco_eval.eval['precision']
        # IoU
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:, :, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
            # cacluate AP(average precision) for each category
            num_classes = len(p.catIds)
            avg_ap = 0.0
            for i in range(0, num_classes):
                result_str += '{}, '.format(np.mean(s[:, :, i, :]))
                avg_ap += np.mean(s[:, :, i, :])
            # result_str += ('{} \n'.format(avg_ap / num_classes))
            result_str += ('{} \n'.format(mean_s))

        return result_str

    id2name = {}
    for _, cat in coco_eval.cocoGt.cats.items():
        id2name[cat['id']] = cat['name']
    title_str = 'metric, '
    for cid in coco_eval.params.catIds:
        title_str += '{}, '.format(id2name[cid])
    title_str += 'avg \n'
    print(title_str)

    results = [title_str]
    results.append(_summarize())
    results.append(_summarize(iouThr=.5, maxDets=coco_eval.params.maxDets[2]))
    results.append(_summarize(areaRng='small', maxDets=coco_eval.params.maxDets[2]))
    results.append(_summarize(areaRng='medium', maxDets=coco_eval.params.maxDets[2]))
    results.append(_summarize(areaRng='large', maxDets=coco_eval.params.maxDets[2]))

    with open(csv_output, 'w') as f:
        for result in results:
            f.writelines(result)