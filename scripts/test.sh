#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1
export TRANSFORMERS_CACHE=MODEL/
echo "Testing!!!!"
# MODEL_NAME="11-6-SGDet-close-set-RAHP-without-distillation-loss"
MODEL_NAME="glip_zeroshot_test"

# python tools/test_grounding_net.py \
#     --task_config configs/vg150/finetune.yaml \
#     --config-file configs/pretrain/glip_Swin_T_O365_GoldG.yaml \
#     SOLVER.IMS_PER_BATCH $(( 1 * ${NUM_GPUS} )) \
#     TEST.IMS_PER_BATCH $(( 1 * ${NUM_GPUS} )) \
#     MODEL.DYHEAD.RELATION_REP_REFINER False \
#     MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False \
#     MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
#     MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
#     DATASETS.VG150_OPEN_VOCAB_MODE False \
#     MODEL.WEIGHT MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth \
#     MODEL.LANGUAGE_BACKBONE.MODEL_TYPE bert-base-uncased \
#     MODEL.LANGUAGE_BACKBONE.MODEL_DIR MODEL/bert-base-uncased \
#     MODEL.LANGUAGE_BACKBONE.TOKENIZER_LOCAL_FILES_ONLY True \
#     OUTPUT_DIR ./OUTPUT/glip_zeroshot_test


CUDA_VISIBLE_DEVICES=3 python tools/test_grounding_net.py \
    --task_config configs/vg150/finetune.yaml \
    --config-file configs/pretrain/glip_Swin_T_O365_GoldG.yaml \
    MODEL.WEIGHT MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth \
    MODEL.LANGUAGE_BACKBONE.MODEL_DIR MODEL/bert-base-uncased \
    MODEL.LANGUAGE_BACKBONE.TOKENIZER_LOCAL_FILES_ONLY True \
    MODEL.DYHEAD.OV.DYNAMIC_CLIP_CLASSIFIER_WEIGHT_CACHE_PTH "MODEL/relation_aware_weight_cache.pth" \
    OUTPUT_DIR ./OUTPUT/rahp_paper_30entities_test