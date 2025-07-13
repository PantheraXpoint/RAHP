#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1
echo "Testing!!!!"
MODEL_NAME="11-6-SGDet-close-set-RAHP-without-distillation-loss"

python -m torch.distributed.launch \
    --master_port 8888 --nproc_per_node=${NUM_GPUS} \
    tools/test_grounding_net.py \
    --task_config configs/vg150/finetune.yaml \
    --config-file configs/pretrain/glip_Swin_T_O365_GoldG.yaml \
    SOLVER.IMS_PER_BATCH $(( 1 * ${NUM_GPUS} )) \
    TEST.IMS_PER_BATCH $(( 1 * ${NUM_GPUS} )) \
    MODEL.DYHEAD.RELATION_REP_REFINER False \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    DATASETS.VG150_OPEN_VOCAB_MODE False \
    MODEL.WEIGHT /storage/data/v-liutao/VS3/OUTPUT4/${MODEL_NAME}/model_0048000.pth \
    OUTPUT_DIR /storage/data/v-liutao/VS3/OUTPUT4/${MODEL_NAME}
