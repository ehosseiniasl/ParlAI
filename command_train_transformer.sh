
GPU=5
CHECKPOINT=/export/home/ehsan/ParlAI/checkpoints
LAYER=2
HEAD=2
DK=256
DV=256
D_MODEL=256
D_INNER=1024
D_WORD=256
EXPERIMENT=convai2_transformer_l${LAYER}_h${HEAD}
MODEL_FILE=$CHECKPOINT/$EXPERIMENT
TAG=model,numlayers,n_head,d_word_vec,d_model,d_inner,d_k,d_v

CUDA_VISIBLE_DEVICES=$GPU python projects/convai2/baselines/transformer/train.py --model_file $MODEL_FILE --tensorboard_tag $TAG --numlayers $LAYER --n_head $HEAD --d_word_vec $D_WORD --d_model $D_MODEL --d_inner $D_INNER --d_k $DK --d_v $DV

