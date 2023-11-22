work_dir=work_dirs/maxvit_2xb32_in1k-384px_e100_new/20231115_082634
config=$work_dir/vis_data/config.py
checkpoint=work_dirs/maxvit_2xb32_in1k-384px_e100_new/20231115_082634/best_single-label_f1-score_epoch_94.pth

python tools/test.py \
    $config \
    $checkpoint \
    --work-dir $work_dir \
    --out $work_dir/vis_data/pred.pkl \
    --out-item pred \
    --show-dir $work_dir/vis_data/pred
