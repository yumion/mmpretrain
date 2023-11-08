work_dir=work_dirs/maxvit_2xb32_in1k-384px_e100/20231025_071533
config=$work_dir/vis_data/config.py
checkpoint=work_dirs/maxvit_2xb32_in1k-384px_e100/epoch_100.pth

tools/dist_test.sh \
    $config \
    $checkpoint \
    1 \
    --work-dir $work_dir \
    --out $work_dir/vis_data/pred.pkl \
    --out-item pred \
    --show-dir $work_dir/vis_data/pred
