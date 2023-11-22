work_dir=work_dirs/maxvit_2xb32_in1k-384px_e100_new/20231115_082634
config=$work_dir/vis_data/config.py
pred_result=$work_dir/vis_data/pred.pkl

python tools/analysis_tools/confusion_matrix.py \
    $config \
    $pred_result \
    --show-path $work_dir/vis_data/confusion_matrix.png \
    --cmap Blues \
    --include-values
