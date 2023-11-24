work_dir=$1
config=$work_dir/vis_data/config.py
checkpoint=$2

python tools/test.py \
    $config \
    $checkpoint \
    --work-dir $work_dir \
    --out $work_dir/vis_data/pred.pkl \
    --out-item pred \
    --show-dir $work_dir/vis_data/pred \
    --cfg-options "test_evaluator.average=None" "test_evaluator.items=[precision,recall,f1-score,support]"
