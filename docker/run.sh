CONTAINER_NAME=$1

cd /mnt/cloudy_3/SigMA/engineering/atsushi/mmpretrain
docker rm -f $CONTAINER_NAME && \
docker run \
    --gpus all \
    --shm-size=64gb \
    -v /mnt:/mnt \
    -itd \
    --name $CONTAINER_NAME \
    mmpretrain:pytorch1.12.1-cuda11.3-cudnn8-mmcv2.0.1-mmengine0.8.4

docker exec -it $CONTAINER_NAME bash -c \
    "cd $PWD && pip install -r requirements/optional.txt && pip install --no-cache-dir -e ."
