CONTAINER_NAME=$1

cd /data2/src/atsushi/mmpretrain
docker rm -f $CONTAINER_NAME && \
docker run \
    --gpus all \
    --shm-size=64gb \
    --restart unless-stopped\
    -v /data1:/data1 \
    -v /data2:/data2 \
    -itd \
    --name $CONTAINER_NAME \
    mmpretrain:pytorch1.12.1-cuda11.3-cudnn8-mmcv2.0.1-mmengine0.8.4

docker exec -it $CONTAINER_NAME bash -c \
    "cd $PWD && pip install -r requirements/optional.txt && pip install --no-cache-dir -e ."
