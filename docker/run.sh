CONTAINER_NAME=$1
USER_ID=$(id -u)
GROUP_ID=$(id -g)
USER_NAME=$(whoami)

cd /mnt/SigMA/engineering/atsushi/mmpretrain
docker rm -f $CONTAINER_NAME && \
docker run \
    --gpus all \
    --shm-size=64gb \
    -v /mnt:/mnt \
    -itd \
    --name $CONTAINER_NAME \
    mmpretrain:pytorch1.12.1-cuda11.3-cudnn8-mmcv2.0.1-mmengine0.9.0

docker exec -it $CONTAINER_NAME bash -c \
    "groupadd -g $GROUP_ID $USER_NAME && useradd -d $PWD -u $USER_ID -g $GROUP_ID $USER_NAME"

docker exec -it $CONTAINER_NAME bash -c \
    "cd $PWD && pip install -r requirements/optional.txt && pip install --no-cache-dir -e ."
