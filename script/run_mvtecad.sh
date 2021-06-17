source ./script/setup.sh MVTec_AD

GPU=0
OPTS="$2"

TEXTURES="carpet grid leather tile wood"
OBJECTS_A="bottle cable capsule hazelnut metal_nut"
OBJECTS_B="pill screw toothbrush transistor zipper"
ALL="${TEXTURES} ${OBJECTS_A} ${OBJECTS_B}"

if [[ $1 = TEX ]]; then
    TARGET=$TEXTURES
elif [[ $1 = OBJA ]]; then
    TARGET=$OBJECTS_A
elif [[ $1 = OBJB ]]; then
    TARGET=$OBJECTS_B
elif [[ $1 = ALL ]]; then
    TARGET=$ALL
else
    echo "$0 [TEX|OBJA|OBJB|ALL] [<args>]"
fi

for CATEGORY in $TARGET
do
    echo "CUDA_VISIBLE_DEVICES=$GPU python train.py --dataroot $DATA --category $CATEGORY $OPTS"
    CUDA_VISIBLE_DEVICES=$GPU python train.py --dataroot $DATA --category $CATEGORY $OPTS
done
