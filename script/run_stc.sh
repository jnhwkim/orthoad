# mSTC
source ./script/setup.sh STC_Test
python train.py --dataset mstc --model resnet18 --k 100 --metric auroc --fpr 1.0 --seed 5555
