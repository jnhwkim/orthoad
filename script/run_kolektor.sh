# KolektorSDD
source ./script/setup.sh KolektorSDD
python train.py --dataset kolektor --model resnet18 --k 100 --metric auroc --fpr 1.0 --fold 0
python train.py --dataset kolektor --model resnet18 --k 100 --metric auroc --fpr 1.0 --fold 1
python train.py --dataset kolektor --model resnet18 --k 100 --metric auroc --fpr 1.0 --fold 2

# KolektorSDD2
source ./script/setup.sh KolektorSDD2
python train.py --dataset kolektor2 --model resnet18 --k 100 --metric auroc --fpr 1.0