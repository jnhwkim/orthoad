# Semi-Orthogonal Embedding for Efficient Unsupervised Anomaly Segmentation

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/semi-orthogonal-embedding-for-efficient/anomaly-detection-on-mvtec-ad)](https://paperswithcode.com/sota/anomaly-detection-on-mvtec-ad?p=semi-orthogonal-embedding-for-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/semi-orthogonal-embedding-for-efficient/unsupervised-anomaly-detection-on-kolektorsdd)](https://paperswithcode.com/sota/unsupervised-anomaly-detection-on-kolektorsdd?p=semi-orthogonal-embedding-for-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/semi-orthogonal-embedding-for-efficient/unsupervised-anomaly-detection-on)](https://paperswithcode.com/sota/unsupervised-anomaly-detection-on?p=semi-orthogonal-embedding-for-efficient)

We use the semi-orthogonal embedding for unsupervised anomaly segmentation. The multi-scale features from pre-trained CNNs are recently used for the localized Mahalanobis distances with significant performance. Here, we aim for robust approximation, cubically reducing the computational cost for the inverse of multi-dimensional covariance tensor. The proposed method achieves a new state-of-the-art with a significant margin for the MVTec AD (.942 and .982 for PRO and ROC, respectively), KolektorSDD, KolektorSDD2, and mSTC datasets.

## Requirements

- PyTorch 1.2 (not tested for < 1.2)
- Install dependencies using

```bash
conda install --file requirements.txt
```

or 

```bash
pip install -r requirements.txt
apt-get install libxrender1 libsm6 libglib2.0-0 libxext6  # for opencv
```

### MobileNetv3

```bash
git clone https://github.com/d-li14/mobilenetv3.pytorch.git ../
ln -s ../mobilenetv3.pytorch/mobilenetv3.py mobilenetv3.py
```

## Dataset

For the MVTec AD dataset, please download [MVTec AD dataset](ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz) and place under `--dataroot` path.

For the Kolektor Surface-Defect Dataset (KolektorSDD), please visit [this site](http://www.vicos.si/Downloads/KolektorSDD) and [this site](http://www.vicos.si/Downloads/KolektorSDD2).

For the ShaghaiTech Campus dataset (mSTC), the link for the [official site](https://svip-lab.github.io/dataset/campus_dataset.html) was broken. So, please use the [Baidu Disk](https://pan.baidu.com/s/1j0TEt-2Dw3kcfdX-LCF0YQ#list/path=%2Fdatasets%2FShanghaiTechDataset) link introduced on the [MLEP github page](https://github.com/svip-lab/MLEP) to obtain the dataset and place them under `--dataroot` path. 
If you succesfully download the dataset, run the script by `bash stc_preprocess.sh` to preprocess the dataset. The script unzip .zip files under `./converted` and converts the training videos (.avi format) and pixel masks (.npy format) into frames (.jpg format) under `./archive`. It takes about 1.5 hour on Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz.

## Training

### MVTec AD

The environment variable `DATA` is used for the option `--dataroot`. For the [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad/), `source ./script/setup.sh MVTec_AD` will recursively find the path to `MVTec_AD` directory and set the environment variable.
For the KolektorSDD and KolektorSDD2, a similar approach would be working.

Please run `train.py` scripts with the category option, which performs the evaluation afterward. *You might need 12G+ GPU memory to run this script.*

```bash
python train.py --category carpet --metric auproc --fpr 0.3 # aurpoc
python train.py --category carpet --metric auroc --fpr 1.0  # auroc
```

### KolektorSDD

The below script run for the three folds of KolektorSDD dataset and the KolektorSDD2 dataset.

```bash
./script/run_kolektor.sh
```

### mSTC

For the preprocess, please run `./tools/stc_preprocess.sh` as described above.

```bash
./script/run_stc.sh
```

For more options, please run:

```bash
python train.py -h
```

## Visualization

After running `train.py`, run the below command to visualize the results using `matplotlib`. The PDF file will be located under a given path with `--ckpt`.

```bash
python visualizer.py --category carpet --ckpt /path/to/save
```

## Performance

The previous work [Bergmann'19] proposes a threshold-free metric based on the per-region overlap (PRO). This metric is the area under the receiver operating characteristic curve (ROC) while it takes the average of true positive rates for each connected component in the ground truth. Because the score of a single large region can overwhelm those of small regions, the PRO promotes multiple regions' sensitivity. It calculates up to the false-positive rate of 30% (100% for ROC, of course). The ROC is a natural way to cost-and-benefit analysis of anomaly decision making.

#### MVTec AD

Model                 |   PRO   |   ROC
----------------------|---------|---------
L2-AE [Bergmann'20]   |  .790   |  .820
SSIM-AE [Yi'20]       |    -    |  .818
Student [Bergmann'20] |  .857   |    -
VE VAE [Liu'20]       |    -    |  .861
VAE Proj [Dehaene'20] |    -    |  .893
Patch-SVDD [Yi'20]    |    -    |  .957
SPADE [Cohen'20]      |  .917   |  .965
PaDiM [Defard'20]     |  .921   |  .979
Ours                  |**.942** | **.982**

Notice that `SSIM-AE` reports are not consistent in [Bergmann'20] and [Yi'20]. :confused:

#### Unsupervised KolektorSDD and KolektorSDD2

We use only anomaly-free images for unsupervised training. For the ResNet-18 with k=100,

Model                 | Fold 1 | Fold 2 | Fold 3 | Avg (Std)     | KolektorSDD2
----------------------|--------|--------|--------|---------------|--------------
Student [Bergmann'20] | .904   | .883   | .902   | .896 (.012)   | .950 (.005)
PaDiM [Defard'20]     | .939   | .935   | .962   | .945 (.015)   | .956
Ours                  |**.953**|**.951**|**.976**|**.960 (.014)**|**.981**

#### mSTC

|             Model            |    ROC   |
|:----------------------------:|:--------:|
| CAVGA-RU [Venkataramanan'19] |    .85   |
|       SPADE [Cohen'20]       |   .899   |
|       PaDiM [Defard'20]      |   .912   |
|           **Ours**           | **.921** |

## License

GNU General Public License version 3.0
