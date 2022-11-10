# Classifyer_Irislab
```
git clone https://github.com/hanu14/Classifyer_Irislab.git

cd Classifyer_Irislab
conda env create -f environment.yaml
conda activate ClassifyNet
```

```
cd data_yfcc
tar -xvf yfcc_json.tar.gz
tar -xvf yfcc_images.tar.gz
```

```
cd ../scripts
./download_pretrained_resnet_101.sh
./extract_yfcc_features.sh
```

```
cd ..
./train.sh
```

```
./eval.sh
```
