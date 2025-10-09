**Quasi-multimodal-based pathophysiological feature learning for retinal disease diagnosis**<br/>
Lu Zhang, Mengyu Jia, Huizhen Yu, Zuowei Wang, Fu Gui, Yatu Guo, and Wei Zhang<br/>


## Training and Running QMP-RETNet ##

Python version 3.8 is required and all major packages used and their versions are listed in `requirements.txt`.

### QMP-RETNet on MuReD Dataset ###
Download MuReD data (5.68G)
```
wget https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/pc4mb3h8hz-1.zip
```

Train New Model
```
python train.py  --batch_size 16  --lr 0.00001 --classes 20  --dataset_name 'MuReD' --dataroot data/
```

### QMP-RETNet on DDR Dataset ###
Download DDR data (14G)
```
https://drive.google.com/drive/folders/1z6tSFmxW_aNayUqVxx6h6bY4kwGzUTEC
```

Train New Model
```
python train.py  --batch_size 16  --lr 0.00001 --classes 5  --dataset_name 'DDR' --dataroot data/
```

## Citing ##

```
