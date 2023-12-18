# CaCao
This is the official repository for the paper "Visually-Prompted Language Model for Fine-Grained Scene Graph Generation in an Open World" (Accepted by ICCV 2023)
![framework](figures/architecture.png)
# Complete code for CaCao and boosted SGG
Here we provide sample code for CaCao boosting SGG dataset in standard setting and open-world setting.
# Enhanced fine-grained predicates for VG
Download the enhanced dataset for VG training, you can use this [Google drive link](https://drive.google.com/drive/folders/1WOeumjptstD7nZQJgkJiqbQo9A_05gkh?usp=sharing).
# Dataset prepare
the [VG dataset](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html) is required and put all the images in a same folder VG_100K

download 

[VG-SGG.h5](https://svl.stanford.edu/projects/scene-graph/dataset/VG-SGG.h5)

[objects.json](https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects.json.zip)

[relationships.json](https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationships.json.zip)


[image_data.json](https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip)

put above in ./datasets/vg/ folder

put [coco2014_train](https://cocodataset.org/#download) in ./datasets/coco folder

put [Vit](https://huggingface.co/google/vit-base-patch32-224-in21k/tree/main)   in ./vit-base-patch32-224-in21k folder

put [bert pytorch.bin](https://huggingface.co/bert-base-uncased/tree/main) in ./bert-base-uncased folder

## Running Script Tutorial
```bash
# creat imdb_512.h5
python vg_to_imdb.py
```
```bash
# obtain initialized clusters for CaCao
python adaptive_cluster.py 
# establish the mapping from open-world boosted data to target predicates for enhancement
python fine_grained_mapping.py 
```
```bash
# obtain cross-modal prompt tuning models for better predicate boosting
python cross_modal_tuning.py --mode 50 
python cross_modal_tuning.py --mode all
# enhance the existing SGG dataset with our CaCao model in <pre_trained_visually_prompted_model>
python fine_grained_predicate_boosting_data_prepare.py --mode 50 
python fine_grained_predicate_boosting_data_prepare.py --mode 50

python fine_grained_predicate_boosting.py --mode all
python fine_grained_predicate_boosting.py --mode all 
```
# Quantitative Analysis
![image](https://github.com/Yuqifan1117/CaCao/assets/48062034/edd8b9bf-9d00-4f0f-894d-fce3b631fea5)
# Qualitative Analysis
![visualization](figures/visualization.png)
![visualization](figures/open-world.png)
## Predicate Boosting
![image](https://user-images.githubusercontent.com/48062034/204218380-3e2eedea-0adb-4acf-b3b6-c574c9e2dbfd.png)
## Predicate Prediction Distribution
![image](https://user-images.githubusercontent.com/48062034/204217723-3c053991-3df8-45c0-b99b-a9830cc2319e.png)
![image](https://user-images.githubusercontent.com/48062034/204218044-93bcd22e-96da-4fe7-8fb1-dacd7646d563.png)

## Acknowledgement
The SGG part code is implemented based on [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch), [FGPL](https://github.com/XinyuLyu/FGPL), and [SSRCNN(One-Stage)](https://github.com/MCGNJU/Structured-Sparse-RCNN). Thanks for their great works! 
## 📜 Citation
If you find this work useful for your research, please cite our paper and star our git repo:
```bibtex
@article{yu2023visually,
  title={Visually-Prompted Language Model for Fine-Grained Scene Graph Generation in an Open World},
  author={Yu, Qifan and Li, Juncheng and Wu, Yu and Tang, Siliang and Ji, Wei and Zhuang, Yueting},
  journal={arXiv preprint arXiv:2303.13233},
  year={2023}
}
```
or
```bibtex
@inproceedings{yu2023visually,
  title={Visually-Prompted Language Model for Fine-Grained Scene Graph Generation in an Open World},
  author={Yu, Qifan and Li, Juncheng and Wu, Yu and Tang, Siliang and Ji, Wei and Zhuang, Yueting},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```
