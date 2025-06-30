# uLayout: Unified Room Layout Estimation for Perspective and Panoramic Images
<div style="text-align: center;">
    Jonathan Lee<sup>1</sup>,
    Bolivar Solarte<sup>1</sup>,
    Chin-Hsuan Wu<sup>1</sup>,
    Jin-Cheng Jhang<sup>1</sup>,
    Fu-En Wang<sup>1</sup>,
    Yi-Hsuan Tsai<sup>2</sup>,
    Min Sun<sup>1</sup>
</div>

<div style="text-align: center;">
  <!-- affiliations -->
  <p class="subtitle is-5" style="font-size: 1.0em; text-align: center;"> 
    <sup>1</sup> National Tsing Hua University, Taiwan
    <sup>2</sup> Atmanity Inc.
  </p>
</div>


<p align="center"> 
  <a href="https://arxiv.org/abs/2503.21562" target='_**blank**'>
    <img src="https://img.shields.io/badge/arXiv-2503.21562-b31b1b.svg" alt="arXiv">
    </a>
  <a href="https://openaccess.thecvf.com/content/WACV2025/supplemental/Lee_uLayout_Unified_Room_WACV_2025_supplemental.pdf" target='_**blank**'>
    <img src="https://img.shields.io/badge/supplemental-blue.svg" alt="supplemental">
    </a>
</p>


<div style="text-align: center;">
  <img src="./teaser.png" alt="teaser" width="100%">
</div>

## 🚀 News
- [2025/03/27] Paper released on arXiv.
- [2025/06/30] Code released on GitHub.

## ℹ️ installation
```
git clone https://github.com/JonathanLee112/uLayout.git
cd uLayout
conda create -n uLayout python=3.10
conda activate uLayout
pip install -r requirements.txt
cd external
git submodule add https://github.com/jinlinyi/PerspectiveFields external/PerspectiveFields
git submodule update --init --recursive
```
## 📂 Dataset
## Matterport3D
Office MatterportLayout dataset is at [here](https://github.com/ericsujw/Matterport3DLayoutAnnotation). We transfer the annotations to the format of 2D pixel corner. The ground truth is at [here](https://drive.google.com/drive/u/1/folders/1sBRLlsPgh8s3J1c7YxTr2-Efme5l9oKb)
Please put the dataset in `src/pano/mp3d` directory.
```
|-- train
|   |-- img
|   |   |-- 1LXtFkjw3qL_4b77e304d83943999198c3cd4457512c.png
|   |-- label_cor
|   |   |-- 1LXtFkjw3qL_4b77e304d83943999198c3cd4457512c.txt
|-- val  # validation set follows the same structure as train
|-- test # test swt follows the same structure as train
```
## PanoContext and Stanford 2D-3D
We use the same preprocessed pano/S2D3D dataset provided by [HorizonNet](https://github.com/sunset1995/HorizonNet#dataset).
You can also download the dataset directly from [this link](https://drive.google.com/drive/u/1/folders/1sBRLlsPgh8s3J1c7YxTr2-Efme5l9oKb).
Please put the dataset in `src/pano/pano_st2d3d` directory.
```
|-- train
|   |-- img
|   |   |-- camera_00d10d86db1e435081a837ced388375f_office_24_frame_equirectangular_domain_.png
|   |-- label_cor
|   |   |-- camera_0000896878bd47b2a624ad180aac062e_conferenceRoom_3_frame_equirectangular_domain_.txt
|-- val  # validation set follows the same structure as train
|-- test # test swt follows the same structure as train
```
## LSUN
The original LSUN dataset is from LSUN layout challenge. However, the link is no longer available. The current available dataset is from [lsun-room](https://github.com/leVirve/lsun-room).
You can download the dataset directly from [this link](https://drive.google.com/drive/u/1/folders/1sBRLlsPgh8s3J1c7YxTr2-Efme5l9oKb). 

In folder `lsun_ori`, we provide the LSUN dataset that the images resized to 640x640 and the annotations are in the format of ceiling, floor, ceiling-wall, floor-wall corners
In folder `lsun_align`, we provide the LSUN dataset that the images already preprocessed and project to equirectangular format(img_aligned_pano) and the annotation also processed in the same way(boundary_aligned_pano).
Please put the dataset in `src/pp` directory. 
```
# In folder lsun_aligned.  The datastructure is as follows:
|-- train
|   |-- img_aligned_pano 
|   |   |-- 0a5c601a476b916a5a2b09513c301e52b2a92afc.png
|   |-- boundary_aligned_pano
|   |   |-- 0a5c601a476b916a5a2b09513c301e52b2a92afc.json
|   |-- train_lsun_pred_horizon.json
|   |-- train_scene_list.json
|-- val  # validation set follows the same structure as train
# we don't provide test set because the LSUN dataset don't provide the test set ground truth.
```
Preprocessed LSUN dataset can follow three steps:
1. vertical image alignment which ensures that all vertical structures are aligned to the Y-axis in perspective domain.
```bash
cd preprocessing/perspective/alignment
python lsun_preprocess.py output_dir=output/pp_align lsun.data_mode=validation save_pd=true

# output_dir: the directory to save the preprocessed images and annotations.
# lsun.data_mode: the mode of the dataset, can be training or validation.
# save_pd: whether to save data.
```
2. transfer perspective images to panoramic images.
```bash
cd preprocessing/perspective/transfer2pano
python transfer2pano.py lsun.data_dir=src/pp/lsun_align lsun.mode=val out_dir=output/pp_align

# data_dir: the directory of the preprocessed LSUN dataset.(step 1 output)
# mode: the mode of the dataset, can be train or val.
# out_dir: the directory to save the panoramic images and annotations.
```
3. utilize the PerspectiveFields to predict the horizon for the image that only has the ceiling or floor boundary.
```bash
python lsun_pred_horizon.py --original_data_dir src/pp/lsun_ori/images_ori --data_mode train --output_dir output/pred_horizon

# --original_data_dir: the directory of the original LSUN dataset.
# --data_mode: the mode of the dataset, can be train or val.
# --output_dir: the directory to save the predicted horizon.
```
Please put the step 3 output in the same directory as step 2 output like `lsun_align`.

## 📈 Training
To train the model, you can use the following command:
```bash
python main.py pano_dataset=mp3d pp_dataset=lsun  mode=train

# pano_dataset: the dataset for panoramic images, can be mp3d or pano or st2d3d.
# pp_dataset: the dataset for perspective images, can be lsun
# split: the split of the dataset, can be train or val or test.
```
## 🧪 Evaluation
To evaluate the model, you can use the following command:
```bash
python main.py pano_dataset=mp3d pp_dataset=lsun mode=test ckpt_dir=ckpt/best_mp3d.pth

# --ckpt_dir: the path to the checkpoint file.
```
## 📊 Visualization
To visualize the results, you can use the following command:
```bash
python main.py pano_dataset=mp3d pp_dataset=lsun mode=test ckpt_dir=ckpt/best_mp3d.pth vis=true save_pred=true

# vis: whether to visualize the 2D results.
# save_pred: whether to save the predicted boundary. (For the 3D visualization purpose)(only work for pano dataset)
``` 
## 🖼️ 3D Layout Viewer
After save the predicted boundary, you can use the 3D layout viewer to visualize the results.
```bash
cd 3d_layout_viewer
python layout_viewer.py --dataset_dir src/pano/mp3d --dataset mp3d --mode test --index 0 
--layout output/ulayout_mp3d_lsun_test/inference_img/panorama_pred_boundary  --vis --ignore_ceiling --ignore_wireframe

# --dataset_dir: the directory of the dataset, can be src/pano/mp3d or src/pano/pano_st2d3d.
# --dataset: the dataset, can be mp3d or pano or st2d3d.
# --mode: the mode of the dataset, can be train or val or test.
# --index: the index of the json file to visualize. The order of the json file is the same as the order of the images in the dataset.
# --layout: the path to the predicted boundary json file.
# --vis: whether to visualize the results.
# --ignore_ceiling: whether to ignore the ceiling in the visualization.
# --ignore_wireframe: whether to ignore the wireframe in the visualization.
```

## 📝 Citation
If you find our work useful in your research, please consider citing:
```bibtex
@InProceedings{Lee2025uLayout,
    author    = {Lee, Jonathan and E Solarte, Bolivar and Wu, Chin-Hsuan and Jhang, Jin-Cheng and Wang, Fu-En and Tsai, Yi-Hsuan and Sun, Min},
    title     = {uLayout: Unified Room Layout Estimation for Perspective and Panoramic Images},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {8399-8408}
}
```