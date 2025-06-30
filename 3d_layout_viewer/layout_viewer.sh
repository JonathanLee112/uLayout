# choose dataset and dataset_dir

# mp3d dataset
# dataset=mp3d
# dataset_dir=src/pano/mp3d

# pano dataset
dataset=pano
dataset_dir=src/pano/pano_st2d3d

# st2d3d dataset
# dataset=st2d3d
# dataset_dir=src/pano/pano_st2d3d

# first run python main.py to generate the predicted boundary, then set the layout path to the predicted boundary json file
output_dir=output/ulayout_mp3d_lsun_test/inference_img/layout_3d
layout=output/ulayout_mp3d_lsun_test/inference_img/panorama_pred_boundary  

mode=test  # train, val, or test
index=0  # index of the json file to visualize

python layout_viewer.py \
--dataset $dataset \
--dataset_dir $dataset_dir \
--mode $mode \
--index $index \
--layout $layout \
--vis \
--ignore_ceiling \
--ignore_wireframe
