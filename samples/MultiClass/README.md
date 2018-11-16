Added code for multiclasses, inspired from https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46

## Command to run training : 
python -W ignore multiclass.py train --dataset=C:/Mask_RCNN/datasets/multiclass --weights=coco

## Structure of Dataset folder
* C:/Mask_RCNN/datasets/multiclass/
** train/
*** <all_images>
*** "via_region_data.json" containing tagged annotations
** val/
*** <all_images>
*** "via_region_data.json" containing tagged annotations
