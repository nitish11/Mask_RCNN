Added code for multiclasses, inspired from https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46

## Command to run training : 
python -W ignore multiclass.py train --dataset=C:/Mask_RCNN/datasets/multiclass --weights=coco

## Structure of Dataset folder
-C:/Mask_RCNN/datasets/multiclass/ <br/>
--train/<br/>
---<all_images><br/>
---"via_region_data.json" containing tagged annotations<br/>
--val/<br/>
---<all_images><br/>
---"via_region_data.json" containing tagged annotations<br/>

## Structure of JSON File containing annotation
--Dictionary of images as keys and annotation as values <br/>
--- Each annotation is another dictionary 
--- Ex : annotations['0634332006.jpg15119'].keys() <br/>
--- dict_keys(['filename', 'size', 'file_attributes', 'regions'])<br/>
---- annotations['0634332006.jpg15119']['filename'] = '0634332006.jpg'
---- annotations['0634332006.jpg15119']['size'] = 15119
----- annotations['0634332006.jpg15119']['regions'][0].keys() = dict_keys(['shape_attributes', 'region_attributes'])
----- annotations['0634332006.jpg15119']['regions'][0]['region_attributes'] = {'class1':'', 'class2':'yes', 'class3':''}
----- annotations['0634332006.jpg15119']['regions'][0]['shape_attributes'].keys() = dict_keys(['name', 'all_points_y', 'all_points_x'])
------ annotations['0634332006.jpg15119']['regions'][0]['shape_attributes']['all_points_x'] = [153,163,187,199,210,226,233,245,246,256,253,242,233,222, 202,181,163,155,146]

