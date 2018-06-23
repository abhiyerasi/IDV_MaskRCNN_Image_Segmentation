# IDV Segmentation Example

This is an example showing the use of Mask RCNN in a real application.
We train the model to detect Idly Vada Dosa , and then we use the generated 
masks to predict the new images.

IDV - Idly_Dosa_Vada


## Installation
From the [Releases page](https://github.com/matterport/Mask_RCNN/releases) page:
1. Download `mask_rcnn_coco.h5`. Save it in the root directory of the repo (the `mask_rcnn` directory).
2. Download IDV images and the by using the VGG Image anotation tool and polygon masks to be created. Put that in the path `mask_rcnn/datasets/idv/` for test and `mask_rcnn/samples/idvscript/` for train and val.


## Run Jupyter notebooks
Open the `inspect_idv_data.ipynb` or `inspect_idv_model.ipynb` Jupter notebooks. You can use these first notebooks to explore the dataset and chek the masks which we created and making up any sense or not and second one to run through the prediction on the test dataset.


## Train the Balloon model

Train a new model starting from pre-trained COCO weights
```
python main.py train --dataset=/path/to/idv/dataset --weights=coco
```

Resume training a model that you had trained earlier
```
python main.py train --dataset=/path/to/idv/dataset --weights=last
```

Resume training a model that with specified pre trained weight
```
python main.py train --dataset=/path/to/idv/dataset --weights=/path/to/mask_rcnn/mask_rcnn_idv.h5 
```

Train a new model starting from ImageNet weights
```
python main.py train --dataset=/path/to/idv/dataset --weights=imagenet
```

The code in `main.py` is set to train for 3k steps (50 epochs of 60 steps each), and using a batch size of 2. 
Update the schedule to fit your needs.

## Changes in the main.py code

My IDVDataset class looks like this:

class IDVDataset(utils.Dataset):
    def load_idv(self, dataset_dir, subset):
        ...
    def load_mask(self, image_id):
        ...
    def image_reference(self, image_id):
        ...

load_idv reads the JSON file, extracts the annotations, and iteratively calls the internal add_class and add_image functions to build the dataset.

load_mask generates bitmap masks for every object in the image by drawing the polygons.

image_reference simply returns a string that identifies the image for debugging purposes. Here it simply returns the path of the image file.

## Techniques Used:

1. Just with out pretrained weight ran the model but was not converging on small dataset also.
2. Then used the coco pretrained weights given by mask rcnn and trained the model and within 3k steps the model got converged and the error on train was low when compared to validation.
3. Then used the data augmentation technique as the image count was very less and trained the model again using coco weight and the did well onunseen test data also.
4. By setting rpn_loss to 0 but the model loss and val loss was less but the not predicting the masks properly.
