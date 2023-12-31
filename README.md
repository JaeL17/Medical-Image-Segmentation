# Medical-Image-Segmentation
## Overview
Welcome to the repository for Medical Image Segmentation, a crucial task in the field of computer vision that empowers healthcare professionals with advanced tools for precise disease diagnosis and optimisation of treatment strategies. This project focuses on semantic segmentation in medical imaging, automating the segmentation of the organ cells on MRI scans. In recent years, there have been remarkable advancements in image segmentation, with Transformers based architectures or hybrid architectures (integrating Transformer blocks and convolutional blocks) emerging as State-of-the-Art solutions, overtaking ConvNets as the favoured choice for backbones. However, ConvNeXts, a pure ConvNet model, demonstrated competitive performance with Transformers across multiple computer vision benchmarks in terms of accuracy and scalability. In this repository, we fine-tune computer vision models of different architectures such as Segformers, DPT, and UPerNet (ConvNeXt backbone) to achieve advanced semantic segmentation performance in medical imaging.

## Dataset Description
The dataset is sourced from "UW-Madison GI Tract Image Segmentation" Kaggle competition. You can access the dataset here: [dataset](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/overview). The images in this dataset are provided in the form of 16-bit grayscale PNG format. and the ground-truth segmentation annotations are provided using Run-Length Encoding (RLE) in a CSV file. The dataset consists of 3 classes: Stomach, Small Bowel, and Large Bowel. In total, there are 16,590 images with annotations for the specified classes with the following class distribution.

* **Stomach**: 8,627 images (52% of the dataset) have annotations for this class.
* **Small Bowel**: 11,201 images (67.5% of the dataset) have annotations for this class.
* **Large Bowel**: 14,085 images (84.9% of the dataset) have annotations for this class.  

## Contents
1. **utils.py**: Contains code for loading training and validation dataset, as well as pre-processing the data and data augmentation on training dataset.
2. **model.py**: Exlore this file to find code for the segmentation model class used in this project. 
3. **trainer.py**: Code for transfer learning open-source models from Hugging Face on the training dataset. This section also includes computing Dice coefficients on validation dataset.
4. **visualisation.ipynb**: Explore this Jupyter Notebook for code related to displaying sample images and inference resylts of the segmentation model.

## Data Pre-processing and Training Data Augmentation
The dataset is divided into training and validaiton sets with 8:2 ratio. During the training phase, we adopt multiple data augmentation strategies to enhance the generalisation capabilities of models. The key pre-processing and augmentation techniques include:
1. **Image Resizing**: All images are resized to a resolution of 288x288 pixels to ensure uniformity.
2. **Random Flipping**: Both horizontal and vertical fillping are applied randomly to expose the model to variations in orientation.
3. **Brightness and Contrast Adjustments**: Random adjustments to brightness and contrast are applied to simulate diverse lighting conditions.
4. **Coarser Dropout for Regularisation**: Coarser dropout is applied to introduce regularisation during training, preventing overfitting and improving the model's ability to generalise to unseen data.
5. **Random Scaling, Shifting, and Rotation**: Random scaling, shifting, and rotation are applied to improve the model's robustness against spatial transformations.

## Running the trainer code
- Shell script
```
bash run_train.sh
```

* Python
```
CUDA_VISIBLE_DEVICES=2 python trainer.py \
    --base_model "openmmlab/upernet-convnext-small" \
    --train_batch_size 32 \
    --weight_decay 1e-4 \
    --optimizer "AdamW"\
    --scheduler_name "MultiStepLR"\
    --epochs 100 \
    --lr 2e-4 >> ./logs/train_upertnet_small.log &
```

* Logging
```
tail -f logs/train_upertnet_small.log
```

## Evaluation Results and Performance Comparison
In this project, we compare the performance different models with varying parameter sizes for semantic segmentation based on their Dice coefficient scores. The Dice coefficient is a measure of similarity between two samples, with a value of 1 indicating perfect overlap.
* **SegFormer**: The encoder part of SegFormer can be scaled from b0 to b5 by adjusting the number of layers or the dimensions of encoder blocks. The SegFormer models show a clear trend of increasing Dice coefficients as the model complexity (parameter size) increases, with scoring ranging from 0.8813 for the b0 variant to 0.9297 for b5. 
* **UperNet (ConvNeXt backbone)**: Among the evaluated models, the UperNet (ConvNeXt backbone) large stands out with the highest Dice coefficient of 0.9494. Interestingly, increasing the parameters of UperNet does not yield a proportionally significant performance improvement compared to SegFormer models.
* **DPT**: Despite its significantly larger parameter size, the DPT large model achieves a Dice coefficient of 0.9420, slightly below the UperNet (ConvNeXt backbone) small model.
  
|Baseline Model|Type|Dice coefficient|Parameters|
|---|---|---|---|
|[nvidia/segformer-b0-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)|SegFormer|0.8813|3.7M|
|[nvidia/segformer-b4-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b4-finetuned-ade-512-512)|SegFormer|0.9196|64M|
|[nvidia/segformer-b5-finetuned-ade-640-640](https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640)|SegFormer|0.9297|84M|
|[openmmlab/upernet-convnext-small](https://huggingface.co/openmmlab/upernet-convnext-small)|UperNet (ConvNeXt backbone)|0.9459|82M|
|[openmmlab/upernet-convnext-base](https://huggingface.co/openmmlab/upernet-convnext-base)|UperNet (ConvNeXt backbone)|0.9470|122M|
|[openmmlab/upernet-convnext-large](https://huggingface.co/openmmlab/upernet-convnext-large)|UperNet (ConvNeXt backbone)|**0.9494**|234M|
|[Intel/dpt-large](https://huggingface.co/Intel/dpt-large)|DPT|0.9420|343M|

## Displaying Inference Examples
The images below show inference examples of UperNet (ConvNeXt backbone) large, which is the best-performing model.

![ms_image_v1](https://github.com/JaeL17/Medical-Image-Segmentation/assets/73643391/5541e4f2-bbf6-451e-96b7-f7174a8f6423)
![ms_image_v2](https://github.com/JaeL17/Medical-Image-Segmentation/assets/73643391/dd633922-4a31-4eb0-a0bb-41f43267c6b0)
