# Medical-Image-Segmentation
# Overview
Welcome to the repository for Medical Image Segmentation, a crucial task in the field of computer vision. Since 2021, there have been significant advancements in image segmentation, with transformer based architectures or hybrid architectures (combining transformer blocks and convolutional blocks) emerging as State-of-the-Art models. This project focuses on fine-tuning transformers based pre-trained computer vision models such as Segformers, DPT, BEiT, and UPerNet to achieve advanced semantic segmentation performance in medical imaging. 

## Motivation
In 2019, an estimated 5 million people were diagnosed with a cancer of the gastro-intestinal tract worldwide. Of these patients, about half are eligible for radiation therapy, usually delivered over 10-15 minutes a day for 1-6 weeks. Radiation oncologists try to deliver high doses of radiation using X-ray beams pointed to tumors while avoiding the stomach and intestines. With newer technology such as integrated magnetic resonance imaging and linear accelerator systems, also known as MR-Linacs, oncologists are able to visualize the daily position of the tumor and intestines, which can vary day to day. In these scans, radiation oncologists must manually outline the position of the stomach and intestines in order to adjust the direction of the x-ray beams to increase the dose delivery to the tumor and avoid the stomach and intestines. This is a time-consuming and labor intensive process that can prolong treatments from 15 minutes a day to an hour a day, which can be difficult for patients to tolerate—unless deep learning could help automate the segmentation process. A method to segment the stomach and intestines would make treatments much faster and would allow more patients to get more effective treatment. You can access the dataset here: [dataset](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/overview)

## Contents
1. **utils.py**: Contains code for loading training and validation dataset, as well as pre-processing the data and data augmentation on training dataset.
2. **model.py**: Exlore this file to find code for the segmentation model class used in this project. 
3. **trainer.py**: Code for transfer learning open-source models from Hugging Face on the training dataset. This section also includes computing Dice coefficients on validation dataset.
4. **visualisation.ipynb**: Explore this Jupyter Notebook for code related to displaying sample images and inference resylts of the segmentation model.

## Data Pre-processing and Training Data Augmentation
During the training phase, the we adopt multiple data augmentation strategies to enhance the generalisation capabilities of models. The key pre-processing and augmentation techniques include:
* Image resizing to 288x288 pixels.
* Random flipping (both horizontal and vertical).
* Random adjustments to brightness and contrast.
* Coarser dropout for regularisation.
* Random scaling, shifting, and rotation.

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
    --epochs 6 \
    --lr 2e-4 >> ./logs/train_upertnet_small.log &
```

* Logging
```
tail -f logs/train_upertnet_small.log
```

## Evaluation Results and Performance Comparison
In this project, we evaluate different model with varying parameter sizes to compare their performance in semantic segmentation tasks. While UperNet exhibits the best dice coefficient after transfer learning, it's interesting to note that increasing the parameters of UperNet doesn't yield a proportionally significant performance improvement compared to SegFormer

|Model|Type|Dice coefficient|Parameters|
|---|---|---|---|
|[nvidia/segformer-b0-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)|SegFormer|0.8813|3.7M|
|[nvidia/segformer-b4-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b4-finetuned-ade-512-512)|SegFormer|0.9196|64M|
|[nvidia/segformer-b5-finetuned-ade-640-640](https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640)|SegFormer|0.9297|84M|
|[openmmlab/upernet-convnext-small](https://huggingface.co/openmmlab/upernet-convnext-small)|UperNet (ConvNetXt backbone)|0.9459|82M|
|[openmmlab/upernet-convnext-base](https://huggingface.co/openmmlab/upernet-convnext-base)|UperNet (ConvNetXt backbone)|0.9470|122M|
|[openmmlab/upernet-convnext-large](https://huggingface.co/openmmlab/upernet-convnext-large)|UperNet (ConvNetXt backbone)|**0.9494**|234M|
|[Intel/dpt-large](https://huggingface.co/Intel/dpt-large)|DPT|0.9420|343M|

## Displaying Results
![ms_image_v1](https://github.com/JaeL17/Medical-Image-Segmentation/assets/73643391/5541e4f2-bbf6-451e-96b7-f7174a8f6423)
![ms_image_v2](https://github.com/JaeL17/Medical-Image-Segmentation/assets/73643391/dd633922-4a31-4eb0-a0bb-41f43267c6b0)
