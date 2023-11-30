# Medical-Image-Segmentation
# Overview
Welcome to the repository for Medical Image Segmentation, a crucial task in the field of computer vision. Since 2021, there have been significant advancements in image segmentation, with transformer based architectures or hybrid architectures (combining transformer blocks and convolutional blocks) emerging as State-of-the-Art models. This project focuses on implementing transfer learning on pre-trained transformers based computer vision models such as Segformers, DPT, BEiT, and UPerNet to achieve advanced semantic segmentation performance in medical imaging. 

## Motivation
In 2019, an estimated 5 million people were diagnosed with a cancer of the gastro-intestinal tract worldwide. Of these patients, about half are eligible for radiation therapy, usually delivered over 10-15 minutes a day for 1-6 weeks. Radiation oncologists try to deliver high doses of radiation using X-ray beams pointed to tumors while avoiding the stomach and intestines. With newer technology such as integrated magnetic resonance imaging and linear accelerator systems, also known as MR-Linacs, oncologists are able to visualize the daily position of the tumor and intestines, which can vary day to day. In these scans, radiation oncologists must manually outline the position of the stomach and intestines in order to adjust the direction of the x-ray beams to increase the dose delivery to the tumor and avoid the stomach and intestines. This is a time-consuming and labor intensive process that can prolong treatments from 15 minutes a day to an hour a day, which can be difficult for patients to tolerate—unless deep learning could help automate the segmentation process. A method to segment the stomach and intestines would make treatments much faster and would allow more patients to get more effective treatment. You can access the dataset here: [dataset](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/overview)

## Contents
1. **utils.py**: Code for loading training and validation dataset, as well as pre-processing the data.
2. **trainer.py**: Code for transfer learning open-source models from Hugging Face on the training dataset. This section also includes computing Dice coefficients on validation dataset.
3. **visualisation.ipynb**: Code for displaying sample images and inference resylts of a segmentation model.

## Running the trainer code
1. **Training**
   
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
    --lr 2e-4 1>> logs/train_multi_gpus_log.txt &
```

* Logging
```
tail -f logs/train_multi_gpus_log.txt
```

## Test Results and Performance Comparison
Despite its significantly smaller parameter size, the high-level semantic model outperforms the base model and other two open-source sentence embedding models, e5-large-v2 (Microsoft) and ember-v1 (current SOTA model for classification task on [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard)).

|Model|Hidden size|Parameters|
|---|---|---|
|[base model](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)|384|33M|
|high-level semantic model|384|33M|
|[e5-Large-v2](https://huggingface.co/embaas/sentence-transformers-e5-large-v2)|1024|335M|
|[ember-v1](https://huggingface.co/llmrails/ember-v1)|1024|335M|

|Model|Top-1|Top-2|Top-3|
|---|---|---|---|
|[base model](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)|0.851|0.918|0.943|
|high-level semantic model|**0.930**|**0.954**|**0.963**|
|[e5-Large-v2](https://huggingface.co/embaas/sentence-transformers-e5-large-v2)|0.888|0.940|0.956|
|[ember-v1](https://huggingface.co/llmrails/ember-v1)|0.902|0.945|**0.963**|

## Attention Visualisation

- **Base model**

The image below illustrates that the base model focuses on specific keywords, such as **"pin"** and **"card"**.
   
![base_head_view](https://github.com/JaeL17/high-level-semantics-embedding-model/assets/73643391/143ac834-ad9a-43d7-b0c2-d3bd15446279)


* **High-level sematic model**

In contrast, the high-level semantic model, as shown in the image below, unfolds the content of a sentence by focusing on important predicates like **"forgot"**, **"have"**, **"locked"**, and **"using"**, rather than focusing on entity attributes (keywords) within a sentence.

![high_level_semantics](https://github.com/JaeL17/high-level-semantics-embedding-model/assets/73643391/5e2cb81c-cbf0-4cc6-94db-fb82562964e7)
