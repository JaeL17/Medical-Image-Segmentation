# Medical-Image-Segmentation
# Overview
Welcome to the repository for Medical Image Segmentation, a crucial task in the field of computer vision. This project introduces the use of transformers based computer vision models, such as Segformers, DPT 
a novel apporach to train high-level semantic sentence embedding models, with a primary focus on enhancing the understanding of the intention within sentences.

## Motivation
In intent classification, it is essential to grasp a broader understanding of the underlying meaning in a sentence. However, conventional embedding models tend to focus on specific entity attributes within a sentence, such as certain keywords. To address this limitation, I implemented a data augmentation method to construct a triplet training dataset using the entity attributes and entity relations information extracted from FewRel relation classification dataset. You can access the FewRel dataset here: [FewRel dataset](https://paperswithcode.com/dataset/fewrel)

## Contents
1. **Data Parsing**: Code for constructing the triplet training dataset.
2. **Training**: Code for fine-tuning open-source sentence embedding models on the triplet training dataset.
3. **Testing**: Code for computing the top-3 accuracy of **MTEB/mtop_intent** dataset using a retrieval task approach based on Annoy index. In this approach, the trainset of MTEB/mtop_intent is indexed, and the testset is used as a search query.
4. **Visualising Attention**: Code for visualising the attention mechanisms within each layer of the embedding models.

## Running the code
1. **Data Parsing**
```
cd data
python data_parser.py
```

2. **Training**
   
- For single GPU
```
nohup python -u train.py \
   --base_model "sentence-transformers/all-MiniLM-L12-v2" \
   --triplet_margin 0.5 \
   --train_batch_size 32 \
   --output_model_name trained_model >> logs/train_log.txt &

tail -f logs/train_log.txt
```

* For multi-GPUs
```
CUDA_VISIBLE_DEVICES=0,1,2 nohup python -u train_multi_gpus.py \
   --base_model "sentence-transformers/all-MiniLM-L12-v2" \
   --triplet_margin 0.5 \
   --train_batch_size 128 \
   --output_model_name trained_model >> logs/train_multi_gpus_log.txt &

tail -f logs/train_multi_gpus_log.txt
```

3. **Testing**
```
nohup python -u test.py \
   --test_model_path "trained_model" \
   --annoy_tree_num 20 >> logs/test_log.txt &

tail -f logs/test_log.txt
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
