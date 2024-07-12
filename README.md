# TF-ID
This repository contains the full training code to reproduce all TF-ID models. We also open-source the model weights and human annotated dataset all under mit license.

## Model Summary
![TF-ID](https://github.com/ai8hyf/TF-ID/blob/main/assets/cover.png)

TF-ID (Table/Figure IDentifier) is a family of object detection models finetuned to extract tables and figures in academic papers created by [Yifei Hu](https://x.com/hu_yifei). They come in four versions:
| Model   | Model size | Model Description | 
| ------- | ------------- |   ------------- |  
| TF-ID-base[[HF]](https://huggingface.co/yifeihu/TF-ID-base) | 0.23B  | Extract tables/figures and their caption text  
| TF-ID-large[[HF]](https://huggingface.co/yifeihu/TF-ID-large) (Recommended) | 0.77B  | Extract tables/figures and their caption text  
| TF-ID-base-no-caption[[HF]](https://huggingface.co/yifeihu/TF-ID-base-no-caption) | 0.23B  | Extract tables/figures without caption text
| TF-ID-large-no-caption[[HF]](https://huggingface.co/yifeihu/TF-ID-large-no-caption) (Recommended) | 0.77B  | Extract tables/figures without caption text

All TF-ID models are finetuned from [microsoft/Florence-2](https://huggingface.co/microsoft/Florence-2-large-ft) checkpoints.

## Sample Usage
- Use `python inference.py` to extract bounding boxes from one given image
- Use `python pdf_to_table_figures.py`to extract all tables and figures from one pdf paper and save the cropped figures and tables under `./sample_output`
- **TF-ID-large** are used in the scripts by default. You can swtich to a different variant by changing the model_id in the scripts, but large models are always recommended.

## Train TF-ID models from scratch
1. Clone the repo: `git clone https://github.com/ai8hyf/TF-ID`
2. `cd TF-ID`
3. Download the [huggingface.co/datasets/yifeihu/TF-ID-arxiv-papers](https://huggingface.co/datasets/yifeihu/TF-ID-arxiv-papers) from Hugging Face
4. Move **annotations_with_caption.json** to `./annotations` (Use **annotations_no_caption.json** if you don't want the bounding boxes to include text captions)
5. Unzip the **arxiv_paper_images.zip** and move the .png images to `./images`
6. Convert the coco format dataset to florence 2 format: `python coco_to_florence.py`
7. You should see **train.jsonl** and **test.jsonl** under `./annotations`
8. Train the model with Accelerate: `accelerate launch train.py`
9. The checkpoints will be saved under `./model_checkpoints`

## Hardware Requirement
With [microsoft/Florence-2-large-ft](https://huggingface.co/microsoft/Florence-2-large-ft), `BATCH_SIZE=4` will require at least 40GB VRAM on a single GPU. The [microsoft/Florence-2-base-ft](https://huggingface.co/microsoft/Florence-2-base-ft) model takes much less VRAM. Please modify the `BATCH_SIZE` and `CHECKPOINT` parameter in the `train.py` before you start training.

## Benchmarks
We tested the models on paper pages outside the training dataset. The papers are a subset of huggingface daily paper.
Correct output - the model draws correct bounding boxes for every table/figure in the given page.

| Model                                                         | Total Images | Correct Output | Success Rate |
|---------------------------------------------------------------|--------------|----------------|--------------|
| TF-ID-base[[HF]](https://huggingface.co/yifeihu/TF-ID-base)   | 258          | 251            | 97.29%       |
| TF-ID-large[[HF]](https://huggingface.co/yifeihu/TF-ID-large) | 258          | 253            | 98.06%       |
| TF-ID-base-no-caption[[HF]](https://huggingface.co/yifeihu/TF-ID-base-no-caption)   | 261          | 253            | 96.93%       |
| TF-ID-large-no-caption[[HF]](https://huggingface.co/yifeihu/TF-ID-large-no-caption) | 261          | 254            | 97.32%       |

Depending on the use cases, some "incorrect" output could be totally usable. For example, the model draw two bounding boxes for one figure with two child components.

## Acknowledgement
- I learned how to work with Florence 2 models from this [Roboflow's awesome tutorial](https://blog.roboflow.com/fine-tune-florence-2-object-detection/).
- My friend Yi Zhang helped annotate some data to train our proof-of-concept models including a yolo-based TF-ID model.

## Citation
If you find TD-ID useful, please cite this project as:
```
@misc{TF-ID,
  author = {Yifei Hu},
  title = {TF-ID: Table/Figure IDentifier for academic papers},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ai8hyf/TF-ID}},
}
```