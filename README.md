# Official Implementation of "From Pixels to Graphs: Open-Vocabulary Scene Graph Generation with Vision-Language Models"

## Introductionp
Our paper ["Relation-aware Hierarchical Prompt for Open-vocabulary Scene Graph Generation"](https://arxiv.org/abs/2412.19021) AAAI 2025.


## Installation and Setup

```
pip install -r requirements.txt

# follow this page https://stackoverflow.com/questions/72988735/replacing-thc-thc-h-module-to-aten-aten-h-module
export TORCH_CUDA_ARCH_LIST="8.6"
python setup.py build develop --user

pip uninstall -y opencv opencv-python opencv-contrib-python \
  opencv-python-headless opencv-contrib-python-headless
rm -rf /usr/local/lib/python3.10/dist-packages/cv2 \
       /usr/local/lib/python3.10/dist-packages/opencv*dist-info 2>/dev/null || true
       
pip install --no-deps --no-cache-dir opencv-python-headless==4.10.0.84
pip install "git+https://github.com/openai/CLIP.git"
pip install google-generativeai
```

***Pre-trained Visual-Semantic Space.*** Download the pre-trained `GLIP-T` and `GLIP-L` [checkpoints](https://github.com/microsoft/GLIP#model-zoo) into the ``MODEL`` folder. 
(!! GLIP has updated the downloading paths, please find these checkpoints following https://github.com/microsoft/GLIP#model-zoo)
```
mkdir MODEL
wget https://huggingface.co/GLIPModel/GLIP/resolve/main/glip_tiny_model_o365_goldg_cc_sbu.pth -O MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth
wget https://huggingface.co/GLIPModel/GLIP/blob/main/glip_large_model.pth -O MODEL/glip_large_model.pth

# Download VLM (Qwen2.5-14B-Instruct-AWQ)
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-14B-Instruct-AWQ
```

## Dataset Preparation

### Docker Container Setup

This repository can be run in a Docker container with proper volume mounts to access datasets. The following setup ensures all symbolic links work correctly inside the container.

#### **Container Creation**

Create and run the container with the following command:

```bash
docker run -it --name avas-sgg --gpus all --runtime=nvidia \
  -v /home/quang/sensys/new/RAHP/:/RAHP \
  -v /ssd2/datasets/:/ssd2/datasets \
  -v /ssd0/datasets:/ssd0/datasets \
  -e DATASET=/RAHP/DATASET \
  -w /RAHP \
  --network host \
  -u 0:0 \
  nvcr.io/nvidia/pytorch:24.10-py3
```

**Volume Mounts Explained:**
- `/home/quang/sensys/new/RAHP/:/RAHP` - Mounts the codebase
- `/ssd2/datasets/:/ssd2/datasets` - Mounts VG150 dataset (preserves absolute paths)
- `/ssd0/datasets:/ssd0/datasets` - Mounts COCO dataset (preserves absolute paths)
- `-e DATASET=/RAHP/DATASET` - Sets environment variable for path resolution
- `-w /RAHP` - Sets working directory inside container

#### **Verification Inside Container**

After starting the container, verify the setup:

```bash
# Check environment variable
echo $DATASET

# Verify symbolic links are working
ls -la /RAHP/DATASET/coco/
ls -la /RAHP/DATASET/VG150/

# Test path resolution
python3 -c "
from maskrcnn_benchmark.config.paths_catalog import try_to_find
print('VG150 path:', try_to_find('VG150/VG_100K'))
print('COCO path:', try_to_find('coco/train2017'))
"
```

### Symbolic Links Setup

This repository uses symbolic links to avoid duplicating large dataset files. The following symbolic links are expected to be set up:

1. **Visual Genome Dataset**:
   ```bash
   # Link VG_100K images to the actual dataset location
   ln -s /ssd2/datasets/VG150/VG_100K /home/quang/sensys/new/RAHP/DATASET/VG150/VG_100K
   ```

2. **COCO Dataset**:
   ```bash
   # Link COCO train2017 images and annotations
   ln -s /ssd0/datasets/COCO/rahp/train2017 /home/quang/sensys/new/RAHP/DATASET/coco/train2017
   ln -s /ssd0/datasets/COCO/rahp/annotations /home/quang/sensys/new/RAHP/DATASET/coco/annotations
   ```

**Important Notes:**
- The symbolic links use **absolute paths** (`/ssd2/datasets/...`, `/ssd0/datasets/...`)
- When using Docker, mount the dataset directories with the **same absolute paths** to preserve link functionality
- Adjust the source paths according to your actual dataset locations
- The `DATASET` environment variable helps the path resolution system find datasets correctly

#### **Troubleshooting Container Issues**

If you encounter issues with symbolic links in the container:

1. **Check if symbolic links are broken:**
   ```bash
   ls -la /RAHP/DATASET/coco/
   ls -la /RAHP/DATASET/VG150/
   ```
   Look for red text or `@` symbols indicating broken links.

2. **Verify volume mounts:**
   ```bash
   ls -la /ssd2/datasets/VG150/VG_100K/ | head -5
   ls -la /ssd0/datasets/COCO/rahp/annotations/ | head -5
   ```

3. **Test path resolution:**
   ```bash
   python3 -c "
   import os
   print('DATASET env var:', os.environ.get('DATASET'))
   print('Current dir:', os.getcwd())
   "
   ```

4. **Recreate symbolic links if needed:**
   ```bash
   # Remove broken links
   rm /RAHP/DATASET/coco/annotations /RAHP/DATASET/coco/train2017
   rm /RAHP/DATASET/VG150/VG_100K
   
   # Recreate them
   ln -s /ssd0/datasets/COCO/rahp/annotations /RAHP/DATASET/coco/annotations
   ln -s /ssd0/datasets/COCO/rahp/train2017 /RAHP/DATASET/coco/train2017
   ln -s /ssd2/datasets/VG150/VG_100K /RAHP/DATASET/VG150/VG_100K
   ```

### Dataset Downloads

1. Visual Genome
* ``Visual Genome (VG)``: Download the following files into ``DATASET/VG150`` folder:
  - **VG_100K images**: Download from [Visual Genome API](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html)
  - **image_data.json**: Download from [image_data.json.zip](https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip)
  - **region_descriptions.json**: Download from [region_descriptions.json.zip](https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/region_descriptions.json.zip)
  - **relationship_alias.txt**: Download from [relationship_alias.txt](https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationship_alias.txt)
  - **VG-SGG-with-attri.h5**: Download from [Hugging Face](https://huggingface.co/datasets/kb-kim/LLM4SGG/resolve/main/VG-SGG-with-attri.h5)
  - **VG-SGG-dicts-with-attri.json**: Download from [GitHub](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/datasets/vg/VG-SGG-dicts-with-attri.json)

2. Openimage V6
* ``Openimage V6``: 
    1. The initial dataset(oidv6/v4-train/test/validation-annotations-vrd.csv) can be downloaded from [offical website]( https://storage.googleapis.com/openimages/web/download.html).

    2. The Openimage is a very large dataset, however, most of images doesn't have relationship annotations. 
To this end, we filter those non-relationship annotations and obtain the subset of dataset ([.ipynb for processing](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EebESIOrpR5NrOYgQXU5PREBPR9EAxcVmgzsTDiWA1BQ8w?e=46iDwn) ). 

    3. You can download the processed dataset: [Openimage V6(38GB)](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EXdZWvR_vrpNmQVvubG7vhABbdmeKKzX6PJFlIdrCS80vw?e=uQREX3)
    4. By unzip the downloaded datasets, the dataset dir contains the `images` and `annotations` folder. 
    Link the `open-imagev6` dir to the `./cache/openimages` then you are ready to go.
    ```bash
    mkdir datasets/openimages
    ln -s /path/to/open_imagev6 datasets/openimages ./cache/cache
    ```


The `DATASET` directory is organized roughly as follows:
```
├─Openimage V6
│  ├─annotations
│  ├─images
└─VG150
    ├─VG_100K
    ├─weak_supervisions
    ├─image_data.json
    ├─VG-SGG-dicts-with-attri.json
    ├─region_descriptions.json
    └─VG-SGG-with-attri.h5 
```

Since GLIP pre-training has seen part of VG150 test images, we remove these images and get new VG150 split and write it to `VG-SGG-with-attri.h5`. 
Please refer to [tools/cleaned_split_GLIPunseen.ipynb](tools/cleaned_split_GLIPunseen.ipynb).

### **Relation-Aware & Entity-Aware Prompt Generation Guide**

Follow these steps to create high-quality prompts driven by dataset-specific predicate superclasses.

#### **Step 1: Cluster predicates into superclasses (dataset-specific)**
```bash
cd tools
python cluster_entity_2_super_class.py
```
- Goal: Cluster the predicates in your chosen dataset into groups.
- Expectation: The number of clusters ideally equals the number of predicate superclasses you want to use later.

#### **Step 2: Generate superclass names and validate uniqueness**
```bash
python check_super_entity_class.py
```
- Purpose: For each cluster from Step 1, generate a concise superclass name and validate the count.
- Validation rule: On each round, if the generated superclass name already exists from a previous round, treat it as invalid and regenerate until all superclass names are unique (and the count matches the clusters).

#### **Step 3: Generate relation-aware prompts using the superclasses**
```bash
python relation_aware_prompt_generation.py
```
- This uses the validated superclass names from Step 2 to produce relation-aware prompts that incorporate both entity hierarchies and relation contexts.



## Training & Evaluation

1. Training
```
bash scripts/train.sh
```

2. Evaluation

```
bash scripts/test.sh
```

## Acknowledgement

This repo is based on [VS3](https://github.com/zyong812/VS3_CVPR23), [PGSG](https://github.com/SHTUPLUS/Pix2Grp_CVPR2024/tree/main), [GLIP](https://github.com/microsoft/GLIP), [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch), [SGG_from_NLS](https://github.com/YiwuZhong/SGG_from_NLS). Thanks for their contribution.