# Multimodal Large Language Models Training on LUMI
## Introduction
Large Language Models (LLMs) have transformed the field of artificial intelligence by enabling machines to "understand" and "generate" human language with remarkable performance. Models such as GPT, Llama, Qwen, and Mistral have demonstrated strong capabilities in tasks including text generation, question answering, summarization, translation, and code generation.

Despite their success, traditional LLMs are limited to textual information. Humans, however, interact with the world through multiple modalities, including vision, audio, and language. Many real-world tasks require reasoning across these different forms of information. For example:

- Understanding an image and answering questions about it.
- Describing a video in natural language.
- Interpreting speech and generating textual responses.
- Combining visual and textual information to perform complex reasoning.

These limitations have motivated the development of Multimodal Large Language Models (MLLMs).

### Multimodal Large Language Models
Multimodal Large Language Models extend traditional LLMs by incorporating additional input, such as images, audio, video or other type of data, beyond text. The model processes these inputs and produces outputs typically in textual form, although some advanced systems can also generate images, audio, or videos.

Compared with text-only LLMs, multimodal models introduce several additional components: vision encoders for processing images, or audio encoders for speech and sound. Modality projection layers that align different modalities into a shared embedding space. Cross-modal reasoning mechanisms that enable information fusion across modalities. As a result, MLLMs generally require larger datasets, more complex training pipelines, significantly higher computational resources, efficient distributed training strategies for large-scale training. Several multimodal models have become widely adopted in both academia and industry such as Qwen-VL series, LLaVA , InternVL and etc.

### Purpose of This Blog
Training modern multimodal models is computationally expensive, especially when scaling across multiple compute nodes. Understanding scaling performance is critical for maximizing the utilization of large supercomputing systems. In this blog, we investigate the scaling performance of several popular multimodal models on the LUMI supercomputer. We evaluate different distributed training frameworks and analyze scaling behavior across varying cluster sizes. The results presented here should be viewed as practical observations and rough performance estimates rather than exhaustive benchmarking studies.
## Data

In our experiments, we focus on a synthetic image–text dataset generated following the methodology described in the repository available at [here](https://github.com/shanshanwangcsc/synth-data-bench-training/blob/main/README_LUMI.md).

Using this workflow, we generated a total of 1 million image–text training samples. The synthetic dataset provides a controlled environment for benchmarking training throughput and scaling efficiency while avoiding the variability often present in real-world multimodal datasets.

### Data Format and Data Loader

Efficient data storage and loading are critical for large-scale multimodal training. As the number of GPUs increases, the data pipeline can easily become a bottleneck if storage access and preprocessing are not carefully optimized.

For our experiments, the synthetic image–text dataset is stored in the WebDataset format. WebDataset packages samples into multiple tar shards, where each sample consists of an image file and its corresponding metadata stored as a JSON file. This format is widely used for large-scale foundation model training because it enables efficient sequential I/O and minimizes filesystem metadata overhead.

The dataset structure is organized as follows:
```
data/
├── shard-000000.tar
├── shard-000001.tar
├── ...
└── shard-000999.tar
```
The dataset contains 1,000 tar shards in total. Each shard contains 1,000 training samples, organized as image–JSON pairs:
```
sample_00000000.jpg
sample_00000000.json
...
sample_00000999.jpg
sample_00000999.json
```
Each .jpg and .json pair forms a single multimodal training sample. 

The WebDataset representation enables purely sequential I/O pipelines, which are particularly beneficial for large-scale deep learning workloads. Compared to random-access file loading, sequential reads can achieve significantly higher throughput on both local storage and distributed filesystems. This approach is especially important when training on hundreds of GPUs, where even small I/O inefficiencies can lead to substantial reductions in overall utilization.

For data loading, we use the Energon dataloader, which provides efficient support for distributed foundation model training. Energon integrates seamlessly with WebDataset and offers automatic sharding, worker balancing, asynchronous prefetching, and scalable data streaming across multiple nodes. These features help ensure that the GPUs remain fully utilized throughout training and that the input pipeline scales together with the compute resources.

### Example Training Sample

A typical training sample consists of an image and a corresponding conversational annotation:
```
sample_00000000.jpg
sample_00000000.json
```
The associated JSON file contains a unique identifier and a conversation pair representing the visual question-answering task:
```
{
  "id": "6916c960-824b-4275-80a1-4874d5ed17e2",
  "conversations": [
    {
      "from": "human",
      "value": "What is the color of the dog in the picture?"
    },
    {
      "from": "gpt",
      "value": "The dog is white."
    }
  ]
}
```

During training, the image is processed by the vision encoder while the conversation is tokenized and fed into the language model. The model learns to associate visual features extracted from the image with the corresponding textual response, enabling multimodal reasoning and visual question answering.


## Models
We explored several models from the Qwen ecosystem because of their strong performance and widespread adoption in the research community. The evaluated models include Qwen3-VL-2B-Instruct, Qwen3-VL-8B-Instruct, Qwen3.5-2B and Qwen3.5-9B. 

Qwen3-VL series adopts a modular fusion approach, where a separate vision encoder is appended to a text-only language backbone. This requires dedicated projection layers to map visual features into the text embedding space. Instead of bolting vision adapters onto a text base like Qwen3-VL, Qwen3.5 is designed from the ground up for "native multimodality", unifies text and vision into a single native architecture.


The selected model sizes also allow us to observe how scaling behavior changes as model complexity increases.

## Training 
### High-Level Multimodal Training Workflow 

At a high level, training a Qwen3-VL model follows a pipeline that combines visual feature extraction, multimodal alignment, and language modeling. The goal is to teach the model to understand visual information together with natural language instructions and generate appropriate textual responses.

The workflow consists of the following stages:

**1. Multimodal Input Preparation**

Each training sample contains an image together with a text-based instruction or conversation as shown in the data section. The image and text are processed separately before being combined.

**2. Visual Feature Extraction**

The input image is first processed by the vision encoder, typically based on architectures such as Vision Transformer (ViT) or SigLIP. The vision encoder divides the image into patches and converts them into a sequence of visual embeddings:

```
Image
  │
  ▼
Vision Encoder
  │
  ▼
Visual Tokens
```

These visual tokens represent high-level information extracted from the image, such as objects, shapes, textures, and spatial relationships.

**3. Vision-Language Alignment**

The visual embeddings produced by the vision encoder usually exist in a different feature space from the language model. Therefore, Qwen-VL introduces a multimodal projection or alignment module to transform visual features into the same embedding space used by the language model.

```
Visual Tokens
      │
      ▼
Projection Layer
      │
      ▼
Aligned Visual Embeddings
```

After this step, visual tokens can be treated similarly to language tokens and processed together by the large language model.

**4. Multimodal Sequence Construction**

The visual embeddings are inserted together with the tokenized text prompt to form a unified multimodal sequence.

Conceptually:

```
[Image Tokens] + [User Instruction Tokens] + [Assistant Response Tokens]
```

For example:

```
<Vision Tokens>
What is the color of the dog?
The dog is white.
```

The language model receives this combined sequence as input and learns the relationship between visual information and language.

**5. Autoregressive Language Modeling**

The Qwen language model backbone processes the multimodal sequence and predicts the assistant response token-by-token. During training, the model is optimized using next-token prediction:

```
Input:
<Image> What is the color of the dog?

Target:
The dog is white.
```

The model predicts:

```
The → dog → is → white → .
```

The difference between predicted tokens and ground-truth tokens is measured using cross-entropy loss.

**6. Loss Calculation and Backpropagation**

Typically, in supervised instruction tuning, the training loss is computed only on the target assistant response tokens. User instructions and image tokens provide context but are not directly optimized.

The gradients are then propagated backward through:

```
Language Model
      ↑
Projection Layer
      ↑
Vision Encoder
```

Depending on the training stage, different components may be frozen or updated. For large-scale multimodal instruction tuning, it is common to train the projection layer and language model while optionally fine-tuning the vision encoder.

### Training Framework

We evaluated several distributed training strategies for large-scale deep learning.

#### PyTorch Distributed Data Parallel (DDP)

DDP replicates the full model on each GPU and synchronizes gradients after every training step. It is easy to use and provides strong performance for moderate-sized models. However, it becomes impractical when memory limits are exceeded.

**Usage:** We use DDP for smaller models, including Qwen3-VL-2B-Instruct and Qwen3.5-2B.

#### Fully Sharded Data Parallel (FSDP)

FSDP shards model parameters, gradients, and optimizer states across GPUs, reducing memory consumption and enabling larger model training.

**Advantages:**
- Lower memory footprint.
- Supports larger models.
- Improves scalability for memory-intensive workloads.

**Limitations:**
- Higher communication overhead.
- More complex configuration.

**Usage:** We use FSDP for larger models, including Qwen3-VL-8B-Instruct and Qwen3.5-9B.


## Results and Analysis
The experiments were conducted on the LUMI supercomputer, specifically the LUMI-G GPU partition. Each LUMI-G compute node is equipped with:

- 1 × 64-core AMD EPYC 7A53 "Trento" CPU
- 4 × AMD Instinct MI250X GPU accelerators
- Each MI250X accelerator contains 2 Graphics Compute Dies (GCDs)
- Each GCD provides 64 GB of HBM2e memory

Therefore, each node provides:

- 8 GCDs (8 GPU devices from the ROCm/Slurm perspective)
- 512 GB total HBM2e memory (8 × 64 GB)
- High-bandwidth GPU-to-GPU communication through AMD Infinity Fabric
- HPE Slingshot-11 high-speed interconnect for multi-node communication

We evaluated three cluster configurations:

- **1 node:** 8 GPU devices (8 GCDs), 512 GB HBM2e memory
- **16 nodes:** 128 GPU devices (128 GCDs), 8 TB HBM2e memory
- **32 nodes:** 256 GPU devices (256 GCDs), 16 TB HBM2e memory

In this experiment, we measure token throughput (tokens per second per GPU) and compute utilization (TFLOPS per GPU) as our metrics for throughput. We also track weak scaling efficiency—the red line in the top charts—to evaluate how effectively these models leverage additional hardware when scaling from 8 to 256 GPUs on the LUMI supercomputer. The detailed results are presented in the figures listed below for the different models.

**Qwen3VL-2b**
![Qwen3VL-2b](Qwen_Qwen3-VL-2B-Instruct.png)
**Qwen3VL-8b**
![Qwen3VL-8b](Qwen_Qwen3-VL-8B-Instruct.png)
**Qwen3.5-2b**
![Qwen3.5-2b](Qwen_Qwen3.5-2B.png)
**Qwen3.5-9b**
![Qwen3.5-9b](Qwen_Qwen3.5-9B.png)

Grouping the 2B models together (Qwen3-VL-2B and Qwen3.5-2B, both trained at sequence length 8192), we observe that both scale down to roughly 88–89% efficiency at 256 GPUs (VL-2B: 3,414 → 3,026 tokens/s/GPU; 3.5-2B: 2,632 → 2,316 tokens/s/GPU). This is the classic signature of a compute-to-communication imbalance in small models: the per-GPU FLOP count is modest, so the roughly constant cost of synchronizing gradients over the Slingshot fabric consumes a growing share of each step. Notably, Qwen3-VL-2B still posts the highest single-node numbers in this experiment (63.7 TFLOPS/GPU), confirming that the degradation is a scaling artifact rather than a compute deficiency.

The two mid-size models form the most interesting contrast, since they are run under matched conditions (sequence length 4096, FSDP). Qwen3.5-9B scales nearly flat: 1,443 → 1,414 tokens/s/GPU, i.e. 98% efficiency at 256 GPUs, with per-GPU TFLOPS essentially unchanged (41.9 → 41.0). Qwen3-VL-8B, by contrast, falls from 1,637 to 1,184 tokens/s/GPU (72% efficiency), and its achieved TFLOPS drops from 49.5 to 35.8 — meaning that at scale its GPUs spend an increasing fraction of every step not doing useful compute.
Because parameter count, sequence length, and parallelization strategy are matched between these two runs, the divergence cannot be attributed to FSDP gradient-sync volume — the vision encoder and projector contribute only a few percent of the synchronized parameters. The more plausible mechanism is rank-level load imbalance and pipeline synchronization inherent to the modular vision path: Qwen3-VL's dynamic-resolution ViT produces a variable number of visual tokens per sample, so per-microbatch compute differs across ranks and the slowest rank sets the step time; as the rank count grows, the worst-case imbalance grows with it. The ViT forward also acts as a separate, unsharded stage that must complete before the language backbone proceeds, exposing the step to inter-node latency. Qwen3.5's native multimodal design presents the trainer with a homogeneous token stream, keeping per-rank compute balanced and free of such barriers — consistent with its near-ideal weak scaling. Disentangling the relative weight of these effects would require per-step profiling of NCCL communication versus idle time.

Maybe need to remove some text above.

The implementation code is available in the [repo](https://github.com/shanshanwangcsc/vlm-training/blob/main/README_LUMI.md).
## Acknowledgement 
We would like to thank ....


