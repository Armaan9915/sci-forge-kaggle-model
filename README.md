# sci-forge-kaggle-model
# 🔍 DINOv2 Copy-Move Forgery Detection

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![DINOv2](https://img.shields.io/badge/Backbone-DINOv2-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Research_Prototype-yellow?style=for-the-badge)

A state-of-the-art approach to **Image Copy-Move Forgery Detection** (CMFD) utilizing Self-Supervised Vision Transformers (DINOv2) and Pixel-wise Contrastive Learning. 

This project moves beyond traditional semantic segmentation by learning a **metric space** where cloned regions cluster together and authentic backgrounds remain distinct, allowing for the detection of arbitrary, unseen manipulation patterns.

---

## 🚀 Key Technical Highlights

For the technical recruiter or hiring manager, here is what makes this project stand out:

*   **Foundation Model Integration:** Leverages **DINOv2 (ViT-B/14)** frozen embeddings, exploiting its robust understanding of semantic layout and texture.
*   **Metric Learning Approach:** Instead of binary classification, this model utilizes **Pixel-wise Supervised Contrastive Loss (SupCon)**. It pulls embeddings of the same instance together and pushes different instances apart.
*   **Custom Loss Landscape:** Implements a sophisticated weighted loss function combining:
    *   `PixelSupConLoss`: For clustering capability.
    *   `UniformityLoss`: To minimize intra-cluster variance.
    *   `InterInstanceSeparationLoss`: To maximize the distance between distinct clones.
*   **Unsupervised Inference Pipeline:** Uses **HDBSCAN** clustering + Cosine Similarity geometry checks during inference, making the system robust to object shapes it has never seen during training.
*   **Distributed Training:** Built with `torch.distributed` (DDP) for multi-GPU scalability.

---

## 🧠 System Architecture

### 1. The Encoder (Trainable)
The model takes an input image ($462 \times 462$) and processes it through a pipeline:
1.  **Backbone:** DINOv2 (Frozen) extracts dense features.
2.  **Projection Head:** A 3-layer MLP ($768 \to 1024 \to 512$) projects features into a normalized embedding space.
3.  **Output:** A $33 \times 33$ grid of 512-dimensional unit vectors.

### 2. The Inference Engine (Clustering)
Unlike standard segmentation networks (like U-Net) that output a fixed class map, this system outputs embeddings. To generate masks, we perform a **Cascade Detection**:
1.  **Strict HDBSCAN:** Looks for very dense, obvious clusters.
2.  **Anomaly Detection:** If no clusters are found, it calculates a global "suspicion score" based on embedding outliers.
3.  **Region Growing:** If a core forgery is found, a region-growing algorithm expands the mask to capture edges based on cosine similarity.

---

## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| `model.py` | Contains the **DINOv2 + MLP** architecture. |
| `losses.py` | Custom implementations of **SupCon**, **Uniformity**, and **Separation** losses. |
| `clustering.py` | The complex inference logic: HDBSCAN wrapper, anomaly scoring, and region growing. |
| `train.py` | Distributed Data Parallel (DDP) training loop with Mixed Precision (AMP). |
| `dataset.py` | Custom Dataset class with Albumentations (Blur, JPEG, Noise, Rotation). |
| `config.py` | Centralized configuration for hyperparameters, paths, and compute settings. |
| `tuning.py` | Automated hyperparameter search to optimize HDBSCAN parameters ($MinPts$, $\epsilon$). |

---

## 📊 Performance Metric: Object-centric F1 (oF1)

This project evaluates performance using **oF1**, a metric designed for instance-level detection.
*   Standard pixel-F1 is biased by background pixels.
*   **oF1** uses the **Hungarian Algorithm** to match predicted forgery instances to ground truth instances, ensuring the model accurately separates distinct cloned regions.

---

## 🛠️ Installation & Usage

### Prerequisites
*   Python 3.8+
*   PyTorch 2.0+ (CUDA support recommended)
*   `pip install -r requirements.txt` (Assumed dependencies: torch, numpy, opencv-python, albumentations, hdbscan, scikit-learn, tqdm)

### Training
The training script utilizes Distributed Data Parallel (DDP). To run on a single node with 2 GPUs:

python train.py
Modify config.py to point to your dataset paths before running.
Hyperparameter Tuning
To automatically find the best clustering parameters for the trained model:
code
Bash
python tuning.py
Inference / Testing
To evaluate the model on the test set and generate metrics:
code
Bash
python test.py
🔬 Methodology Visualization
<img width="2200" height="1618" alt="image" src="https://github.com/user-attachments/assets/628e18e1-b017-4605-9386-953fb6cc784e" />

