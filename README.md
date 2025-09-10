# Tornado-detection-using-radar-images

This repository contains a deep learning pipeline for **automatic tornado detection** using radar data.  
The project leverages **PyTorch** and **PyTorch Lightning** to train convolutional neural networks (CNNs) that can detect rare tornado events, addressing the challenge of **imbalanced datasets**.  

The ultimate goal is to contribute to **early warning systems** by improving the ability to identify tornado signatures in radar imagery, thus reducing the risk of casualties and damage.  

---

## Background  

Tornadoes are among the most violent and unpredictable weather phenomena. Detecting them in real time is a major scientific and technological challenge:  
- Traditional algorithms rely on **hand-crafted radar features**, which may miss subtle patterns.  
- Tornado events are **rare** compared to non-tornadic weather, making supervised learning challenging.  
- False alarms reduce trust in warnings, while missed detections can be catastrophic.  

This project aims to explore **deep learning methods** to automatically detect tornado signatures directly from radar data, overcoming some limitations of traditional techniques.  

---

## Dataset & Inspiration  

This project was inspired by the **[TORNET project from MIT](https://github.com/valentius27/tornet)**, which provides a public framework for tornado detection using machine learning. My project aims to advance the model by taking a different approach that emphasizes the temporal dimension. 

### Dataset  
- Based on **radar observations** of severe weather events.  
- Contains multiple variables such as **reflectivity, radial velocity, spectrum width**, etc.  
- Highly **imbalanced dataset**: only a small fraction of samples correspond to tornado events.  
- Preprocessing includes channel-wise normalization and slicing radar volumes into manageable inputs.  

## Features  

- **Model Architecture**  
  - 3D Convolutional Neural Network baseline.  
  - Batch normalization and max pooling layers for stable and efficient training.  
  - Modular design: easy to add or replace layers.  

- **Preprocessing**  
  - Radar data normalization using channel-wise min–max values.  
  - Support for multiple input channels (reflectivity, velocity, spectrum width, etc.).  

- **Training & Evaluation**  
  - Implemented with **PyTorch Lightning** for cleaner code and reproducibility.  
  - Checkpointing: automatically saves best models and allows resume from checkpoints.  
  - Learning rate scheduling and weight decay.  
  - TensorBoard logging for loss curves and metrics.  

- **Metrics**  
  - AUROC and AUPRC (suitable for **imbalanced classification**).  
  - Accuracy, Precision, Recall, F1-score.  
  - Per-batch and per-epoch logging.  

- **Loss Functions**  
  - `CrossEntropyLoss` with label smoothing.  
  - `BCEWithLogitsLoss` with `pos_weight` for skewed datasets.  
  - Easy extension to **focal loss** or other imbalance-aware functions.  

- **Custom Optimizers**  
  - RMSProp reimplemented from scratch (educational).  
  - Support for Adam, SGD, and PyTorch-native optimizers.  

---

## Objective  

- Train CNN models to distinguish between **tornado** and **non-tornado** radar observations.  
- Evaluate performance using **rare event metrics** rather than raw accuracy.  
- Explore techniques to address **extreme class imbalance**.  
- Provide a **baseline** for future research into AI-driven severe weather detection.  

---

## Tech Stack  

- **Core**: Python 3.9+, PyTorch, PyTorch Lightning  
- **Evaluation**: TorchMetrics  
- **Visualization**: Matplotlib, TensorBoard  
- **Utilities**: NumPy, Pandas, Scikit-learn  
