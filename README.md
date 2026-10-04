# 👕 Fashion-MNIST — CNN & Transfer Learning

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.16-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Airflow](https://img.shields.io/badge/Airflow-2.9-017CEE?logo=apacheairflow&logoColor=white)](https://airflow.apache.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/RafaelGallo/Convolutional-Neural-Network-Transfer-Learning)](https://github.com/RafaelGallo/Convolutional-Neural-Network-Transfer-Learning/commits/main)

![](https://github.com/RafaelGallo/Convolutional-Neural-Network-Transfer-Learning/blob/main/img/2c79ca3a-f4b3-48a9-8843-c1105d9fbd22.jpg?raw=true)

## Table of contents

1. [Business problem](#business-problem)
2. [Key results](#key-results)
3. [Dataset](#dataset)
4. [Project structure](#project-structure)
5. [Methodology](#methodology)
6. [Results in detail](#results-in-detail)
7. [Streamlit app](#streamlit-app)
8. [How to run](#how-to-run)
9. [Limitations and next steps](#limitations-and-next-steps)
10. [Troubleshooting](#troubleshooting)
11. [License and acknowledgements](#license-and-acknowledgements)

## Business problem

An online fashion retailer receives thousands of new product images every week from suppliers and
marketplace sellers. Each image is categorized by hand (T-shirt, Trouser, Dress, Sneaker, Bag, etc.),
which is slow, inconsistent and expensive. Miscategorized products lower search relevance, hurt
recommendations and reduce conversion.

**Objective:** build image classifiers that assign each image to one of 10 apparel categories, and
decide whether Transfer Learning models outperform a custom CNN trained from scratch.

**Success criteria**

- Accuracy and macro F1-score on the 10,000-image test set higher than the baseline CNN, with a
  margin of at least 0.005 to count as a real gain.
- Identification of the most confused classes to guide a human review process.
- Trade-off analysis between performance, training time and model size for deployment.

## Key results

Test set: 10,000 images (1,000 per class).

| Model | Accuracy | Macro F1 | Macro AUC | Parameters | Training time (s) |
|---|---|---|---|---|---|
| **CNN baseline (from scratch)** | **0.9421** | **0.9419** | **0.9974** | 288,618 | not recorded |
| VGG19 | 0.9238 | 0.9233 | 0.9955 | 20,029,514 | 1,191 |
| VGG16 | 0.9226 | 0.9220 | 0.9953 | 14,719,818 | 956 |
| ResNet50 | 0.9205 | 0.9199 | 0.9953 | 23,608,202 | 492 |
| MobileNetV2 | 0.9165 | 0.9163 | 0.9951 | 2,270,794 | 185 |
| EfficientNetB0 | 0.9080 | 0.9078 | 0.9942 | 4,062,381 | 235 |

- **The baseline CNN wins.** It is 1.8 points above the best backbone in accuracy (VGG19), a gap larger
  than 5 standard errors, with 8 to 82 times fewer parameters.
- **No Transfer Learning model met the success criterion** under this configuration.
- **Shirt is the bottleneck for every model.** Its F1 is 0.826 in the baseline and 0.733 to 0.771
  in the backbones.
- **Recommendation:** deploy the baseline CNN and send low-confidence predictions among
  T-shirt/top, Shirt, Pullover and Coat to human review.

## Hardware and environment

| Item | Value |
|---|---|
| Platform | Google Colab, used from VS Code with the official Colab extension |
| Accelerator | GPU, NVIDIA Tesla T4 (compute capability 7.5, about 15 GB) |
| Python | 3.13.15 |
| TensorFlow | 2.20.0 (with the bundled Keras 3) |
| Local OS (data download, SQLite, app) | Windows, Python 3.10+ |

All models were trained on the same Colab runtime setup, so the training times in the results
table are comparable between models. They depend on the GPU model and on the load of the shared
machine, so expect different values on other hardware.

- **Training:** Colab runtime with a T4 GPU, batch size 128, seed 42.
- **Data:** the notebook reads the CSV files from the Kaggle dataset cache (`/kaggle/input/fashionmnist`),
  downloaded with `kagglehub`. The same files are copied to `input/` and to a SQLite database by
  `download_dataset.py`, which runs on the local Windows machine.
- **Inference app:** runs on CPU. The baseline CNN has 288,618 parameters (about 1.1 MB).
- **Outputs:** models and results are saved to Google Drive, because the Colab `/content` folder
  is erased when the runtime restarts.

> **GPU note:** the runtime must have a GPU supported by TensorFlow 2.20. An NVIDIA Blackwell GPU
> (compute capability 12.0) fails with `CUDA_ERROR_INVALID_HANDLE`, so a T4 or L4 was used.

## Dataset

[Fashion MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist) from Zalando Research:
70,000 grayscale images of 28 x 28 pixels in 10 balanced classes.

| Label | Class | Label | Class |
|---|---|---|---|
| 0 | T-shirt/top | 5 | Sandal |
| 1 | Trouser | 6 | Shirt |
| 2 | Pullover | 7 | Sneaker |
| 3 | Dress | 8 | Bag |
| 4 | Coat | 9 | Ankle boot |

| Split | Images | Per class | Use |
|---|---|---|---|
| Train | 48,000 | 4,800 | Fit the weights |
| Validation | 12,000 | 1,200 | Model selection and learning curves |
| Test | 10,000 | 1,000 | Final evaluation (official Zalando test set) |

The 60,000 original training images were split 80/20 with `stratify=y` and `random_state=42`. The same
split is used by every model, so the comparison is fair. Pixels are scaled to [0, 1].

![Class distribution](https://github.com/RafaelGallo/Convolutional-Neural-Network-Transfer-Learning/blob/main/output/__results___27_0.png?raw=true)

The classes are perfectly balanced (6,000 images each in the full training set), so accuracy is a
reliable metric and no resampling is needed.

![Random samples per class](https://github.com/RafaelGallo/Convolutional-Neural-Network-Transfer-Learning/blob/main/output/__results___29_0.png?raw=true)

The average image of each class shows why four upper-body classes (T-shirt/top, Pullover, Coat and
Shirt) are hard: their templates are almost the same silhouette.

![Average image per class](https://github.com/RafaelGallo/Convolutional-Neural-Network-Transfer-Learning/blob/main/output/__results___31_0.png?raw=true)

## Project structure

```
Convolutional-Neural-Network-Transfer-Learning/
├── input/                  # Fashion MNIST CSV files (not versioned)
├── sql/                    # SQLite database with the dataset (not versioned)
├── mflow/                  # MLflow runs and model artifacts (not versioned)
├── images/                 # Figures used in this README
├── download_dataset.py     # Downloads the dataset and builds the SQLite database
├── app.py                  # Streamlit app
├── notebook.ipynb          # Data analysis, CNN, Transfer Learning and evaluation
├── .env                    # Kaggle credentials (not versioned)
├── .gitignore
└── README.md
```

Rename `app.py` and `notebook.ipynb` to match your actual file names.

## Methodology

### Baseline CNN (trained from scratch)

Three convolutional blocks followed by a dense classifier, working on 28 x 28 x 1 inputs.

| Stage | Layers |
|---|---|
| Block 1 | Conv2D 32 + BatchNorm, Conv2D 32, MaxPool, Dropout 0.25 |
| Block 2 | Conv2D 64 + BatchNorm, Conv2D 64, MaxPool, Dropout 0.25 |
| Block 3 | Conv2D 128 + BatchNorm, MaxPool, Dropout 0.25 |
| Head | Flatten, Dense 128 (ReLU), Dropout 0.5, Dense 10 (softmax) |

- 288,618 parameters (about 1.1 MB), about 22 M multiply-accumulates per image, so it runs on CPU.
- Adam (learning rate 1e-3), sparse categorical cross-entropy, batch size 128, up to 50 epochs,
  seed 42.

![Baseline training curves](https://github.com/RafaelGallo/Convolutional-Neural-Network-Transfer-Learning/blob/main/output/__results___44_0.png?raw=true)

Validation accuracy plateaus near 0.93 to 0.94 from epoch 30. After that the training loss keeps
falling while the validation loss stays flat, which is mild overfitting.

### Transfer Learning

Five ImageNet backbones share the same wrapper, so differences come from the backbone only.

```
28x28x1 -> Resizing 96x96 -> Concatenate x3 (RGB) -> backbone-specific scaling
        -> backbone (include_top=False, pooling="avg") -> Dropout 0.3 -> Dense 10 (softmax)
```

Resizing, channel replication and scaling live inside the model, so inference takes raw 28 x 28
images and no 96 x 96 x 3 copy of the dataset is stored in memory (about 5.3 GB for the training set).

| Backbone | Scaling | Feature dim | Layers unfrozen in phase 2 |
|---|---|---|---|
| MobileNetV2 | [-1, 1] | 1,280 | 30 |
| VGG16 | [0, 255] minus ImageNet mean | 512 | 4 |
| VGG19 | [0, 255] minus ImageNet mean | 512 | 5 |
| ResNet50 | [0, 255] minus ImageNet mean | 2,048 | 30 |
| EfficientNetB0 | [0, 255] (normalizes internally) | 1,280 | 30 |

**Two-phase training**

1. **Feature extraction:** backbone frozen, Adam with learning rate 1e-3, 8 epochs, batch size 128.
2. **Fine-tuning:** last layers unfrozen (BatchNormalization stays frozen), Adam with learning rate
   1e-5, 5 epochs, batch size 128.

Every model used its full epoch budget, so the backbones were limited by the budget and not by convergence.

## Results in detail

### Baseline classification report

| Class | Precision | Recall | F1 |
|---|---|---|---|
| T-shirt/top | 0.9010 | 0.9010 | 0.9010 |
| Trouser | 0.9950 | 0.9940 | 0.9945 |
| Pullover | 0.9216 | 0.9050 | 0.9132 |
| Dress | 0.9425 | 0.9500 | 0.9462 |
| Coat | 0.8922 | 0.9350 | 0.9131 |
| Sandal | 0.9949 | 0.9800 | 0.9874 |
| **Shirt** | **0.8420** | **0.8100** | **0.8257** |
| Sneaker | 0.9636 | 0.9790 | 0.9712 |
| Bag | 0.9930 | 0.9910 | 0.9920 |
| Ankle boot | 0.9741 | 0.9760 | 0.9750 |
| **Accuracy** | | | **0.9421** |
| **Macro avg** | 0.9420 | 0.9421 | 0.9419 |

### Shirt across models

| Model | Shirt recall | Shirt F1 |
|---|---|---|
| CNN baseline | 0.8100 | 0.8257 |
| VGG16 | 0.7420 | 0.7705 |
| VGG19 | 0.7340 | 0.7666 |
| ResNet50 | 0.7190 | 0.7572 |
| MobileNetV2 | 0.7390 | 0.7495 |
| EfficientNetB0 | 0.7250 | 0.7331 |

### Confusion matrix (baseline)

![Baseline confusion matrix](https://github.com/RafaelGallo/Convolutional-Neural-Network-Transfer-Learning/blob/main/output/__results___51_0.png?raw=true)

The baseline makes 579 errors in 10,000 images. The four upper-body classes account for 449 of them
(about 78%), and Shirt alone for 190 (about 33%).

| Confusion | Errors |
|---|---|
| Shirt predicted as T-shirt/top | 77 |
| T-shirt/top predicted as Shirt | 64 |
| Shirt predicted as Coat | 53 |
| Shirt predicted as Pullover | 37 |
| Pullover predicted as Shirt | 36 |
| Pullover predicted as Coat | 35 |
| Ankle boot predicted as Sneaker | 24 |

### ROC curves (one-vs-rest)

![Baseline ROC curves](https://github.com/RafaelGallo/Convolutional-Neural-Network-Transfer-Learning/blob/main/output/__results___47_0.png?raw=true)

The baseline has the highest ROC curve for the macro average at every operating point.

| Class group | AUC range |
|---|---|
| Trouser, Bag, Sandal, Sneaker, Ankle boot | 0.9996 to 1.0000 |
| Dress, Coat, Pullover, T-shirt/top | 0.9956 to 0.9983 |
| Shirt | 0.9872 |

To find 80% of the Shirts, the baseline accepts about 135 false alarms. The backbones need between
about 250 and 360.

### Transfer Learning example: MobileNetV2

![MobileNetV2 training curves](https://github.com/RafaelGallo/Convolutional-Neural-Network-Transfer-Learning/blob/main/output/__results___76_0.png?raw=true)

The curves show the usual pattern. Validation accuracy plateaus near 0.89 with the backbone frozen,
then improves after the fine-tuning start (dashed line).

![MobileNetV2 ROC curves](https://github.com/RafaelGallo/Convolutional-Neural-Network-Transfer-Learning/blob/main/output/__results___82_2.png?raw=true)

### Predictions and label noise

![Baseline predictions](https://github.com/RafaelGallo/Convolutional-Neural-Network-Transfer-Learning/blob/main/output/__results___60_0.png?raw=true)

The most confident errors (about 100% probability) are dominated by doubtful images, for example
a plain short-sleeve top labelled Shirt, or sneakers and ankle boots that are hard to tell apart.

![Most confident errors](https://github.com/RafaelGallo/Convolutional-Neural-Network-Transfer-Learning/blob/main/output/__results___61_0.png?raw=true)

This points to label noise and class ambiguity (a visual judgment that needs a manual audit). These
errors cannot be removed by a confidence threshold and limit the accuracy any model can reach.

### Why Transfer Learning did not win (hypotheses)

- **Domain gap:** ImageNet features come from color, high-resolution photos, while the input is a
  smooth grayscale image upscaled 3.4 times. Copying one channel into three collapses the color filters
  of the first layer.
- **Small final feature map:** at 96 x 96 the last map is 3 x 3, which loses details (collars, buttons,
  zippers) that separate Shirt from its neighbors.
- **Frozen BatchNorm** keeps ImageNet statistics in MobileNetV2, ResNet50 and EfficientNetB0.
- **Shallow fine-tuning:** `n_unfreeze` counts layers and not parameters, and 5 epochs at 1e-5 change
  the backbone only slightly.

## Streamlit app

The app classifies your own image with the model you choose.

- Pick the model in the sidebar. It shows the test accuracy, macro F1 and parameter count.
- Upload a PNG, JPG or JPEG of a clothing item or shoe (simple background, centered).
- The image is converted to 28 x 28 grayscale, as in training.
- Use **Inverter cores** (invert colors) for photos with a light background, since Fashion MNIST
  items are light on a dark background.
- The app shows the predicted class, the confidence and the probability of every class.

| Upload and preprocessing | Prediction |
|---|---|
| ![App upload](https://github.com/RafaelGallo/Convolutional-Neural-Network-Transfer-Learning/blob/main/output/08.png?raw=true) | ![App prediction](https://github.com/RafaelGallo/Convolutional-Neural-Network-Transfer-Learning/blob/main/output/09.png?raw=true) |
| ![App upload sneaker](https://github.com/RafaelGallo/Convolutional-Neural-Network-Transfer-Learning/blob/main/output/010.png?raw=true) | ![App prediction sneaker](https://github.com/RafaelGallo/Convolutional-Neural-Network-Transfer-Learning/blob/main/output/011.png?raw=true) |

Run it with:

```bash
streamlit run app.py
```

## How to run

### 1. Clone and install

```bash
git clone https://github.com/RafaelGallo/Convolutional-Neural-Network-Transfer-Learning.git
cd Convolutional-Neural-Network-Transfer-Learning

python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate

pip install tensorflow scikit-learn pandas numpy matplotlib seaborn pillow \
            kagglehub python-dotenv streamlit mlflow
```

### 2. Kaggle credentials

Create a token in Kaggle under **Settings > API** and save it in a `.env` file in the project root:

```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key
```

If your token is in the new format (starts with `KGAT_`), use `KAGGLE_API_TOKEN=your_token` instead.
Never commit this file.

### 3. Download the dataset

```bash
python download_dataset.py
```

Run it in a local terminal. The script downloads the dataset with `kagglehub`, copies the two CSV
files to `input/` and loads them into the SQLite database `sql/fashion_mnist.db`
(tables `fashion_mnist_train` and `fashion_mnist_test`).

### 4. Train and evaluate

Open the notebook and run the cells in order: data loading, baseline CNN, Transfer Learning models,
and evaluation (classification report, confusion matrix, ROC and predictions).

For a GPU, use Kaggle or Google Colab (a T4 or L4 works well). With the Colab extension for VS Code,
the notebook cells run on the remote runtime, so paths such as `C:\...` do not exist there.
Save the models to Google Drive or download them before the runtime ends.

### 5. Models

The `.keras` files are not stored in this repository (GitHub limits files to 100 MB). Train them with the
notebook or download them from: `<add link>`. Then place them where `app.py` expects them.

## Limitations and next steps

- Results come from a single run and a single random seed. Accuracy differences below about 0.005,
  and per-class recall differences below about 0.02 to 0.03, are within the uncertainty of the test set.
- The backbones were evaluated with 96 x 96 inputs and short fine-tuning.

Next steps:

- Fine-tune longer, with more unfrozen layers, a higher learning rate and early stopping.
- Try 128 x 128 inputs (final feature map of 4 x 4).
- Run paired tests against the baseline (McNemar for accuracy, DeLong or paired bootstrap for AUC).
- Test an ensemble of the baseline with the best backbone.
- Audit the labels of images that all models get wrong, and report accuracy on both the official
  and the cleaned test set.
- Measure calibration (ECE) and test temperature scaling before using a confidence threshold in production.

## Troubleshooting

| Problem | Cause and solution |
|---|---|
| `CUDA_ERROR_INVALID_HANDLE` on the first TensorFlow operation | The GPU is an NVIDIA Blackwell (compute capability 12.0), not supported by TensorFlow 2.20. Switch to a T4 or L4 runtime. |
| `FileNotFoundError` for a `C:\...` path in Colab | The kernel is remote (Linux). Use a local kernel, or read from `/kaggle/input/fashionmnist`. |
| `git push` rejected for files above 100 MB | Do not version CSV, `.db` or `.keras` files. Keep `input/`, `sql/*.db`, `mflow/` and `*.keras` in `.gitignore`. |
| Kaggle error 401 or 403 | Check the credentials in `.env` and accept the dataset terms once on its Kaggle page. |

Recommended `.gitignore`:

```
.env
input/
mflow/
mlruns/
sql/*.db
*.keras
*.h5
```

## License and acknowledgements

- Dataset: Fashion-MNIST, Zalando Research, MIT License.
  [github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)
- Converted to CSV by the script at [pjreddie.com/projects/mnist-in-csv](https://pjreddie.com/projects/mnist-in-csv/).
- Add a `LICENSE` file to this repository to define the terms of the code.

**Author:** [@RafaelGallo](https://github.com/RafaelGallo)
