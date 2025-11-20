# DMML_GROUP_12: Used Car Resale Price Prediction and Image Classification

## Project Overview
This project implements and compares three different machine learning regression algorithms to accurately predict the resale price of used cars:
1. **Linear Regression**
2. **XGBoost** (Extreme Gradient Boosting)
3. **Random Forest**

Additionally, it includes an **Image Classification Model (CNN)** using transfer learning to extract core car attributes (Maker, Body Type, Colour, Model) directly from various angled images of cars.

### Team Members
* Aditya Vikram Singh Dahiya
* Joseph Cherian
* Laiba Shehzad
* Mohammed Nibras
* Vir Wadwani


---

## Execution Environments

This project is designed to run in two specific environments to optimize performance for each task:

* **Regression Models (Linear Regression, XGBoost, Random Forest):**
    These models are implemented and executed in **Google Colab**. The code handles data preprocessing and model training on the tabular dataset.

* **Image Classification (CNN):**
    The Convolutional Neural Network was developed and trained **locally on an Apple M5 Pro** to leverage local GPU acceleration (Metal Performance Shaders - MPS) for efficient deep learning training.

---

## Data Sources
The models are trained using the DVM-CAR dataset organized into the following components:

### Tabular Data
*Used for Regression tasks in Colab.*
* **`Adv_table.csv`:** Contains car advertisement data, including mileage, engine size, body type, fuel type, and the target variable: `'Price'` (Resale Price).
* **`Price_table.csv`:** Contains MSRP data (`'Entry_price'`) used for feature enrichment.

### Image Data
*Used for CNN Classification tasks locally.*
* **`Basic_table.csv`:** Used by the CNN to source the `'Automaker'` label.
* **`Image_table.csv`:** Links `'Adv_ID'` to image file names and viewpoint data.
* **Images:** A root folder containing resized car images used for the classification tasks.

---

## Prerequisites & Setup (Google Colab)
The regression analysis is designed to run in a Google Colab environment connected to Google Drive.

**1. Dependencies:**
The tabular regression models rely on standard scientific Python libraries (`pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`).

**2. Google Drive Connection:**
Before running any modeling cells in the Colab notebook (`DMML_GROUP_12.ipynb`), you must execute the initial setup cell to mount your Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

# Car Maker, Body Type, Color & Model Classification

Research question: *Can Body type and manufacturer be accurately inferred from listing images of various angles of backgroundless cars?*

This project fine-tunes an EfficientNet-B0 backbone with four softmax heads to predict:
1. Manufacturer (brand)
2. Body type (derived by mapping `Genmodel_ID` from image filename to `Adv_table.csv`)
3. Exterior color (taken from the folder name)
4. Model (brand-normalized: `maker::model_folder`)

## Data Assumptions
Directory layout: `resized_DVM/Brand/Model/Year/Color/*.jpg`  
Filename pattern example: `Bentley$$Arnage$$2003$$Silver$$10_1$$21$$image_0.jpg`
- Token[4] = `Genmodel_ID`. We use this to look up Bodytype in `Adv_table.csv`.
- Year is ignored for classification per requirements.

Each (Brand, Model, Year, Color) group is split 80% train / 20% test per group's images to reduce leakage across similar views.

### Color & Model
- Color folders (e.g., `.../2017/Red/`) and model folder names now feed dedicated heads. Colors/models are mapped to contiguous indices stored inside checkpoints.
- Years remain unused.

## Environment
Use the existing `venv/` in this workspace:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

## CNN Training 
# We were forced to run the CNN model locally on a MacBook Pro laptop M5, as we ran out of compilations time on Google Colab as well as the GPU available on the M5 was more powerful and could train faster as our dataset was exceptionally large containing roughly ~ 1.4 Million images.
```bash
python train.py --epochs 10 --batch_size 64 --lr 1e-3 --num_workers 8 --pin_memory --device mps
```

Key levers for Apple Silicon:
- `--device mps` to keep all matrix math on the GPU.
- `--num_workers 8` and `--pin_memory` to overlap dataloading with GPU execution.
- `--freeze_backbone` when you only want to adapt the classifier heads (useful for quick color/model warm-up).
- `--resume_from checkpoint.pt` to continue from an existing maker/body checkpoint instead of retraining.

**Two-phase fine-tune to add color + model without a full retrain**
1. Warm the new heads while freezing EfficientNet (fast ~6–8 minutes for 2 epochs on an M2 Pro when using 64× batches over the 80% training split):
	 ```bash
	 python train.py \
		 --resume_from checkpoint.pt \
		 --epochs 2 \
		 --batch_size 64 \
		 --lr 1e-3 \
		 --num_workers 8 \
		 --pin_memory \
		 --freeze_backbone \
		 --device mps \
		 --output checkpoint_stage1.pt
	 ```
2. Unfreeze the backbone and continue training (≈40–50 minutes for 8 epochs on the same hardware, still far quicker than a full-from-scratch run because weights start from the prior maker/body solution):
	 ```bash
	 python train.py \
		 --resume_from checkpoint_stage1.pt \
		 --epochs 8 \
		 --batch_size 64 \
		 --lr 5e-4 \
		 --num_workers 8 \
		 --pin_memory \
		 --device mps \
		 --output checkpoint_color_model.pt
	 ```

This keeps total wall-clock under an hour on modern Apple Silicon while fully leveraging the GPU and ensuring the new heads converge quickly. Adjust epoch counts if you observe under/overfitting—macro F1 for model/color tends to stabilize after 6–8 unfrozen epochs.

Every run prints maker/body/color/model accuracy, precision, recall, and F1 per epoch, then stores all label mappings inside the checkpoint.

## Evaluation
```bash
python evaluate.py --checkpoint checkpoint_color_model.pt --batch_size 64 --device mps
```
Reports maker/body/color/model macro metrics on the held-out split (color/model are omitted automatically if the checkpoint predates those heads).

## Inference on New Images
```bash
python predict.py --image path/to/car.jpg --checkpoint checkpoint.pt --device mps
```
Outputs the top-k manufacturer, body type, color, and model predictions (default 3 each) with probabilities. Heads are shown only when they exist in the checkpoint.

### Working with the Test Split
- Export a list (and optional copy) of the 20% held-out images:
	```bash
	python export_test_split.py --checkpoint checkpoint.pt --list_out test_images.txt --copy_to test_subset --relative
	```
	- `test_images.txt` records every test image path (relative to `resized_DVM` when `--relative` is set).
	- `test_subset/` will contain copies of those files, preserving the brand/model/year/color subfolders. Skip `--copy_to` if you only need the list.
	- Use the generated paths with `predict.py` to run batch or ad-hoc inference on held-out images.

## Train/Test Split Details
- Each `(brand, model, year, color)` folder is split **once** into 80% train and 20% test images.
- Training loops only consume the train subset; the test subset stays untouched until evaluation.
- To inspect the actual filenames chosen for each split:
	```python
	from src.dataset import CarDataset
	train_ds = CarDataset('resized_DVM','Adv_table.csv','train')
	test_ds = CarDataset('resized_DVM','Adv_table.csv','test')
	print(len(train_ds), len(test_ds))
	print(train_ds.samples[:5])
	print(test_ds.samples[:5])
	```
	Each `samples` entry contains `(path, maker_idx, body_idx, color_idx, model_idx)`; indices map back to class names via the dataset's mapping dictionaries.

## Model
`DualHeadCarNet` wraps EfficientNet-B0 and replaces the classifier with up to four linear heads (maker/body/color/model). Apply softmax to each head's logits for probabilities. Missing heads are skipped automatically so older checkpoints stay usable.

## Extending
- Add finer-grained heads (trim level, drivetrain, etc.) using the same pattern if more metadata becomes reliable.
- Implement hierarchical loss (manufacturer -> model -> body type).
- Consider weighted sampling if class imbalance is high.

## Metrics
Primary success: macro F1 and accuracy for body type & manufacturer on test set.
