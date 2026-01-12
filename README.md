# User Guide for Training a Food Recognition Model

## Requirements
- Python version 3.10  
- `requirements.txt`

## Directory Structure

```
Trenowanie/
├─ model/
│  ├─ best_model.h5
│  ├─ converter.py
│  ├─ history_finetune.json
│  ├─ training_results.png
│  ├─ converted/
│  │  ├─ converted_model.tflite
├─ dataset/
├─ GPUtrening.py
├─ instrukcje.txt
├─ requirements.txt
├─ sortowanie.py
```

## 1. Download the Dataset

Download the dataset from Kaggle:

https://www.kaggle.com/datasets/dansbecker/food-101

Extract the dataset into the `Trenowanie` directory.

## 2. Create a Virtual Environment

### PyCharm
Go to:  
`Settings → Project → Python Interpreter → Add… → New Virtualenv`  

Select the `.venv` folder, then choose **Install from Requirements** and point to the `requirements.txt` file located in the project directory.

### Windows PowerShell
```bash
py -3.10 -m venv .venv; .\.venv\Scripts\Activate.ps1
```

### Windows CMD
```bash
py -3.10 -m venv .venv && .\.venv\Scripts\activate
```

### Linux/macOS (bash/zsh)
```bash
python3.10 -m venv .venv && source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Split the Dataset into Training and Test Sets

```bash
python sortowanie.py
```

The script splits the dataset into:
- 80% training set  
- 20% test set  

This follows the recommendations provided by the creators of the dataset.

## 5. Start Model Training

```bash
python GPUtrening.py
```

After the training process is completed, the trained model will be saved in the `model` directory.

## 6. Test the Model

```bash
python model/test.py
```

This is a simple program that allows you to select an image and displays the model’s prediction results.
