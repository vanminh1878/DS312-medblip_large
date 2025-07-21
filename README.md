# ImageCLEFmedical 2025 Caption Competition - BLIP Model

This repository contains the code and instructions for training, predicting, and postprocessing using the BLIP model for the ImageCLEFmedical 2025 Caption competition.

## Overview
Our team utilized the BLIP model to generate image captions for the competition, using the official dataset provided by the organizers. This README provides instructions for training the model, generating predictions, and postprocessing the results.

## Requirements
- Python 3.x  
- Required libraries: PyTorch, pandas, numpy, transformers, tqdm (please specify all dependencies in your environment)  
- Official ImageCLEFmedical 2025 dataset  
- Pre-trained weights file (`blip_large.pth`) if loading pre-trained weights

## Training
To train the BLIP model, run the following command:

```bash
python main.py train \
  --root_path /kaggle/input/official-imageclef-2025-dataset/ \
  --batch_size 4 \
  --num_epochs 2 \
  --lr 1e-5 \
  --load_weights True


Parameters

--root_path: Path to the official ImageCLEFmedical 2025 dataset (default: ./)
--batch_size: Number of samples per batch (default: 4)
--num_epochs: Number of training epochs (default: 16)
--lr: Learning rate (default: 1e-5)
--load_weights: Load pre-trained weights (default: False)
--path_weights: Path to save/load weights (default: ./)

Note: The trained weights are saved as blip_large.pth in the path_weights directory, which is used for both saving and loading weights during training.

```
## Prediction
To generate captions for the test and validation sets, run:
```bash
python main.py predict --root_path /kaggle/input/official-imageclef-2025-dataset/ --path_weights /kaggle/working/

Parameters

--root_path: Path to the official ImageCLEFmedical 2025 dataset for prediction (default: ./)
--path_weights: Path to the trained weights file (default: ./blip_large.pth)
```

Output: The prediction process generates two files:

run.csv: Predicted captions for the test set
valid.csv: Predicted captions for the validation set

## Postprocessing
To refine the generated captions, execute the postprocessing script:
jupyter notebook Postprocessing.ipynb

Follow the instructions in the Postprocessing.ipynb notebook to process the run.csv and valid.csv files for improved results.
Evaluation
To evaluate the model’s performance, use:
python evaluate.py eval --root {path to evaluation dataset} --score {rouge, bleu, meteor, bertscore}

Parameters

--root: Path to the official ImageCLEFmedical 2025 evaluation dataset
--score: Evaluation metric (options: rouge, bleu, meteor, bertscore)

## Notes

Ensure the dataset path (/kaggle/input/official-imageclef-2025-dataset/) points to the official ImageCLEFmedical 2025 dataset provided by the organizers.
The path_weights directory must contain blip_large.pth if --load_weights is True.
Verify that the Postprocessing.ipynb notebook is in the working directory or provide the correct path.


