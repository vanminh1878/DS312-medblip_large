# Import important libraries
import pandas as pd
import transformers
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoProcessor

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize
import os

from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt

from dataset import ImgCaptionDataset
import argparse

def save_df_to_csv(df, file_path):
    df.to_csv(file_path, index=False)

def load_df(dir_caption):
    """
    Function load dataframe
    """
    return pd.read_csv(dir_caption, delimiter=",")

def train(root_path, batch_size=4, num_epochs=2, lr=1e-5, load_weights=False, path_weights="/kaggle/working/"):
    """
    Function training
    root_path: root folder of dataset
    batch_size: batch size
    num_epochs: number of epochs
    lr: learning rate
    load_weights: load weights
    path_weights: path for weights
    """
    # Define paths
    train_dir = os.path.join(root_path, "train/train")
    train_captions = os.path.join(root_path, "train/train/train_captions.csv")

    # Load dataset csv
    df_train = load_df(dir_caption=train_captions)

    # Load weights
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float32)

    # Parameters
    image_size = (224, 224)
    max_length = 200

    # Initialize step counter
    global_step = 0

    # Load weights if load_weights=True
    if load_weights:
        checkpoint_path = os.path.join(path_weights, "medblip_large.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            global_step = checkpoint['global_step']
            print(f"Loaded checkpoint from step {global_step}")
        else:
            print(f"No checkpoint found at {checkpoint_path}, starting from scratch")
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Set up cuda and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    # Create dataset for training
    train_dataset = ImgCaptionDataset(
        df=df_train,
        path=train_dir,
        processor=processor,
        image_size=image_size,
        max_length=max_length
    )
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    # Start training
    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        for batch in tqdm(train_dataloader):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device)
            attention_marks = batch.pop("attention_mask").to(device)
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                labels=input_ids,
                attention_mask=attention_marks
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

            # Save checkpoint every 100 steps
            if global_step % 100 == 0:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step,
                    'epoch': epoch,
                    'loss': loss.item()
                }
                checkpoint_path = os.path.join(path_weights, "medblip_large.pth")
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint at step {global_step}: {checkpoint_path}")

        # Save checkpoint at the end of each epoch
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
            'epoch': epoch,
            'loss': loss.item()
        }
        checkpoint_path = os.path.join(path_weights, "medblip_large.pth")
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint at end of epoch {epoch}: {checkpoint_path}")
        print("Loss:", loss.item())

def predict(root_path, path_weights="/kaggle/working/"):
    """
    root_path: root folder of dataset
    path_weights: path for weights file
    """
    # Load weights pretrained
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    model.load_state_dict(torch.load(os.path.join(path_weights, "medblip_large.pth")))
    model.eval()
    model.to("cuda")

    df_valid = os.path.join(root_path, "valid/valid/valid_captions.csv")
    df_valid = pd.read_csv(df_valid)
    dir_valid = os.path.join(root_path, "valid/valid")

    dir_test = os.path.join(root_path, "test/test")
    dir_caption = os.path.join(root_path, "valid/valid/valid_captions.csv")

    test_ID = os.listdir(dir_test)
    for i in range(len(test_ID)):
        test_ID[i] = test_ID[i].replace(".jpg", "")

    def get_inferences(IDs, model, paths, max_new_tokens=200):
        """
        Function to get inferences
        """
        data = []
        for ID in tqdm(IDs):
            path = os.path.join(paths, ID + ".jpg")
            image = cv2.imread(path)
            image = cv2.resize(image, (224, 224))
            inputs = processor(image, return_tensors="pt").to("cuda")
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                no_repeat_ngram_size=2,
                num_beams=5
            )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            data.append([ID, generated_text])

        df = pd.DataFrame(data, columns=['ID', 'Caption'])
        return df

    # Get inferences test
    test_results = get_inferences(test_ID, model, dir_test)
    save_df_to_csv(test_results, "/kaggle/working/run.csv")

    # Get inferences valid
    valid_ID = df_valid["ID"]
    valid_results = get_inferences(valid_ID, model, dir_valid)
    save_df_to_csv(valid_results, "/kaggle/working/valid.csv")
    len(valid_results)

def main():
    # Initializes a parser for command-line arguments
    parser = argparse.ArgumentParser()

    # Creates subparsers for different commands
    subparsers = parser.add_subparsers(dest='command')

    # Adds a subparser for the 'train' command
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--root_path', type=str, default='/kaggle/input/oggyyy-dataset/')
    parser_train.add_argument('--batch_size', type=int, default=4)
    parser_train.add_argument('--num_epochs', type=int, default=16)
    parser_train.add_argument('--lr', type=float, default=1e-5)
    parser_train.add_argument('--load_weights', type=bool, default=False)
    parser_train.add_argument('--path_weights', type=str, default='/kaggle/working/')

    # Adds a subparser for the 'predict' command
    parser_predict = subparsers.add_parser('predict')
    parser_predict.add_argument('--root_path', type=str, default='/kaggle/input/oggyyy-dataset/')
    parser_predict.add_argument('--path_weights', type=str, default='/kaggle/working/')

    # Parses the command-line arguments
    args = parser.parse_args()

    if args.command == 'train':
        train(args.root_path, args.batch_size, args.num_epochs, args.lr, args.load_weights, args.path_weights)
    elif args.command == 'predict':
        predict(args.root_path, args.path_weights)

if __name__ == "__main__":
    main()