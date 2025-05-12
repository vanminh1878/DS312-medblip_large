from torch.utils.data import Dataset
from PIL import Image
import os

class ImgCaptionDataset(Dataset):
    def __init__(self, df, path, processor, image_size=(224, 224), max_length=64):
        self.df = df
        self.path = path
        self.processor = processor
        self.image_size = image_size
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]["ID"]
        caption = self.df.iloc[idx]["Caption"]
        
        # Đảm bảo img_id có phần mở rộng .jpg
        if not img_id.endswith(".jpg"):
            img_id = img_id + ".jpg"
        
        img_path = os.path.join(self.path, img_id)
        image = Image.open(img_path).convert("RGB")
        
        # Resize hình ảnh
        image = image.resize(self.image_size)
        
        # Sử dụng prompt chuẩn cho MiniCPM
        prompt = "<|im_start|>user: Please provide a caption for this image. <|im_end|><|im_start|>assistant: "
        caption_with_prompt = f"{prompt}{caption}<|im_end|>"
        
        # Tokenize hình ảnh và caption
        encoding = self.processor(
            text=caption_with_prompt,
            images=image,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
            truncation=True
        )
        
        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }