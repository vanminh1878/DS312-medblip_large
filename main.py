# Nhập các thư viện cần thiết
import pandas as pd
import transformers
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoProcessor
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize
import os
import glob
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from dataset import ImgCaptionDataset
import argparse

def save_df_to_csv(df, file_path):
    """Lưu dataframe vào file CSV"""
    df.to_csv(file_path, index=False)

def load_df(dir_caption):
    """Tải dataframe từ file CSV"""
    return pd.read_csv(dir_caption, delimiter=",")

def train(root_path, batch_size=4, num_epochs=2, lr=1e-5, load_weights=False, path_weights="/kaggle/working/"):
    """
    Hàm huấn luyện mô hình
    root_path: thư mục gốc của dataset
    batch_size: kích thước batch
    num_epochs: số epoch
    lr: tốc độ học
    load_weights: tải trọng số đã lưu
    path_weights: đường dẫn lưu trọng số
    """
    # Định nghĩa thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")

    # Định nghĩa các đường dẫn
    train_dir = os.path.join(root_path, "train/train")
    train_captions = os.path.join(root_path, "train/train/train_captions.csv")

    # Tải dataset từ CSV
    df_train = load_df(dir_caption=train_captions)

    # Tải mô hình và processor
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float32)
    model = model.to(device)  # Chuyển mô hình sang GPU ngay sau khi khởi tạo

    # Thiết lập tham số
    image_size = (224, 224)
    max_length = 200

    # Khởi tạo optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Khởi tạo bộ đếm bước và epoch
    global_step = 0
    start_epoch = 0

    # Tải trọng số nếu load_weights=True
    if load_weights:
        # Tìm file checkpoint mới nhất
        checkpoint_files = glob.glob(os.path.join(path_weights, "medblip_large_step_*.pth"))
        if checkpoint_files:
            checkpoint_path = max(checkpoint_files, key=lambda x: int(x.split('_step_')[-1].split('.pth')[0]))
            checkpoint = torch.load(checkpoint_path, map_location=device)  # Tải checkpoint lên GPU
            model.load_state_dict(checkpoint['model_state_dict'])  # Sửa lỗi cú pháp
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Chuyển trạng thái optimizer sang GPU
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            global_step = checkpoint['global_step']
            start_epoch = checkpoint['epoch'] + 1  # Tiếp tục từ epoch tiếp theo
            print(f"Đã tải checkpoint từ bước {global_step}, epoch {checkpoint['epoch']}: {checkpoint_path}")
        else:
            print(f"Không tìm thấy checkpoint trong {path_weights}, bắt đầu từ đầu")
    else:
        print("Bắt đầu huấn luyện từ đầu")

    # Tạo dataset cho huấn luyện
    train_dataset = ImgCaptionDataset(
        df=df_train,
        path=train_dir,
        processor=processor,
        image_size=image_size,
        max_length=max_length
    )
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    # Bắt đầu huấn luyện
    model.train()
    for epoch in range(start_epoch, start_epoch + num_epochs):
        print(f"Epoch: {epoch + 1}")  # Hiển thị epoch bắt đầu từ 1 cho dễ hiểu
        # Thanh tiến trình cho batch trong epoch
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}"):
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

        # Lưu checkpoint sau mỗi epoch
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
            'epoch': epoch,
            'loss': loss.item()
        }
        checkpoint_path = os.path.join(path_weights, f"medblip_large_step_{global_step}.pth")
        
        # Lưu checkpoint mới
        torch.save(checkpoint, checkpoint_path)
        print(f"Đã lưu checkpoint sau epoch {epoch + 1}: {checkpoint_path}")
        print(f"Loss: {loss.item()}")

def predict(root_path, path_weights="/kaggle/working/"):
    """
    Hàm dự đoán
    root_path: thư mục gốc của dataset
    path_weights: đường dẫn đến file trọng số
    """
    # Tải mô hình và processor
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    # Tìm file checkpoint mới nhất
    checkpoint_files = glob.glob(os.path.join(path_weights, "medblip_large_step_*.pth"))
    if not checkpoint_files:
        raise FileNotFoundError(f"Không tìm thấy checkpoint trong {path_weights}")
    checkpoint_path = max(checkpoint_files, key=lambda x: int(x.split('_step_')[-1].split('.pth')[0]))
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    print(f"Đã tải checkpoint: {checkpoint_path}")
    
    model.eval()
    model.to("cuda")

    # # Tải dữ liệu validation
    # df_valid = os.path.join(root_path, "valid/valid/valid_captions.csv")
    # df_valid = pd.read_csv(df_valid)
    # dir_valid = os.path.join(root_path, "valid/valid")

    dir_test = os.path.join(root_path, "test/test")
    #dir_caption = os.path.join(root_path, "valid/valid/valid_captions.csv")

    test_ID = os.listdir(dir_test)
    for i in range(len(test_ID)):
        test_ID[i] = test_ID[i].replace(".jpg", "")

    def get_inferences(IDs, model, paths, max_new_tokens=200):
        """Hàm lấy kết quả dự đoán"""
        data = []
        for ID in tqdm(IDs, desc="Generating captions"):
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

    # Lấy kết quả dự đoán cho tập test
    test_results = get_inferences(test_ID, model, dir_test)
    save_df_to_csv(test_results, "/kaggle/working/run.csv")

    # # Lấy kết quả dự đoán cho tập valid
    # valid_ID = df_valid["ID"]
    # valid_results = get_inferences(valid_ID, model, dir_valid)
    # save_df_to_csv(valid_results, "/kaggle/working/valid.csv")
    # len(valid_results)

def main():
    """Hàm chính"""
    # Khởi tạo parser cho tham số dòng lệnh
    parser = argparse.ArgumentParser()

    # Tạo subparser cho các lệnh
    subparsers = parser.add_subparsers(dest='command')

    # Thêm subparser cho lệnh 'train'
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--root_path', type=str, default='/kaggle/input/if-u-know-u-know/')
    parser_train.add_argument('--batch_size', type=int, default=4)
    parser_train.add_argument('--num_epochs', type=int, default=2)  # Đặt mặc định thành 2
    parser_train.add_argument('--lr', type=float, default=1e-5)
    parser_train.add_argument('--load_weights', type=bool, default=False)
    parser_train.add_argument('--path_weights', type=str, default='/kaggle/working/')

    # Thêm subparser cho lệnh 'predict'
    parser_predict = subparsers.add_parser('predict')
    parser_predict.add_argument('--root_path', type=str, default='/kaggle/input/oggy-ds312/')
    parser_predict.add_argument('--path_weights', type=str, default='/kaggle/working/')

    # Phân tích tham số dòng lệnh
    args = parser.parse_args()

    if args.command == 'train':
        train(args.root_path, args.batch_size, args.num_epochs, args.lr, args.load_weights, args.path_weights)
    elif args.command == 'predict':
        predict(args.root_path, args.path_weights)

if __name__ == "__main__":
    main()