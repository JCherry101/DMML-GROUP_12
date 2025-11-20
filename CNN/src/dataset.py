import os
import random
from typing import Dict, List, Optional, Tuple
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

class CarDataset(Dataset):
    """Dataset for car images with manufacturer, body type, color, and model labels.

    Folder structure: root/Brand/Model/Year/Color/*.jpg
    File name pattern (example): Bentley$$Arnage$$2003$$Silver$$10_1$$21$$image_0.jpg
    Where token[4] = Genmodel_ID used to map to Bodytype via CSV.

    We ignore year for classification per requirements. Labels: manufacturer (brand), body type, exterior color, and model name.
    Split performed at (brand, model, year, color) group to allocate 80% images to train, 20% to test.
    """

    def __init__(
        self,
        root_dir: str,
        csv_path: str,
        split: str = "train",
        seed: int = 42,
        image_size: int = 224,
        maker_to_idx: Optional[Dict[str, int]] = None,
        body_to_idx: Optional[Dict[str, int]] = None,
        color_to_idx: Optional[Dict[str, int]] = None,
        model_to_idx: Optional[Dict[str, int]] = None,
    ):
        assert split in {"train", "test"}
        self.root_dir = root_dir
        self.csv_path = csv_path
        self.split = split
        self.seed = seed
        random.seed(seed)

        df = pd.read_csv(self.csv_path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        if not {"Genmodel_ID", "Bodytype", "Maker"}.issubset(df.columns):
            raise ValueError("Expected columns Genmodel_ID, Bodytype, Maker not all found in CSV")
        df = df.dropna(subset=["Bodytype", "Maker", "Genmodel_ID"])
        genmodel_to_body = df.groupby("Genmodel_ID")["Bodytype"].first().to_dict()
        genmodel_to_maker = df.groupby("Genmodel_ID")["Maker"].first().to_dict()

        self.genmodel_to_body = genmodel_to_body
        self.genmodel_to_maker = genmodel_to_maker

        all_samples: List[Tuple[str, str, str, str, str]] = []

        for brand in sorted(os.listdir(self.root_dir)):
            brand_path = os.path.join(self.root_dir, brand)
            if not os.path.isdir(brand_path):
                continue
            for model in sorted(os.listdir(brand_path)):
                model_path = os.path.join(brand_path, model)
                if not os.path.isdir(model_path):
                    continue
                for year in sorted(os.listdir(model_path)):
                    year_path = os.path.join(model_path, year)
                    if not os.path.isdir(year_path):
                        continue
                    for color in sorted(os.listdir(year_path)):
                        color_path = os.path.join(year_path, color)
                        if not os.path.isdir(color_path):
                            continue
                        group_files = []
                        for fname in os.listdir(color_path):
                            fpath = os.path.join(color_path, fname)
                            ext = os.path.splitext(fname)[1].lower()
                            if ext not in IMAGE_EXTENSIONS:
                                continue
                            tokens = fname.split("$$")
                            if len(tokens) < 6:
                                continue
                            genmodel_id = tokens[4]
                            maker = brand
                            maker = self.genmodel_to_maker.get(genmodel_id, maker)
                            bodytype = self.genmodel_to_body.get(genmodel_id)
                            if bodytype is None:
                                continue
                            color_name = color
                            model_key = f"{maker}::{model}"
                            group_files.append((fpath, maker, bodytype, color_name, model_key))
                        if not group_files:
                            continue
                        random.shuffle(group_files)
                        split_idx = int(0.8 * len(group_files))
                        train_group = group_files[:split_idx]
                        test_group = group_files[split_idx:]
                        if self.split == "train":
                            all_samples.extend(train_group)
                        else:
                            all_samples.extend(test_group)

        if maker_to_idx is None:
            makers = sorted({m for _, m, _, _, _ in all_samples})
            self.maker_to_idx = {m: i for i, m in enumerate(makers)}
        else:
            self.maker_to_idx = dict(maker_to_idx)

        if body_to_idx is None:
            bodytypes = sorted({b for _, _, b, _, _ in all_samples})
            self.body_to_idx = {b: i for i, b in enumerate(bodytypes)}
        else:
            self.body_to_idx = dict(body_to_idx)

        if color_to_idx is None:
            colors = sorted({c for _, _, _, c, _ in all_samples})
            self.color_to_idx = {c: i for i, c in enumerate(colors)}
        else:
            self.color_to_idx = dict(color_to_idx)

        if model_to_idx is None:
            models = sorted({m for _, _, _, _, m in all_samples})
            self.model_to_idx = {m: i for i, m in enumerate(models)}
        else:
            self.model_to_idx = dict(model_to_idx)

        filtered_samples: List[Tuple[str, int, int, int, int]] = []
        for path, maker_name, body_name, color_name, model_name in all_samples:
            maker_idx = self.maker_to_idx.get(maker_name)
            body_idx = self.body_to_idx.get(body_name)
            color_idx = self.color_to_idx.get(color_name)
            model_idx = self.model_to_idx.get(model_name)
            if None in {maker_idx, body_idx, color_idx, model_idx}:
                continue
            filtered_samples.append((path, maker_idx, body_idx, color_idx, model_idx))

        self.samples = filtered_samples

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, maker_idx, body_idx, color_idx, model_idx = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            tensor = self.transform(img)
        return (
            tensor,
            torch.tensor(maker_idx, dtype=torch.long),
            torch.tensor(body_idx, dtype=torch.long),
            torch.tensor(color_idx, dtype=torch.long),
            torch.tensor(model_idx, dtype=torch.long),
        )

    @property
    def num_makers(self) -> int:
        return len(self.maker_to_idx)

    @property
    def num_bodytypes(self) -> int:
        return len(self.body_to_idx)

    @property
    def num_colors(self) -> int:
        return len(self.color_to_idx)

    @property
    def num_models(self) -> int:
        return len(self.model_to_idx)
if __name__ == "__main__":
    dataset = CarDataset(
        root_dir=os.path.join(os.path.dirname(__file__), "..", "resized_DVM"),
        csv_path=os.path.join(os.path.dirname(__file__), "..", "Adv_table.csv"),
        split="train",
    )
    print("Samples:", len(dataset))
    print(
        "Manufacturers:", dataset.num_makers,
        "Bodytypes:", dataset.num_bodytypes,
        "Colors:", dataset.num_colors,
        "Models:", dataset.num_models,
    )
    if len(dataset):
        x, m, b, c, mo = dataset[0]
        print(
            "One sample tensor shape:", x.shape,
            "maker idx:", m.item(),
            "body idx:", b.item(),
            "color idx:", c.item(),
            "model idx:", mo.item(),
        )
