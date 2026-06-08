import pandas as pd
from torch.utils.data import DataLoader
from sequence_dataset import YieldForecastDataset

df = pd.read_csv("productivity_data_v2.csv")

crop_name = 'Пшеница яровая'

train_ds = YieldForecastDataset(
    df=df,
    crop_name=crop_name,
    split="train",
    val_size=0.2,
    seed=42,
    step_days=7,
    windows=(7, 30, 90, 180, 365),
)

val_ds = YieldForecastDataset(
    df=df,
    crop_name=crop_name,
    split="val",
    val_size=0.2,
    seed=42,
    step_days=7,
    windows=(7, 30, 90, 180, 365),
)

train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=0,
)

batch = next(iter(train_loader))

print(batch["x"].shape)
print(batch["mask"].shape)
print(batch["y"].shape)
print(train_ds.feature_names[:20])
