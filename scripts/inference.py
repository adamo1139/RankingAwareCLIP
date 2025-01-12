"""eval.py
Evaluate the CLIP-Adapter for all task.
"""
import typing as t
import json

import numpy as np
import pandas as pd
import pydantic
from pydantic_argparse import ArgumentParser
from pathlib import Path
from PIL import Image, ImageFile

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
from rich import print
from rich.progress import track

import open_clip
from open_clip.clip_model_adapter import L2RCLIP

from learn_to_compare.utils import evaluate_clip_on_count


ImageFile.LOAD_TRUNCATED_IMAGES = True  # AVA dataset seems to have issued images


def main():
    parser = ArgumentParser(model=Arguments)
    args = parser.parse_typed_args()
    print(args)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'{device=}')

    _, _, transform = open_clip.create_model_and_transforms(
        model_name=args.base_model_name,
    )

    # No crop, do resize.
    transform = tv.transforms.Compose([
        tv.transforms.Resize((args.image_size, args.image_size)),
        transform.transforms[2],  # to RGB
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),  # hardcode value from open_clip. (Checked for ViT-B/16, ConvNext-L)
    ])
    tokenizer = open_clip.get_tokenizer(args.base_model_name)

    # Load model
    with open(args.config_path, 'r') as f:
        model_cfg = json.load(f)
    print(model_cfg)
    model = L2RCLIP(
        **model_cfg,
        model_version='clip-adapter-v2',
    )

    checkpoint_path = args.checkpoint_path
    print(f'Use checkpoint: {checkpoint_path}')
    imcompetible_keys = open_clip.load_checkpoint(
        model,
        str(checkpoint_path),
    )
    print(f'{imcompetible_keys=}')
    model.eval()
    model.to(device)

    results = inference_dataframe_fmt(
        model=model,
        dataframe_path=args.data_to_inference,
        image_transform=transform,
        tokenizer=tokenizer,
        device=device,
        batch_size=64,
    )

    result_stats = run_task_stats(
        task_name='object-count',
        task_results=results,
        is_ordinal_regression=args.is_ordinal_regression,
    )
    print(result_stats)
    if args.output_path is not None:
        with open(f'{args.output_path}.json', 'w') as f:
            json.dump(result_stats, f, indent=4)


class Arguments(pydantic.BaseModel):
    base_model_name: str = pydantic.Field(
        description='BaseModel Name (For loading configuration)',
        default='convnext_large_d_320-adapter',
    )
    config_path: Path = pydantic.Field(
        description='Path to the model configuration',
    )
    data_to_inference: Path = pydantic.Field(
        description='Path to the data to inference',
    )
    checkpoint_path: t.Optional[str] = pydantic.Field(
        description='Checkpoint name in the model directory',
    )
    output_path: str = pydantic.Field(
        description='Filename of the output.',
        default='./tmp/result-stats.json',
    )
    image_size: int = pydantic.Field(
        description='Image size of the input (ConvNext-L=320)',
        default=320,
    )
    is_ordinal_regression: bool = pydantic.Field(
        description='If the target is in bins, do the rounding to prediction (e.g. HCI / AgeEst)',
        default=False,
    )


def inference_dataframe_fmt(
    model,
    dataframe_path: str,
    image_transform: t.Callable,
    tokenizer: t.Callable,
    device: str,
    batch_size: int = 64,
) -> dict:
    df = pd.read_csv(dataframe_path, sep='\t')
    unique_cnames = df.cname.unique().tolist()  # category-name / property / ...

    eval_results = {}
    for cname in unique_cnames:
        df_subset = df[df.cname == cname]
        df_subset.reset_index(drop=True, inplace=True)

        inference_dataset = InferenceDataFrame(
            dataframe=df_subset,
            image_transform=image_transform,
            tokenizer=tokenizer,
        )
        dataloader = DataLoader(
            inference_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4,
        )
        y_trues, y_preds = [], []
        for image_inputs, text_inputs, y_trues_ in track(dataloader, description=f'{cname}...'):
            text_inputs = text_inputs.squeeze(1)  # [N, 1, M] -> [N, M]

            with torch.no_grad():
                y_preds_ = model(
                    image_inputs.to(device),
                    text_inputs.to(device),
                )  # format: dict

            adapter_logits = y_preds_['adapter_logits'].cpu().numpy().ravel()
            y_trues_ = y_trues_.cpu().numpy().astype(float)

            y_trues.append(y_trues_)
            y_preds.append(adapter_logits.astype(float))
        y_trues = np.concatenate(y_trues)
        y_preds = np.concatenate(y_preds)
        eval_results[cname] = {
            'image_ids': '',
            'cname': cname,
            'y_trues': y_trues.tolist(),
            'y_preds': y_preds.tolist(),
        }
    return eval_results


class InferenceDataFrame(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_transform: t.Callable,
        tokenizer: t.Callable,
        caption: t.Optional[str] = None,
    ):
        self.df = dataframe
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.caption = caption

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_path = row['filepath']
        image = Image.open(image_path).convert('RGB')
        image_inputs = self.image_transform(image)

        caption = row['title'] if self.caption is None else self.caption
        text_inputs = self.tokenizer(caption)

        value = row['value']
        return image_inputs, text_inputs, float(value)


def run_task_stats(
    task_name: str,
    task_results: dict,
    is_ordinal_regression: bool = False,
) -> t.Sequence[dict]:
    results = []
    # Run by grouped (sub-class) images statistics
    for sub_category, task_result in task_results.items():
        yp = np.asarray(task_result['y_preds'])
        if is_ordinal_regression:
            yp = np.round(yp)
        result_dict = evaluate_clip_on_count(
            y_trues=np.asarray(task_result['y_trues']),
            y_preds=yp,
        )
        result_dict['task_name'] = task_name
        result_dict['sub-category'] = sub_category
        results.append(result_dict)
    # Run by images
    result_dict = evaluate_clip_on_count(
        y_trues=np.concatenate([tr['y_trues'] for tr in task_results.values()]),
        y_preds=np.concatenate([
            np.round(tr['y_preds']) if is_ordinal_regression else tr['y_preds']
            for tr in task_results.values()
        ]),
    )
    result_dict['task_name'] = task_name
    result_dict['sub-category'] = '_all_'

    results.append(result_dict)
    return results


if __name__ == '__main__':
    main()
