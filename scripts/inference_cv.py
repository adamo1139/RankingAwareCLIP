"""eval.py
Evaluate the CLIP-Adapter for all task.
"""
import typing as t
import json
import os

import numpy as np
import pandas as pd
import pydantic
from pydantic_argparse import ArgumentParser
from pathlib import Path
from PIL import Image, ImageFile
import cv2
import matplotlib.pyplot as plt

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

    # Skip create_model_and_transforms() which fails due to model registry issues
    # Instead, directly create the transforms pipeline based on the Colab demo
    transform = tv.transforms.Compose([
        tv.transforms.Resize((args.image_size, args.image_size)),
        # We'll handle the convert('RGB') in the dataset's __getitem__ method
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),  # hardcode value from open_clip. (Checked for ViT-B/16, ConvNext-L)
    ])
    # Use the default tokenizer instead of one dependent on model name
    tokenizer = open_clip.tokenizer.SimpleTokenizer()

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

    # Load the dataset
    df = pd.read_csv(args.data_to_inference, sep='\t')
    unique_categories = df.cname.unique().tolist()
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each category
    results = {}
    result_stats = []
    
    for category in unique_categories:
        print(f"\nProcessing category: {category}")
        df_subset = df[df.cname == category].reset_index(drop=True)
        
        # Get file paths and true values
        image_paths = df_subset['filepath'].tolist()
        true_values = df_subset['value'].values.astype(float)
        
        # Create prompt for this category - use custom if provided, otherwise default
        if category in args.custom_prompts:
            prompt = args.custom_prompts[category]
        else:
            prompt = f"Rank the numbers of {category}."
        print(f"Using prompt: '{prompt}'")
        
        # Process images in batches to avoid OOM errors
        num_images = len(image_paths)
        batch_size = args.batch_size
        all_predictions = []
        
        print(f"Processing {num_images} images in batches of {batch_size}...")
        
        # Process in batches
        for i in range(0, num_images, batch_size):
            batch_end = min(i + batch_size, num_images)
            batch_paths = image_paths[i:batch_end]
            batch_size_actual = len(batch_paths)
            
            # Tokenize prompt for each image in this batch
            batch_tokens = torch.stack([tokenizer(prompt) for _ in range(batch_size_actual)])
            
            # Load and process this batch of images
            batch_images = []
            for path in batch_paths:
                image = Image.open(path).convert('RGB')
                batch_images.append(transform(image))
            
            batch_tensor = torch.stack(batch_images)
            
            # Run inference on this batch
            with torch.no_grad():
                batch_output = model(
                    batch_tensor.to(device),
                    batch_tokens.to(device).squeeze(1),
                )
            
            # Get and store predictions for this batch
            batch_predictions = batch_output['adapter_logits'].cpu().numpy().ravel()
            all_predictions.extend(batch_predictions)
        
        # Convert list to numpy array
        predictions = np.array(all_predictions)
        
        # Store results
        results[category] = {
            'image_paths': image_paths,
            'y_trues': true_values.tolist(),
            'y_preds': predictions.tolist(),
        }
        
        # Create visualization if enabled
        if args.create_visualizations:
            # Determine visualization directory
            vis_dir = args.visualization_dir if args.visualization_dir else output_dir
            os.makedirs(vis_dir, exist_ok=True)
            
            vis_path = f"{vis_dir}/{category}_ranking.png"
            render_figures_with_order(
                image_paths=image_paths,
                values=predictions,
                sup_values=true_values,
                figure_title=f"Ranking by {prompt}",
                output_path=vis_path,
                show_value=True,
            )
            print(f"Ranking visualization saved to {vis_path}")
            
        # Create CSV with ranking information if enabled
        if args.create_csv:
            # Determine CSV directory
            csv_dir = args.csv_dir if args.csv_dir else output_dir
            os.makedirs(csv_dir, exist_ok=True)
            
            csv_path = f"{csv_dir}/{category}_ranking.csv"
            save_to_csv(
                image_paths=image_paths,
                prediction_values=predictions,
                true_values=true_values,
                output_path=csv_path,
            )
        
        # Compute statistics
        yp = np.asarray(predictions)
        if args.is_ordinal_regression:
            yp = np.round(yp)
            
        result_dict = evaluate_clip_on_count(
            y_trues=true_values,
            y_preds=yp,
        )
        result_dict['task_name'] = 'object-count'
        result_dict['sub-category'] = category
        result_stats.append(result_dict)
    
    # Save numerical results
    if args.output_path is not None:
        # Avoid the double .json issue
        result_path = args.output_path
        if result_path.endswith('.json'):
            result_path = result_path[:-5]  # Remove .json
            
        with open(f'{result_path}.json', 'w') as f:
            json.dump(result_stats, f, indent=4)
        print(f"Numerical results saved to {result_path}.json")
    
    # Print statistics
    print("\nResults Summary:")
    for metric_name in ['PLCC', 'SRCC', 'MAE']:
        classwise_result = [r[metric_name] for r in result_stats]
        print(f'Mean {metric_name}: {np.mean(classwise_result):.3f}')
        
    print("\nDone! Check the output directory for visualizations and result files.")


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
        description='Base filename/path for output files',
        default='./tmp/result-stats',
    )
    image_size: int = pydantic.Field(
        description='Image size of the input (ConvNext-L=320)',
        default=320,
    )
    batch_size: int = pydantic.Field(
        description='Batch size for processing images to avoid OOM errors',
        default=100,
    )
    is_ordinal_regression: bool = pydantic.Field(
        description='If the target is in bins, do the rounding to prediction (e.g. HCI / AgeEst)',
        default=False,
    )
    custom_prompts: t.Dict[str, str] = pydantic.Field(
        description='Custom prompts for each category in JSON format, e.g., \'{"apple": "Count the apples", "horse": "Count the horses"}\'',
        default_factory=dict,
    )
    create_visualizations: bool = pydantic.Field(
        description='Create visualizations of the ranked images',
        default=False,
    )
    visualization_dir: t.Optional[str] = pydantic.Field(
        description='Directory to save visualizations (defaults to same directory as output_path)',
        default=None,
    )
    create_csv: bool = pydantic.Field(
        description='Create CSV files with ranking information for each category',
        default=True,
    )
    csv_dir: t.Optional[str] = pydantic.Field(
        description='Directory to save CSV files (defaults to same directory as output_path)',
        default=None,
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


def render_figures_with_order(
    image_paths,
    values,
    sup_values=None,
    figure_title=None,
    output_path=None,
    show_value=True,
):
    """Render figures sorted by their prediction values.
    
    Args:
        image_paths: List of image file paths to display
        values: List of prediction values for each image
        sup_values: Optional list of supplementary values (e.g., ground truth)
        figure_title: Optional title for the figure
        output_path: Optional path to save the figure
        show_value: Whether to display values on the images
    """
    # Load images and pair with values
    images = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            images.append(img)
        else:
            print(f"Warning: Could not load image {path}")
    
    if sup_values is None:
        compact = list(zip(images, values, image_paths))
    else:
        compact = list(zip(images, values, sup_values, image_paths))
    
    # Sort by prediction values (descending)
    sorted_data = sorted(compact, key=lambda x: x[1], reverse=True)
    
    # Create figure
    plt.figure(figsize=(16, 12))
    if figure_title:
        plt.suptitle(figure_title, fontsize=16)
    
    # Plot each image
    for i, data in enumerate(sorted_data):
        if sup_values is None:
            image, value, path = data
        else:
            image, value, sup_value, path = data
        
        plt.subplot(1, len(images), i + 1)
        plt.imshow(image)
        plt.axis('off')
        
        if show_value:
            if sup_values is None:
                v = f'score={value:.4f}'
            else:
                v = f'true/pred={sup_value:.4f}/{value:.4f}'
            plt.title(v)
    
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Figure saved to {output_path}")
    
    return plt.gcf()  # Return the figure


def save_to_csv(
    image_paths,
    prediction_values,
    true_values=None,
    output_path=None,
):
    """Save ranking results to a CSV file.
    
    Args:
        image_paths: List of paths to the images
        prediction_values: Model prediction values for each image
        true_values: Optional ground truth values for each image
        output_path: Path to save the CSV file
    """
    # Prepare data for ranking
    if true_values is not None:
        data = list(zip(image_paths, prediction_values, true_values))
        columns = ['image_path', 'prediction_score', 'true_value']
    else:
        data = list(zip(image_paths, prediction_values))
        columns = ['image_path', 'prediction_score']
    
    # Sort by prediction values (descending)
    ranked_data = sorted(data, key=lambda x: x[1], reverse=True)
    
    # Add ranking column
    ranked_data_with_rank = []
    for i, item in enumerate(ranked_data):
        rank = i + 1  # Start ranking from 1
        ranked_data_with_rank.append((*item, rank))
    
    # Add rank column name
    columns.append('rank')
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(ranked_data_with_rank, columns=columns)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Ranking CSV saved to {output_path}")
    
    return df


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
    return results


if __name__ == '__main__':
    main()
