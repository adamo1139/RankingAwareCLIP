import torch
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, recall_score, precision_score


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def inference_image_with_references(
    model,
    target_image,
    reference_images,
    transform_fn,
    caption: str,
    text_tokenizer,
    device='cpu',
    model_version='v1',
):
    target_image_input = transform_fn(target_image)
    reference_image_inputs = torch.stack([transform_fn(image) for image in reference_images])
    image_inputs = torch.concat([target_image_input[None, ...], reference_image_inputs])

    text_inputs = text_tokenizer(caption)
    with torch.no_grad():
        if model_version == 'v1':
            y_pred = model(
                image_inputs.to(device),
                text_inputs.to(device),
            ).ravel()
        elif model_version in ['v2', 'v3']:
            y_pred, *_ = model(
                image_inputs.to(device),
                text_inputs.to(device),
            )
            y_pred = y_pred.ravel()
        else:
            raise NotImplementedError('{model_version} not implemented')
    y_pred = y_pred.detach().cpu().numpy()
    return y_pred


def evaluate_clip_on_count(y_trues, y_preds):
    u = y_trues.reshape((1, -1)) - y_trues.reshape((-1, 1))
    v = y_preds.reshape((1, -1)) - y_preds.reshape((-1, 1))

    # k=1, ignore diag line
    ypv = v[np.tril_indices(len(y_preds), k=1)] >= 0
    ytv = u[np.tril_indices(len(y_preds), k=1)] >= 0

    pearson_corr, _ = stats.pearsonr(y_trues, y_preds)
    spearman_corr, _ = stats.spearmanr(y_trues, y_preds)
    accuracy = accuracy_score(y_true=ytv, y_pred=ypv)
    mae = np.abs(y_trues - y_preds).mean()

    return {
        'PLCC': pearson_corr,
        'SRCC': spearman_corr,
        'MAE': mae,
        'accuracy': accuracy,
    }
