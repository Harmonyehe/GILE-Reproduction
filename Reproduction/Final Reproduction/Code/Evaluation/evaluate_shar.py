
import os
import json
import argparse
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import data_preprocess_shar as data_module
import network_shar as network_module


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--target_domain", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    parser.add_argument("--now_model_name", default="GILE")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epoch", type=int, default=100)
    parser.add_argument("--dataset", default="shar")

    parser.add_argument("--n_feature", type=int, default=3)
    parser.add_argument("--len_sw", type=int, default=151)
    parser.add_argument("--n_class", type=int, default=17)
    parser.add_argument("--d_AE", type=int, default=50)
    parser.add_argument("--sigma", type=float, default=1)
    parser.add_argument("--weight_mmd", type=float, default=1)

    parser.add_argument("--n_domains", type=int, default=6)
    parser.add_argument("--n_target_domains", type=int, default=1)
    parser.add_argument("--beta_d", type=float, default=1)
    parser.add_argument("--beta_x", type=float, default=1)
    parser.add_argument("--beta_y", type=float, default=1)

    parser.add_argument("--x-dim", type=int, default=1152)
    parser.add_argument("--aux_loss_multiplier", type=float, default=1000)
    parser.add_argument("--aux_loss_multiplier_y", type=float, default=1000)
    parser.add_argument("--aux_loss_multiplier_d", type=float, default=1000)
    parser.add_argument("--weight_true", type=float, default=1000)
    parser.add_argument("--weight_false", type=float, default=1000)

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    args.device = device

    # Load the target domain.
    _, target_loader = data_module.prep_domains_shar(
        args,
        SLIDING_WINDOW_LEN=args.len_sw,
        SLIDING_WINDOW_STEP=int(0.5 * args.len_sw),
    )

    # Reconstruct the trained model.
    model = network_module.load_model(args).to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device
    )

    # Support either a raw state_dict or a wrapped checkpoint.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in target_loader:
            # The repository's data loaders provide features and labels.
            x = batch[0].to(device)
            y = batch[1]

            outputs = model.classifier(x)
            logits = outputs[1] if isinstance(outputs, (tuple, list)) else outputs

            predictions = logits.argmax(dim=1)

            all_predictions.extend(
                predictions.detach().cpu().numpy().astype(int).tolist()
            )
            all_labels.extend(
                y.detach().cpu().numpy().astype(int).reshape(-1).tolist()
            )

    if not all_labels:
        raise RuntimeError(
            "The target loader produced no labels. "
            "Check the preprocessing function's return values."
        )

    labels = np.asarray(all_labels)
    predictions = np.asarray(all_predictions)

    result = {
        "dataset": args.dataset,
        "target_domain": args.target_domain,
        "checkpoint": os.path.abspath(args.checkpoint),
        "num_samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "classification_report": classification_report(
            labels,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            labels, predictions
        ).tolist(),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps({
        "dataset": result["dataset"],
        "target_domain": result["target_domain"],
        "num_samples": result["num_samples"],
        "accuracy": result["accuracy"],
        "macro_f1": result["macro_f1"],
        "weighted_f1": result["weighted_f1"],
        "output": os.path.abspath(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
