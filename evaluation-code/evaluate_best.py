
import os
import json
import argparse
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report

import data_preprocess_ucihar
import network_ucihar as net

parser = argparse.ArgumentParser()
parser.add_argument("--now_model_name", type=str, default="GILE")
parser.add_argument("--target_domain", type=str, default="0")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_epoch", type=int, default=150)
parser.add_argument("--dataset", type=str, default="ucihar")
parser.add_argument("--n_feature", type=int, default=9)
parser.add_argument("--len_sw", type=int, default=128)
parser.add_argument("--n_class", type=int, default=6)
parser.add_argument("--d_AE", type=int, default=50)
parser.add_argument("--sigma", type=float, default=1)
parser.add_argument("--weight_mmd", type=float, default=1.0)
parser.add_argument("-n_domains", type=int, default=5)
parser.add_argument("-n_target_domains", type=int, default=1)
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument("--x-dim", type=int, default=1152)
parser.add_argument("--aux_loss_multiplier_y", type=float, default=1000.0)
parser.add_argument("--aux_loss_multiplier_d", type=float, default=1000.0)
parser.add_argument("--beta_d", type=float, default=5.0)
parser.add_argument("--beta_x", type=float, default=0.0)
parser.add_argument("--beta_y", type=float, default=5.0)
parser.add_argument("--weight_true", type=float, default=1000.0)
parser.add_argument("--weight_false", type=float, default=1000.0)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

torch.manual_seed(10)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.device = device

_, target_loader = data_preprocess_ucihar.prep_domains_ucihar(
    args,
    SLIDING_WINDOW_LEN=args.len_sw,
    SLIDING_WINDOW_STEP=int(0.5 * args.len_sw)
)

model = net.load_model(args).to(device)
state = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(state)
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for xs, ys, ds in target_loader:
        xs = xs.to(device)
        outputs = model.classifier(xs)
        activity_onehot = outputs[1]
        predictions = activity_onehot.argmax(dim=1)

        y_true.extend(ys.cpu().numpy().astype(int).tolist())
        y_pred.extend(predictions.cpu().numpy().astype(int).tolist())

metrics = {
    "dataset": "UCI HAR",
    "target_domain": int(args.target_domain),
    "checkpoint": args.checkpoint,
    "n_samples": len(y_true),
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    "classification_report": classification_report(
        y_true, y_pred, zero_division=0, output_dict=True
    ),
    "protocol_note": (
        "Checkpoint selected by the original implementation's target-domain "
        "accuracy during training. This uses target labels for model selection "
        "and must not be interpreted as an unbiased generalisation estimate."
    )
}

with open(args.output, "w") as f:
    json.dump(metrics, f, indent=2)

print(json.dumps({
    "dataset": metrics["dataset"],
    "target_domain": metrics["target_domain"],
    "n_samples": metrics["n_samples"],
    "accuracy": metrics["accuracy"],
    "macro_f1": metrics["macro_f1"],
    "weighted_f1": metrics["weighted_f1"],
    "metrics_file": args.output
}, indent=2))
