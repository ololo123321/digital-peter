import os
from argparse import ArgumentParser

from utils import evaluate, clean_text


if __name__ == "__main__":
    """
    python evaluate.py \
        --predictions_dir=./predictions/valid_predictions \
        --answers_dir=./data/valid_texts
    """
    parser = ArgumentParser()
    parser.add_argument("--predictions_dir")
    parser.add_argument("--answers_dir")
    parser.add_argument("--items_to_display", type=int, default=10)
    args = parser.parse_args()

    true_texts = []
    pred_texts = []
    for file in os.listdir(args.predictions_dir):
        true_texts.append(clean_text(open(os.path.join(args.answers_dir, file)).readline()))
        pred_texts.append(open(os.path.join(args.predictions_dir, file)).readline())

    evaluate(true_texts=true_texts, pred_texts=pred_texts, top_k=args.items_to_display)
