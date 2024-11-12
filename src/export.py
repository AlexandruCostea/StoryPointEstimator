import argparse

from models import get_model_and_tokenizer


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description='Model export details')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to export')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory for exported model')
    parser.add_argument('--model_name', type=str, default='storypoint_estimator', help='Model name')
    return parser.parse_args()


def export_model(args):

    model, tokenizer = get_model_and_tokenizer(args.checkpoint)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == '__main__':
    
    args = parse_args()
    export_model(args)