import argparse

from models import get_model_and_tokenizer
from data_loader import DFGenerator, StoryPointDataset

import numpy as np
import torch
import onnx
import onnxruntime


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

    x, _ = DFGenerator().create_dataframes()
    x = x.sample(1)
    dataset = StoryPointDataset(x, tokenizer)
    sample_data = dataset[0]

    dummy_input = dataset[0]['input_ids'].unsqueeze(0)
    attention_mask = dataset[0]['attention_mask'].unsqueeze(0)


    model.eval()
    with torch.no_grad():
        output = model(dummy_input, attention_mask=attention_mask)

    torch.onnx.export(model,
                      (dummy_input, attention_mask),
                      f'{args.output_dir}/{args.model_name}.onnx',
                      opset_version=14,
                      do_constant_folding=True,
                      input_names=['input_ids', 'attention_mask'],
                      output_names=['output'],
                      dynamic_axes={
                            'input_ids': {0: 'batch_size'},
                            'attention_mask': {0: 'batch_size'},
                            'output': {0: 'batch_size'}
                        })
    

    onnx_model = onnx.load(f'{args.output_dir}/{args.model_name}.onnx')
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(f'{args.output_dir}/{args.model_name}.onnx', providers=['CPUExecutionProvider'])

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    output_tensor = output.logits if hasattr(output, 'logits') else output.last_hidden_state

    ort_inputs = {
        ort_session.get_inputs()[0].name: to_numpy(dummy_input),
        ort_session.get_inputs()[1].name: to_numpy(attention_mask)
    }
    ort_outs = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(to_numpy(output_tensor), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been validated successfully!")


if __name__ == '__main__':
    
    args = parse_args()
    export_model(args)