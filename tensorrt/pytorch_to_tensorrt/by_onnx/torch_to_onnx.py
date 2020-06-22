import torch
import argparse


def main():
    parser = argparse.ArgumentParser(description = "Torch convert to onnx")
    parser.add_argument("--torch_model", "-torch", required = True, type=str, help = "torch model path")
    parser.add_argument("--onnx_model", "-onnx", required = True, type=str, help = "onnx model path")
    parser.add_argument("--input_shape", "-in", required = True, type=int, nargs = "+", help = "input shape contain batch")
    args = parser.parse_args()

    dummy_input = torch.randn(args.input_shape, device='cuda')

    # load model
    model = torch.load(args.torch_model)

    # inference mode sigin
    model.eval()

    torch.onnx.export(
        model, 
        dummy_input, 
        args.onnx_model, 
        verbose=True)

if __name__ == "__main__":
    main()