import os
import torch
import numpy as np
import onnxruntime as ort
import time
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vfm import VFM
import matplotlib.pyplot as plt


def test_onnx_model(
    onnx_model_path, input_data_paths, device="cpu", compare_with_pytorch=True
):
    """
    Test an ONNX model with provided input data and compare with PyTorch output.

    Args:
        onnx_model_path: Path to the ONNX model
        input_data_paths: Dictionary mapping input names to file paths
        device: Device to run PyTorch model on
        compare_with_pytorch: Whether to compare with PyTorch model output
    """
    print(f"Testing ONNX model: {onnx_model_path}")

    # 1. Load input data
    print("Loading input data...")
    inputs = {}
    for name, path in input_data_paths.items():
        if os.path.exists(path):
            inputs[name] = torch.load(path, weights_only=True, map_location=device)
            print(
                f"  Loaded {name}: shape={inputs[name].shape}, dtype={inputs[name].dtype}"
            )
        else:
            raise FileNotFoundError(f"Input file not found: {path}")

    # 2. Create ONNX Runtime session
    print("Creating ONNX Runtime session...")
    try:
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session = ort.InferenceSession(
            onnx_model_path,
            providers=["CPUExecutionProvider"],
            sess_options=session_options,
        )
        print("  Session created successfully")
    except Exception as e:
        print(f"Error creating ONNX session: {e}")
        raise

    # 3. Prepare inputs for ONNX Runtime
    print("Preparing ONNX inputs...")
    ort_inputs = {}
    for input_meta in session.get_inputs():
        input_name = input_meta.name
        if input_name in inputs:
            # Convert PyTorch tensor to numpy
            ort_inputs[input_name] = inputs[input_name].numpy()
            print(f"  Prepared {input_name}: shape={ort_inputs[input_name].shape}")
        else:
            raise ValueError(f"Required input '{input_name}' not provided")

    # 4. Run inference with ONNX Runtime
    print("Running ONNX inference...")
    try:
        start_time = time.time()
        ort_outputs = session.run(None, ort_inputs)
        onnx_inference_time = time.time() - start_time
        print(f"  ONNX inference completed in {onnx_inference_time:.4f} seconds")

        # Display output information
        for i, output_meta in enumerate(session.get_outputs()):
            output_name = output_meta.name
            output = ort_outputs[i]
            print(
                f"  Output '{output_name}': shape={output.shape}, dtype={output.dtype}"
            )
            print(
                f"  Output stats - min: {output.min():.6f}, max: {output.max():.6f}, mean: {output.mean():.6f}"
            )
    except Exception as e:
        print(f"Error during ONNX inference: {e}")
        raise

    # 5. Optional: Compare with PyTorch model output
    if compare_with_pytorch:
        print("Comparing with PyTorch model output...")
        try:
            # Instantiate PyTorch model
            pytorch_model = VFM(device=device, dtype=torch.float32)
            pytorch_model.eval()

            # Run PyTorch inference
            with torch.no_grad():
                start_time = time.time()
                if "tgt_sizes" in inputs:
                    pytorch_outputs = pytorch_model(
                        inputs["all_pixel_values"],
                        inputs["patch_attn_mask"],
                        inputs["tgt_sizes"],
                    )
                else:
                    pytorch_outputs = pytorch_model(
                        inputs["all_pixel_values"], inputs["patch_attn_mask"]
                    )
                pytorch_inference_time = time.time() - start_time

            print(
                f"  PyTorch inference completed in {pytorch_inference_time:.4f} seconds"
            )

            # Convert PyTorch output to numpy for comparison
            if isinstance(pytorch_outputs, tuple):
                pytorch_output_np = pytorch_outputs[0].numpy()
            else:
                pytorch_output_np = pytorch_outputs.numpy()

            # Compare outputs
            ort_output = ort_outputs[0]  # Assuming single output

            # Calculate difference
            abs_diff = np.abs(pytorch_output_np - ort_output)
            max_diff = np.max(abs_diff)
            mean_diff = np.mean(abs_diff)

            print(
                f"  Output comparison - max difference: {max_diff:.6f}, mean difference: {mean_diff:.6f}"
            )

            # Plot histogram of differences
            plt.figure(figsize=(10, 6))
            plt.hist(abs_diff.flatten(), bins=50)
            plt.title(
                "Histogram of Absolute Differences Between PyTorch and ONNX Outputs"
            )
            plt.xlabel("Absolute Difference")
            plt.ylabel("Count")
            plt.yscale("log")
            plt.savefig("onnx_pytorch_diff_histogram.png")
            print("  Saved difference histogram to 'onnx_pytorch_diff_histogram.png'")

            # Speed comparison
            print(
                f"  Speed comparison: ONNX is {pytorch_inference_time/onnx_inference_time:.2f}x {'faster' if onnx_inference_time < pytorch_inference_time else 'slower'} than PyTorch"
            )

        except Exception as e:
            print(f"Error during PyTorch comparison: {e}")

    return ort_outputs


if __name__ == "__main__":
    # Paths to model and data
    onnx_model_path = (
        "models/vfm_fix_outofrange_fp16.onnx"
    )

    input_data_paths = {
        "all_pixel_values": "/home/ubuntu/vlm-vfm-processing-pipeline/test_data/all_pixel_values.pkl",
        "patch_attn_mask": "/home/ubuntu/vlm-vfm-processing-pipeline/test_data/patch_attn_mask.pkl",
    }

    # Test the model
    outputs = test_onnx_model(
        onnx_model_path=onnx_model_path,
        input_data_paths=input_data_paths,
        device="cpu",
        compare_with_pytorch=True,  # Set to False if you don't want to compare with PyTorch
    )

    # Additional visualization if needed
    # Visualize first element of the output
    first_output = outputs[0][0]  # First batch item

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(first_output[:64, :64], cmap="viridis")
    plt.colorbar()
    plt.title("Output Visualization (First 64x64)")

    plt.subplot(1, 2, 2)
    plt.hist(first_output.flatten(), bins=50)
    plt.title("Output Distribution")
    plt.savefig("onnx_output_visualization.png")
    print("Saved output visualization to 'onnx_output_visualization.png'")

    print("ONNX model testing completed successfully!")
