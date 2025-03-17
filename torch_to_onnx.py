import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import os
from typing import List, Dict, Tuple, Optional, Union

from vfm import VFM


def convert_pytorch_to_onnx(
    model: nn.Module,
    input_sample: Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]],
    onnx_path: str,
    input_names: List[str] = None,
    output_names: List[str] = None,
    dynamic_axes: Dict[str, Dict[int, str]] = None,
    opset_version: int = 14,
    export_params: bool = True,
    do_constant_folding: bool = True,
    verbose: bool = False,
    training: torch.onnx.TrainingMode = torch.onnx.TrainingMode.EVAL,
    enable_onnx_checker: bool = True,
    optimize_onnx: bool = True,
) -> str:
    """
    Convert a PyTorch model to ONNX format with comprehensive configuration options.

    Args:
        model: PyTorch model to convert
        input_sample: Sample input(s) matching the model's expected input format
        onnx_path: Output path for the ONNX model file
        input_names: Names for model inputs (default: auto-generated)
        output_names: Names for model outputs (default: auto-generated)
        dynamic_axes: Dictionary specifying dynamic axes for inputs/outputs
        opset_version: ONNX operator set version to use
        export_params: Whether to export model parameters
        do_constant_folding: Whether to execute constant folding optimization
        verbose: Whether to print detailed conversion info
        training: Convert model in training or evaluation mode
        enable_onnx_checker: Run ONNX model checker after conversion
        optimize_onnx: Optimize the ONNX model after conversion

    Returns:
        Path to the saved ONNX model
    """
    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)

    # Ensure model is in the right mode
    if training == torch.onnx.TrainingMode.EVAL:
        model.eval()
    else:
        model.train()

    # Default input/output names if not provided
    if input_names is None:
        if isinstance(input_sample, (list, tuple)):
            input_names = [f"input_{i}" for i in range(len(input_sample))]
        else:
            input_names = ["input"]

    if output_names is None:
        # Run a forward pass to determine number of outputs
        with torch.no_grad():
            outputs = model(input_sample)

        if isinstance(outputs, (list, tuple)):
            output_names = [f"output_{i}" for i in range(len(outputs))]
        else:
            output_names = ["output"]

    # Default dynamic axes if not provided but commonly useful
    if dynamic_axes is None:
        dynamic_axes = {}
        # Add batch dimension as dynamic for all inputs and outputs
        for name in input_names:
            dynamic_axes[name] = {0: "batch_size"}
        for name in output_names:
            dynamic_axes[name] = {0: "batch_size"}

    print(f"Converting model to ONNX with opset version {opset_version}...")
    print(f"Input names: {input_names}")
    print(f"Output names: {output_names}")
    print(f"Dynamic axes: {dynamic_axes}")

    with torch.inference_mode():
        torch.cuda.empty_cache()

        # Export the model
        torch.onnx.export(
            model,  # PyTorch model
            input_sample,  # Model input(s)
            onnx_path,  # Output file path
            export_params=export_params,  # Store the trained parameter weights inside the model file
            opset_version=opset_version,  # ONNX version to use (higher versions support more operators)
            do_constant_folding=do_constant_folding,  # Fold constants for optimization
            input_names=input_names,  # Names for the input nodes
            output_names=output_names,  # Names for the output nodes
            dynamic_axes=dynamic_axes,  # Variable length axes
            verbose=verbose,  # Detailed conversion information
            training=training,  # Export the model in training or inference mode
            # dynamo=True
        )

    # Verify the ONNX model
    if enable_onnx_checker:
        print("Checking ONNX model...")
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid!")

    # Optimize the ONNX model if requested
    if optimize_onnx:
        try:
            print("Optimizing ONNX model...")
            from onnxruntime.transformers import optimizer

            optimized_model = optimizer.optimize_model(
                onnx_path,
                model_type="bert",  # Can be modified depending on model type
                num_heads=12,  # Modify based on model architecture
                hidden_size=768,  # Modify based on model architecture
            )
            optimized_model.save_model_to_file(onnx_path)
            print("ONNX model optimized and saved!")
        except ImportError:
            print("onnxruntime-transformers not available - skipping optimization")
        except Exception as e:
            print(f"Error optimizing model: {e}")
            print("Continuing with unoptimized model")

    # Verify the model runs correctly
    try:
        print("Testing the ONNX model with ONNXRuntime...")
        # Create an ONNXRuntime session
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session = ort.InferenceSession(onnx_path, session_options)

        # Prepare inputs
        if isinstance(input_sample, (list, tuple)):
            ort_inputs = {
                name: input_sample[i].cpu().numpy()
                for i, name in enumerate(input_names)
            }
        else:
            ort_inputs = {input_names[0]: input_sample.cpu().numpy()}

        # Run inference
        ort_outputs = session.run(None, ort_inputs)
        print("ONNXRuntime inference successful!")

        # Optional: Compare PyTorch and ONNX outputs
        with torch.no_grad():
            torch_outputs = model(input_sample)

        if isinstance(torch_outputs, (list, tuple)):
            for i, (torch_out, ort_out) in enumerate(zip(torch_outputs, ort_outputs)):
                torch_out_np = torch_out.cpu().numpy()
                np.testing.assert_allclose(torch_out_np, ort_out, rtol=1e-3, atol=1e-5)
                print(f"Output {i}: PyTorch and ONNX Runtime results match!")
        else:
            torch_out_np = torch_outputs.cpu().numpy()
            np.testing.assert_allclose(
                torch_out_np, ort_outputs[0], rtol=1e-3, atol=1e-5
            )
            print("PyTorch and ONNX Runtime results match!")

    except Exception as e:
        print(f"Error testing ONNX model: {e}")

    print(f"ONNX model saved to: {onnx_path}")
    return onnx_path


if __name__ == "__main__":

    device = "cpu"
    dtype = torch.float32
    # Instantiate the model
    model = VFM(device=device, dtype=dtype)
    model.eval()  # Set model to evaluation mode

    all_pixel_values = torch.load(
        "/home/ubuntu/vlm-vfm-processing-pipeline/test_data/all_pixel_values.pkl",
        weights_only=True,
        map_location=device,
    )
    patch_attn_mask = torch.load(
        "/home/ubuntu/vlm-vfm-processing-pipeline/test_data/patch_attn_mask.pkl",
        weights_only=True,
        map_location=device,
    )
    tgt_sizes = torch.load(
        "/home/ubuntu/vlm-vfm-processing-pipeline/test_data/tgt_sizes.pkl",
        weights_only=True,
        map_location=device,
    )

    # Set dynamic axes for inputs/outputs
    dynamic_axes = {
        "all_pixel_values": {0: "batch_size"},
        "patch_attn_mask": {0: "batch_size"},
        "tgt_sizes": {0: "batch_size"},
        "vision_embedding": {0: "batch_size"},
    }

    convert_pytorch_to_onnx(
        model=model,
        input_sample=(all_pixel_values, patch_attn_mask, tgt_sizes),
        onnx_path="models/vfm.onnx",
        input_names=["all_pixel_values", "patch_attn_mask", "tgt_sizes"],
        output_names=["vision_embedding"],
        dynamic_axes=dynamic_axes,
        opset_version=19,  # Higher opset for newer operators
        export_params=True,  # Include model weights
        do_constant_folding=True,  # Optimize constant operations
        verbose=True,  # Show detailed conversion info
        training=torch.onnx.TrainingMode.EVAL,  # Export in inference mode
        enable_onnx_checker=True,  # Verify model structure
        optimize_onnx=False,  # Apply optimizations
    )
