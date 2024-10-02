"""
The following is a simple example algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./save.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""
from pathlib import Path
from glob import glob
import SimpleITK
import SimpleITK as sitk
import numpy as np
import gc
import os
from torch import nn
import torch

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from batchgenerators.utilities.file_and_folder_operations import join
import scipy.ndimage as ndi
from skimage.morphology import binary_closing, ball
from skimage.measure import label, regionprops
from torch.amp import autocast
import torch.nn.functional as F
import time

print('All required modules are loaded!!!')

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources") 

def postprocess_segmentation_torch(segmentation):
    """
    Post-process the 3D segmentation output to reduce false predictions using PyTorch.

    Parameters:
    segmentation (numpy array): The 3D array with segmentation labels.

    Returns:
    numpy array: Post-processed segmentation.
    """
    
    # Convert segmentation to torch tensor and move to GPU
    segmentation = torch.tensor(segmentation, device='cuda')
    
    # Initialize the post-processed segmentation with zeros
    post_processed = torch.zeros_like(segmentation)
    
    # Get unique labels in the segmentation
    labels = torch.unique(segmentation)
    
    for label_id in labels:
        if label_id == 0:  # Skip the background
            continue
        
        # Extract regions corresponding to the current label
        binary_mask = segmentation == label_id
        
        if label_id < 22:
            post_processed[binary_mask] = label_id
            continue
        
        # Convert binary mask to CPU for connected component analysis
        binary_mask_cpu = binary_mask.cpu().numpy()
        
        # Label connected components
        labeled_mask, num_features = ndi.label(binary_mask_cpu)
        
        # Find the largest connected component
        if num_features > 0:
            regions = regionprops(labeled_mask)
            largest_component = max(regions, key=lambda region: region.area)
            largest_mask = (labeled_mask == largest_component.label)
            
            # Convert back to torch tensor and move to GPU
            largest_mask = torch.tensor(largest_mask, device='cuda')
            
            # Apply binary closing to smooth the boundaries using PyTorch's 3D convolution
            dilated = F.conv3d(largest_mask[None, None, :, :, :].float(), torch.ones(1, 1, 3, 3, 3, device='cuda'), padding=1) > 0
            
            # Apply erosion on the dilated result
            eroded = F.conv3d(dilated.float(),  torch.ones(1, 1, 3, 3, 3, device='cuda'), padding=1)
            closed = (eroded > ( torch.ones(1, 1, 3, 3, 3, device='cuda').sum() - 1)).float()  # Ensure proper binary closing behavior


            # Add the processed mask to the post-processed segmentation
            post_processed[closed[0, 0] == 1] = label_id
    
    return post_processed.cpu().numpy()

def postprocess_segmentation(segmentation):
    """
    Post-process the 3D segmentation output to reduce false predictions.

    Parameters:
    segmentation (numpy array): The 3D array with segmentation labels.

    Returns:
    numpy array: Post-processed segmentation.
    """
    
    # Initialize the post-processed segmentation with zeros
    post_processed = np.zeros_like(segmentation)
    
    # Get unique labels in the segmentation
    labels = np.unique(segmentation)
    
    for label_id in labels:
        if label_id == 0:  # Skip the background
            continue
        
        # Extract regions corresponding to the current label
        binary_mask = segmentation == label_id
        
        if label_id < 22:
            post_processed[binary_mask] = label_id
            continue
        
        # Label connected components
        labeled_mask, num_features = ndi.label(binary_mask)
        
        # Find the largest connected component
        if num_features > 0:
            largest_component = max(regionprops(labeled_mask), key=lambda region: region.area)
            largest_mask = labeled_mask == largest_component.label
            
            # Fill holes within the largest component
            #filled_mask = ndi.binary_fill_holes(largest_mask)
            
            # Apply binary closing to smooth the boundaries
            smoothed_mask = binary_closing(largest_mask, ball(2))
            
            # Add the processed mask to the post-processed segmentation
            post_processed[smoothed_mask] = label_id

    return post_processed

def run():
    # Read the input
    # Read the input
    image, spacing, direction, origin = load_image_file_as_array(
        location=INPUT_PATH / "images/ct-angiography",
    )
    
    # Process the inputs: any way you'd like
    _show_torch_cuda_info()

    ############# Lines You can change ###########
    # Set the environment variable to handle memory fragmentation
    start_time= time.time()
    
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.cuda.empty_cache()
    
    nnUNet_results = "/opt/app/resources/nnUNet_results/"
    #nnUNet_results = "/opt/app/resources/nnUNet_results/nnUNetTrainerUMambaEncCovBot__nnUNetPlans__3d_fullres"
    
    predictor = nnUNetPredictor(
        tile_step_size=0.6,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=True,
        verbose_preprocessing=True,
        allow_tqdm=False
    )
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, 'nnUNetTrainerSwinUMambaRegionLoss3Dscan_c__nnUNetResEncUNetLPlans176128128__3d_fullres'),
        use_folds=("all"),
        checkpoint_name='checkpoint_latest.pth',
    )
    print(spacing)
    props = {
        'sitk_stuff':{
            'spacing': spacing,
            'origin':origin,
            'direction':direction
        },
        'spacing': list(spacing)[::-1]
    }
    image = image[None].astype(np.float32)
    pred_start_time= time.time()
    pred_array = predictor.predict_single_npy_array(image, props, None, None, False)
    print("pred time cost:", time.time()-pred_start_time)
    del image
    pred_array = pred_array.astype(np.uint8)
    print("pred_array.shape: ", pred_array.shape)
    
    postprocess_start_time= time.time()
    aortic_branches = postprocess_segmentation_torch(pred_array)
    print("pred time cost:", time.time()-postprocess_start_time)
    torch.cuda.empty_cache()
    gc.collect()    
    #aortic_branches = pred_label.squeeze().permute(2, 1, 0).numpy()

    print(f"Aortic Branches: Min={np.min(aortic_branches)}, Max={np.max(aortic_branches)}, Type={aortic_branches.dtype}")
    
    ########## Don't Change Anything below this 
    # For some reason if you want to change the lines, make sure the output segmentation has the same properties (spacing, dimension, origin, etc) as the 
    # input volume

    # Save your output
    write_array_as_image_file(
        location=OUTPUT_PATH / "images/aortic-branches",
        array=aortic_branches,
        spacing=spacing, 
        direction=direction, 
        origin=origin,
    )
    print('Saved!!!')
    
    print("time cost:", time.time()-start_time)
    
    return 0

def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    input_files = glob(str(location / "*.tiff")) + glob(str(location / "*.mha"))
    result = SimpleITK.ReadImage(input_files[0])
    spacing = result.GetSpacing()
    direction = result.GetDirection()
    origin = result.GetOrigin()
    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result), spacing, direction, origin


def write_array_as_image_file(*, location, array, spacing, origin, direction):
    location.mkdir(parents=True, exist_ok=True)

    # You may need to change the suffix to .tiff to match the expected output
    suffix = ".mha"

    image = SimpleITK.GetImageFromArray(array)
    image.SetDirection(direction) # My line
    image.SetOrigin(origin)
    SimpleITK.WriteImage(
        image,
        location / f"output{suffix}",
        useCompression=True,
    )


def _show_torch_cuda_info():

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())