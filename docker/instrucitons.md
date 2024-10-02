### Post-processing and Inference Pipeline

The `inference.py` Python script outlines the steps for post-processing a segmentation prediction and running the inference process in a Docker-based image portal for a CT angiography task. Below is a detailed breakdown of the functions and their purpose.

1. **`postprocess_segmentation_torch(segmentation)`**:
    - This function performs post-processing on the 3D segmentation output using PyTorch to reduce false predictions.
    - **Operations**:
        - Converts segmentation to a PyTorch tensor and moves it to the GPU.
        - Identifies unique labels and extracts binary masks for each label.
        - For labels >= 22, connected component analysis is performed to keep the largest component, followed by binary closing to smooth boundaries.
        - The final segmentation mask is returned after processing.

2. **`postprocess_segmentation(segmentation)`**:
    - A NumPy-based version of the post-processing function to clean up segmentation results.
    - **Operations**:
        - Identifies unique labels and processes binary masks for each label.
        - Connected components are labeled, and the largest one is kept for further processing.
        - A binary closing operation smooths the boundaries of the identified regions.

3. **`run()`**:
    - The main function that handles the inference process.
    - **Steps**:
        - Reads the input CT image and metadata (spacing, direction, origin).
        - Initializes the environment and clears CUDA memory.
        - Loads the trained model and runs the segmentation prediction on the input image.
        - Applies test-time augmentation and post-processing using the previously defined `postprocess_segmentation_torch()` function.
        - Saves the final processed segmentation output to the specified location.

4. **`load_image_file_as_array(location)`**:
    - This function loads the input CT image file as a NumPy array using SimpleITK.
    - **Returns**: The 3D image array, spacing, direction, and origin of the image.

5. **`write_array_as_image_file(location, array, spacing, origin, direction)`**:
    - Writes the post-processed segmentation array back to a file with the same properties as the input image (spacing, direction, origin).
    - **Outputs**: The final result is saved as a `.mha` or `.tiff` file.

6. **`_show_torch_cuda_info()`**:
    - Displays information about the available CUDA devices and their properties.

7. **Main Execution**:
    - The `run()` function is called to handle the entire pipeline, from loading the image to saving the processed output.
    - If the script is run as the main module, it exits after completing the entire workflow.
