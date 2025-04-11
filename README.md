# GCpu_OpticalFlow
This code is an accelerated open-source software for 2D optical flow estimation and mechanical strain.
The code is based on [D. Sun](https://cs.brown.edu/people/dqsun/pubs/cvpr_2010_flow.pdf) method with some improvements.
## Requirements
There are two versions of the software: a CPU and a GPU version.
To use the code, you  will need Python 3 (3.6 or higher) with the following modules:
**For the CPU version:**
- [Numpy](https://numpy.org/) 1.20.3 or newer
- [Scikit-image](https://scikit-image.org/) 0.16.2  or newer
- [Scipy](https://scipy.org/) 1.6.3 or newer
- [OpenCV](https://opencv.org/) 4.2.0 or newer
- You may also need [Numba](https://numba.pydata.org/) if Li and Osher filter will be used

**For the GPU version:**
For the GPU version you will first need an NVIDIA CUDA GPU with the Compute Capability 3.0 or larger.
Beside the previous packages some additional ones will be needed to run the code on GPU:
- [Cupy](https://cupy.dev/)
- [cuCIM API](https://docs.rapids.ai/api/cucim/stable/api.html)
The GPU version was tested using [Cupy](https://cupy.dev/) 9.2.0 and [cuCIM API](https://docs.rapids.ai/api/cucim/stable/api.html) 21.10.01

## Installation

### Setting up a virtual environment (recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/GCpu_OpticalFlow.git
   cd GCpu_OpticalFlow
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```

3. Activate the virtual environment:
   - On Linux/macOS:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

4. Install the required packages for the CPU version:
   ```bash
   pip install numpy>=1.20.3 scipy>=1.6.3 scikit-image>=0.16.2 opencv-python>=4.2.0 matplotlib numba
   ```

   Note: If you want to display interactive plots, you'll need the Tk backend for matplotlib. On most Linux systems, you can install it with:
   ```bash
   sudo apt-get install python3-tk
   ```

5. (Optional) For the GPU version, install additional packages:
   ```bash
   pip install cupy cucim
   ```

## Running the Tests

Some image sequences are provided in the folder **Images** to test the code.

### Basic Usage

1. Make sure your virtual environment is activated:
   ```bash
   source venv/bin/activate  # On Linux/macOS
   ```

2. Run the test script on a pair of images:
   ```bash
   cd Test
   python mainscript.py ../Images/Test/ C001H001S0003000001.tif C001H001S0004000001.tif
   ```

   This will:
   - Compute the optical flow between the two images
   - Calculate the strain fields (horizontal and vertical)
   - Generate visualization images:
     - `FlowAndStrainImg.png`: Combined visualization of flow field and strain fields
     - `StrainImg_Exx.png`: Horizontal strain field
     - `StrainImg_Eyy.png`: Vertical strain field
     - `StrainImg.png`: Copy of horizontal strain field (for compatibility)

### Advanced Usage

You can customize the parameters of the optical flow algorithm by passing additional arguments:

```bash
python mainscript.py ../Images/Test/ C001H001S0003000001.tif C001H001S0004000001.tif pyram_levels=4 lmbda=30000 max_iter=15
```

Available parameters:
- `pyram_levels`: Number of pyramid levels (default: 3)
- `factor`: Downsampling factor (default: 2.0)
- `ordre_inter`: Order of interpolation used for resizing (default: 3)
- `size_median_filter`: Median filter size (default: 5)
- `max_linear_iter`: Maximum number of iterations used for linearization (default: 1)
- `max_iter`: Warping steps number (default: 10)
- `lmbda`: Tikhonov Parameter (default: 30000)
- `lambda2`: Li and Osher median parameter for non-local term (default: 0.001)
- `lambda3`: Li and Osher median parameter for smoothing auxiliary fields (default: 1.0)
- `LO_filter`: Whether to use Li and Osher filter (0 or 1, default: 0)

## Output Files

The script generates the following output files:
- `u_cucim.npy`: Horizontal displacement field (NumPy array)
- `v_cucim.npy`: Vertical displacement field (NumPy array)
- `FlowAndStrainImg.png`: Visualization of flow field and strain fields
- `StrainImg_Exx.png`: Visualization of horizontal strain field
- `StrainImg_Eyy.png`: Visualization of vertical strain field

To calculate the strain field, the script uses the [gradient function of NumPy](https://numpy.org/doc/stable/reference/generated/numpy.gradient.html) (or [CuPy](https://docs.cupy.dev/en/stable/reference/generated/cupy.gradient.html) if you are using the GPU version).

## Documentation

For more detailed information, please refer to the [online documentation](https://gcpu-opticalflow.readthedocs.io/en/latest/) or the documentation in the `Doc` directory.
