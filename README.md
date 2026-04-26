# Colorization of Black and White Images

## Overview

This project focuses on automatically converting grayscale (black and white) images into colored images using a **pre-trained deep learning model**.

The model is based on the Caffe framework and uses learned color distributions to predict realistic colors for grayscale inputs.


## Features

* Convert grayscale images to color automatically
* Uses pre-trained deep learning model (no training required)
* Simple command-line execution
* High-quality and realistic outputs


## Technologies Used

* Python
* OpenCV (DNN Module)
* NumPy
* Caffe Model (Pre-trained)


## Project Structure

```
colorize/
│── model/
│   ├── colorization_deploy_v2.prototxt
│   ├── colorization_release_v2.caffemodel
│   ├── pts_in_hull.npy
│
│── images/
│   ├── tiger.jpg
│
│── colorize.py
│── README.md
│── requirements.txt
```


## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/colorization-project.git
cd colorization-project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```


## Download Pre-trained Model

Download the following files and place them inside the `model/` folder:

* `colorization_deploy_v2.prototxt`
* `colorization_release_v2.caffemodel`
* `pts_in_hull.npy`

(You can find these from OpenCV’s official GitHub or colorization model sources.)


## Usage

Run the script using the command line:

```bash
python colorize.py --image path_to_image
```

### Example:

```bash
python colorize.py --image images/input.jpg
```


## Output

* Displays:

  * Original Image
  * Colorized Image


## Error Handling

* Checks if the image path exists
* Handles invalid or unreadable images


## How It Works

1. Load the pre-trained Caffe model
2. Convert the image from BGR to LAB color space
3. Extract the L (lightness) channel
4. Predict `a` and `b` color channels using the model
5. Merge L + (a,b) channels
6. Convert back to BGR format


## Future Improvements

* Add GUI (Streamlit or Tkinter)
* Batch image processing
* Video colorization support
* Model fine-tuning for better accuracy


## Author

Taruna B V


## Acknowledgment

* OpenCV Team
* Research on image colorization using deep learning
