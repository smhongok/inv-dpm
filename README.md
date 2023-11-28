# On Exact Inversion of DPM-solvers

Official repository of a paper 'On Exact Inversion of DPM-solvers'

by [Seongmin Hong](https://smhongok.github.io), Kyeonghyun Lee, [Suh Yoon Jeon](https://www.linkedin.com/in/suhyoonjeon), Hyewon Bae, and [Se Young Chun](https://icl.snu.ac.kr/pi).

Link: [Project webpage](https://smhongok.github.io/inv-dpm.html)

We provide two codes of experiment: Reconstruction, and the Tree-ring watermark detection.

## Reconstruction

<img src=scripts/fig2.png  width="800pt">

### Usage
Check out this [notebook](exact_inversion.ipynb) for reconstruction results. You can perform reconstructions on random images or images with specific prompts.

You can reproduce our experiment results by running the following command.
```
python main_recon.py --num_inference_steps 50 --solver_order 1 --test_num_inference_steps 50 --inv_order 1
```
For more adversarial cases, check and run [here](scripts/run_reconstruction.sh). 

### Parameters
The important hyperparameters are the following
- `num_inference_steps`: Steps of the sampling process.
- `solver_order`: Order of sampling process.
- `test_num_inference_steps`: Steps of the inversion process. The default value is the same as `num_inference_steps`
- `inv_order`: Order of inversion process. The default value is the same as `solver_order`
- `wo_decoder_inv`: Choosing whether to not use exact decoder inversion. The default value is False, to use exact decoder inversion.

## Applications: Tree-ring watermark detection

<img src=scripts/fig4.png  width="800pt">

### Usage
You can reproduce our experiment results by running the following command. Various scenarios of experiments using different sampling methods can be conducted by adjusting the hyperparameters. More detailed explanations of hyperparameters are provided in the next section.
```
python main_watermark_detection.py --length 1 --num_inference_steps 10 --inv_order 2
```

For more cases, check and run [here](scripts/run_detection.sh). Note that the shape of the watermark is fixed to tree-ring, since this is a complicated form of WM to detect.

### More details on experiment results
Upon running the experiment with --length 10, the results will be presented in the following format:
```
Generated with WM1 : |  6 | 4 | 0 | 
Generated with WM2 : |  1 | 9 | 0 | 
Generated with WM3 : |  0 | 0 | 10 | 
```
Each row represents the count of watermark (WM) detections for each generated watermark. The process involves inverting the generated image, calculating the l1 difference with each watermark, and selecting the one with the lowest difference as the detected watermark. In the example above, it indicates that:

The image generated with WM1 was correctly detected in 6 cases.
The image generated with WM2 was correctly detected in 9 cases.
The image generated with WM3 was correctly detected in all 10 cases.
The ideal result for the experiment is characterized by larger values along the diagonal entries, indicating accurate detection of the corresponding watermark.

### Parameters
The important hyperparameters for Tree-Ring are the following:

- `length`: The number of prompts to conduct the experiment. Our experiment is done with 100. In the example, "--length N" runs 3N images, so please carefully choose the number of images N. 
- `w_radius`: The radius of generated WMs. The default value is 6. Check `get_watermarking_patterns` in the main file, the `keys` tensor is used to define the form of WMs. To adjust the radius of WM, you should adjust this `keys` tensor.
- `target_num`: The number of generated WMs used to compare. Our implementation is fixed on 3, but it can be adjusted by slightly changing the size of the array storing metrics and confusion matrix.


Adjusting these parameters allows for fine-tuning the watermark detection process for different scenarios and experiment setups.
