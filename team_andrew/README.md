# Team Andrew: The Quest for Quality Questions

## Hardware
- CPU specs: Intel Xeon Gold CPUs
- Number of CPU cores: 32
- CPU memory: 100GB
- GPU specs: NVIDIA A100-PCIE-40GB
- GPU memory: 40GB
- Number of GPUs: 1


## OS/platform
- Name: Ubuntu
- Version: 20.04.5 LTS (Focal Fossa)


## 3rd-party software
None


## How to train our model
Run the train command (step 1) in `entry_points.md`. This saves a deepspeed checkpoint. Run convert model checkpoint to HuggingFace format command (step 2) to save a HuggingFace model checkpoint.


## How to make predictions on a new test set
Run the predict command (step 3) in `entry_points.md`.


## Important side effects of your code
None


## Key assumptions made by your code
None