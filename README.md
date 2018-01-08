# SG-GAN

TensorFlow implementation of SG-GAN. 

## Prerequisites
- TensorFlow (implemented in v1.3)
- numpy
- scipy
- pillow

## Getting Started
### Train
- Prepare dataset. We present example images in `datasets` folder for reference as data format, and scripts `prepare_data.py` and `segment_class.py` for reference in preparing dataset.

- Train a model:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py
```
Models are saved to `./checkpoints/` (can be changed by passing `--checkpoint_dir=your_dir`). 

- Continue training a model (useful for updating parameters)
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --continue_train 1
```

### Test
- Finally, test the model:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --phase test --img_width 2048 --img_height 1024
```
Adapted test images will be outputted to `./test/` (can be changed by passing `--test_dir=your_dir`).

### Reference
- The TensorFlow implementation of CycleGAN, https://github.com/xhujoy/CycleGAN-tensorflow

