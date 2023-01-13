# S3VAE
Unofficial implementation of [S3VAE](https://arxiv.org/abs/2005.11437) with pytorch \
This repository is inspired by [here](https://github.com/axellkir/S3VAE)

# Data
The Moving MNIST dataset contains 10,000 video sequences, each consisting of 20 frames. In each video sequence, two digits move independently around the frame, which has a spatial resolution of 64×64 pixels. The digits frequently intersect with each other and bounce off the edges of the frame. \
[More details](http://www.cs.toronto.edu/~nitish/unsupervised_video/)

# Results
### original moving mnist
![gif](results/0_original.gif 'gif')
![gif](results/1_original.gif 'gif')
![gif](results/2_original.gif 'gif')
![gif](results/3_original.gif 'gif')
![gif](results/4_original.gif 'gif')
![gif](results/5_original.gif 'gif')
![gif](results/6_original.gif 'gif')
![gif](results/7_original.gif 'gif')
![gif](results/8_original.gif 'gif')
![gif](results/9_original.gif 'gif')
![gif](results/10_original.gif 'gif')

<br>

### generated moving mnsit
![gif](results/0_generated.gif 'gif')
![gif](results/1_generated.gif 'gif')
![gif](results/2_generated.gif 'gif')
![gif](results/3_generated.gif 'gif')
![gif](results/4_generated.gif 'gif')
![gif](results/5_generated.gif 'gif')
![gif](results/6_generated.gif 'gif')
![gif](results/7_generated.gif 'gif')
![gif](results/8_generated.gif 'gif')
![gif](results/9_generated.gif 'gif')
![gif](results/10_generated.gif 'gif')

# How to use
```
python click/predict.py generate-moving-mnist -c s3vae_config.yaml
```

# To Do
- [ ] fix mutual information loss
- [ ] edit wandb log name
