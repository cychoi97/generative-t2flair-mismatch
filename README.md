<h2 align="center"> Generative T2-FLAIR mismatch sign in glioma: <br>
  Effect of augmenting phenotype for isocitrate dehydrogenase mutation prediction <br>
</h2>


<div align="center">
    <img src="assets/t1_t2_flair.png" height="300">
</div>


## Dependencies

Install the other packages in `requirements.txt`, jax, jaxlib, numpy, and opencv-python as following:
```bash
pip install -r requirements.txt
pip install jax==0.4.6 jaxlib==0.4.6 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.htm
pip install numpy==1.23.0
pip install opencv-python==4.5.5.64
```

### Prepare your own dataset

For example, you should set dataset path following:
```text
root_path
    ├── train
          ├── <Patient_Folder>
                ├── T1CE
                      ├── 0001.npy
                      ├── 0002.npy
                      └── 0003.npy
                ├── T2
                └── FLAIR
    └── test
```


## Training

```python
python main.py --config='configs/ve/t1t2flair.py' --workdir='result' --mode=train
```

Model checkpoints and validation samples will be stored in `./result/checkpoints` and `./result/samples`, respectively.


## Sampling

```python
python t1t2flair_sampling.py
```

Sampling results will be stored in `./result/generated_images` as png file.


## Acknowledgement

Our main code is heavily based on [score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch).
