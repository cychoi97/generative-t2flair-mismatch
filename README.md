<h2 align="center">
  Phenotype augmentation using generative AI <br>
  for isocitrate dehydrogenase mutation prediction in glioma
</h2>

<div align="center">
  <a>Ha&nbsp;Kyung&nbsp;Jung</a><sup>1,†</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://github.com/cychoi97" target="_blank">Changyong&nbsp;Choi</a><sup>2,3,†</sup> &ensp; <b>&middot;</b> &ensp;
  <a>Ji&nbsp;Eun&nbsp;Park</a><sup>4,*</sup> &ensp; <b>&middot;</b> &ensp;
  <a>Seo&nbsp;Young&nbsp;Park</a><sup>5</sup> &ensp; <b>&middot;</b> &ensp;
  <a>Jae&nbsp;Ho&nbsp;Lee</a><sup>6</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://scholar.google.com/citations?hl=ko&user=e93LeuwAAAAJ" target="_blank">Namkug&nbsp;Kim</a><sup>2,4</sup> &ensp; <b>&middot;</b> &ensp; 
  <a>Ho&nbsp;Sung&nbsp;Kim</a><sup>4</sup> <br><br>

  <sup>1</sup>Department of Radiology, Keimyung University Dongsan Hospital, Keimyung University School of Medicine, Daegu, Korea <br>
  <sup>2</sup>Department of Convergence Medicine, Asan Medical Center, University of Ulsan College of Medicine <br>
  <sup>3</sup>Department of Biomedical Engineering, AMIST, Asan Medical Center, University of Ulsan College of Medicine <br>
  <sup>4</sup>Department of Radiology and Research Institute of Radiology, Asan Medical Center, University of Ulsan College of Medicine, Seoul, Korea <br>
  <sup>5</sup>Department of Statistics and Data Science, Korea National Open University, Seoul, Korea <br>
  <sup>6</sup>Department of Korea and Center for Imaging Science, Samsung Medical Center, Sungkyunkwan University School of Medicine, Seoul, Korea <br>
  †equal contribution <br>
  *corresponding author <br>
</div>

<div align="center">
    <img src="assets/t1_t2_flair.png" height="300">
</div>

## Abstract
This study investigated the effects of feature augmentation, which uses generated images with specific imaging features, on the performance of isocitrate dehydrogenase (IDH) mutation prediction models in gliomas. A total of 598 patients were included from our institution (310 training, 152 internal test) and the Cancer Genome Atlas (136 external test). Score-based diffusion models were used to generate T2-weighted, FLAIR, and contrast-enhanced T1-weighted image triplets. Three neuroradiologists independently assessed visual Turing tests and various morphological features. Multivariable logistic regression models were developed using real images, random augmented data, and feature-augmented datasets. While random augmentation yielded models with AUCs comparable to real image-based models, it led to reduced specificity, particularly in the external test set (specificity: 83.2% vs. 73.0%, P = .013). In contrast, feature-augmented models maintained stable diagnostic performance; however, when more than 70% of training images included synthetic T2-FLAIR mismatch signs, AUC decreased in the external test set (AUC: 0.905–0.906 for ≤ 70%; 0.902–0.876 for ≥ 80%). These findings highlight the value of phenotype-specific augmentation for IDH prediction, while emphasizing the need to optimize augmentation proportion to avoid performance degradation.

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
