![logo](assets/logo3.png)

# UniLat3D: Geometry-Appearance Unified Latents for Single-Stage 3D Generation

### [Project Page](https://unilat3d.github.io/) | [arXiv Paper](https://arxiv.org/abs/2509.25079) | [Huggingface Live Demo](https://huggingface.co/spaces/UniLat3D/UniLat3D)


<a href="https://guanjunwu.github.io/">Guanjun Wu</a><sup>1,2\*</sup>, <a href="https://jaminfong.cn/">Jiemin Fang</a><sup>1\*✉</sup>, <a href="https://chensjtu.github.io/">Chen Yang</a><sup>1\*</sup>, <a href="https://scholar.google.com/citations?user=2dCJlg4AAAAJ&hl=zh-CN">Sikuang Li</a><sup>1,3</sup>, <a href="https://github.com/taoranyi">Taoran Yi</a><sup>1,2</sup>, <a href="https://github.com/lujzz">Jia Lu</a><sup>1,2</sup>, <a href="https://github.com/Zanue">Zanwei Zhou</a><sup>1,3</sup>, 
<a href="https://jumpat.github.io/jumpcat/">Jiazhong Cen</a><sup>1,3</sup>,   
<a href="http://lingxixie.com/">Lingxi Xie</a><sup>1</sup>, <a href="https://scholar.google.com/citations?user=Ud6aBAcAAAAJ&hl=zh-CN">Xiaopeng Zhang</a><sup>1</sup>, <a href="https://www.eric-weiwei.com/">Wei Wei</a><sup>2</sup>, <a href="http://eic.hust.edu.cn/professor/liuwenyu/">Wenyu Liu</a><sup>2</sup>, <a href="https://xwcv.github.io">Xinggang Wang</a><sup>2</sup>, <a href="https://www.qitian1987.com/">Qi Tian</a><sup>1✉</sup>  

<sup>1</sup> Huawei Inc. 
<sup>2</sup> Huazhong University of Science and Technology 
<sup>3</sup> Shanghai Jiao Tong University  

<sup>*</sup>Equal contributions.  <sup>✉</sup>Corresponding author.


![teaser_image](assets/teaser_image.png)

**Important Note**: This open-source repository is intended to provide a reference implementation.


## News
- 2025.11.10 Release the mesh decoder checkpoints and its corresponding inference code.

- 2025.11.3 Release the Encoder/ 3DGS decoder / Flow model checkpoints, inference code and huggingface live demo. The mesh decoder checkpoints would be released.

- 2025.9.30 - Initialize the repository. The complete code is under reconstruction for release, and model weights will be released. Please stay tuned!

## Get Started
1. clone project:
```bash
git clone --recursive https://github.com/UniLat3D/UniLat3D
cd UniLat3D
```
Then, follow [TRELLIS](https://github.com/microsoft/TRELLIS/) to prepare your environment.

2. prepare DINOv3:

Follow [thislink](https://github.com/facebookresearch/dinov3) to clone dinov3 under your project：
```
mkdir external
git clone https://github.com/facebookresearch/dinov3 external/dinov3
```
Then, put the `dinov3_vith16plus` weights into: `./external/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth`.

3. run

```
python run.py --save_mp4
```


4. Online Demo


```bash
python app.py
```
The webpage would start.


5. Mesh decoder inference

Given a generated UniLat, one can transform it to a 3DGS or a mesh using corresponding decoders. To achieve mesh decoding, you should install spconv library and its dependencies. However, the original spconv library is known for its data [exceeding int32 range bug](https://github.com/traveller59/spconv/issues/706) when the mesh is very large while UniLat3D may synthesize high resolution meshes. To fix this issue, we have modified both the cumm and spconv libraries to skip the int32 data range exceeding check. These modified libraries are included as submodules: [spconv-int32](https://github.com/chensjtu/spconv-int32) and [cumm-int32](https://github.com/chensjtu/cumm-int32). You can install them by running the following commands:
```bash
cd submodules/cumm-int32; pip install -e .
cd submodules/spconv-int32; pip install -e .
```
Please note that when you first run the spconv-int32, it will automatically build the spconv library. This process is time-consuming and the skipped int32 data range exceeding check is only for development purposes.

To generate both 3DGS (for visualization) and mesh outputs from images:
```bash
python run.py --save_mp4 --formats mesh
```

To generate only mesh output without video:
```bash
python run.py --formats mesh
```


## Contributions

**This project is still under development. Please feel free to raise issues or submit pull requests to contribute to our codebase.**

Some source code of ours is borrowed from [TRELLIS](https://github.com/microsoft/TRELLIS/) and [TripoSF](https://github.com/VAST-AI-Research/TripoSF). We sincerely appreciate these excellent works.


## Citation

```
@article{wu2025unilat3d,
  title={UniLat3D: Geometry-Appearance Unified Latents for Single-Stage 3D Generation},
  author={Wu, Guanjun and Fang, Jiemin and Yang, Chen and Li, Sikuang and Yi, Taoran and Lu, Jia and Zhou, Zanwei and Cen, Jiazhong and Xie, Lingxi and Zhang, Xiaopeng and Wei, Wei and Liu, Wenyu and Wang, Xinggang and Tian, Qi},
  journal={arXiv preprint arXiv:2509.25079},
  year={2025}
}
```

## LICENSE

This project is licensed under the Apache License 2.0 for all original code.

However, it includes third-party components that are distributed under their own
licenses. These components retain their original licenses. Use of these
third-party libraries is subject to the terms of their respective licenses.


