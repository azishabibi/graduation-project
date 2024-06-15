follow the official document to install mmocr: https://mmocr.readthedocs.io/en/dev-1.x/
follow the official document to install deeplsd: https://github.com/cvg/DeepLSD
```
pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 
```
This works for me but other combination might work depending on cuda version.

Download the weights, unzip it and put them in the same dir as main.py. The weights can be found here: https://drive.google.com/file/d/1Jh8aC0WRi7cjVM01wOQDMbtwLGe-4aMO/view?usp=drive_link

The structure should be like this:
- output/
- utils/
- weights/
  - train4
  - train11
  - deeplsd_md.tar
- .gitignore
- 0.jpg
- gray.jpg
- main.py
- README.md
- requirements.txt
- server.py


run 
```
python main.py --image-path path/to/image
```
to generate the results. They can be found in ./output. The image contains the visualized images and npy has the prediction results.
The 