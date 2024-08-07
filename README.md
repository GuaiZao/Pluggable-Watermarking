# Pluggable Watermarking of Deepfake Models for Deepfake Detection
## The 33rd International Joint Conference on Artificial Intelligence
The implementation of [**Pluggable Watermarking of Deepfake Models for Deepfake Detection**](https://nesa.zju.edu.cn/download/bh_pdf_ijcai24-update.pdf)


## Usage:

The environment of the code please refer to the deepfake models you want to be watermarked.

Due to the different implementation methods of each Deepfake, please replace the parts in the code that need to be replaced with the corresponding implementation method of the Deepfake method.
We use HiFiface as an example for watermark embedding in the code.

## Function:

get_location.py This code can determine the watermark embedding position and train a mask.

watermarking.py This code has three modes, train, test and generate. 

All the three modes need a pretrained mask generated by get_location.py.

'train' mode: You can train watermark parameters in deepfake models and watermark extractor. 

'generate' mode: You can use a pretrained watermark deepfake model to generate watermarked deepfake images.

'test' mode: You can use a pretrained extractor to detect whether an image contains watermarks.

## Acknowledgment
We would like to express our gratitude to the following open-source projects:

'noise_layers' is mainly based on [Hidden](https://github.com/ando-khachatryan/HiDDeN)

[Hififace](https://github.com/maum-ai/hififace)

[SimSwap](https://github.com/neuralchen/SimSwap)

[FirstOrder](https://github.com/AliaksandrSiarohin/first-order-model)

[LATS](https://github.com/royorel/Lifespan_Age_Transformation_Synthesis)

[FaceShifter](https://github.com/taotaonice/FaceShifter)

[Stable Diffusion v1](https://github.com/CompVis/stable-diffusion)





