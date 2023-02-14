# thesis-SR
Repository for my graduation work

Full thesis in russian language can be found here - https://drive.google.com/file/d/1dZxL5ZqND9nowRU9bOsN7GGO1fLF3sgb/view?usp=share_link

This work explores different methods of images upscaling, which is poorly solved by classic interpolation but impressive results can be obtained with neural networks.

Main part of my work is working with SRGAN, ESRGAN architectures and training them.

Both models were trained on [*CelebA*](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset. [*Set5*](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html) images and others were used for testing.

# Results
Some results of training are shown here.
## SRGAN results
![SRGAN examples](results/srgan_examples.jpg?raw=true "SRGAN")

## ESRGAN results
![ESRGAN examples](results/esrgan_examples.jpg?raw=true "ESRGAN")

Most impressive results are visible on this images with ESRGAN. Upscale factor is 4.

![252px to 1008px](implemenations/esrgan/outputs/LRbic2_x4/sr-baby.png?raw=true "252px to 1008px")
![144px to 576px](implemenations/esrgan/outputs/LRbic2_x4/sr-bird.png?raw=true "144px tp 576px")
![126px to 504px](implemenations/esrgan/outputs/LRbic2_x4/sr-butterfly.png?raw=true "126px to 504px")
