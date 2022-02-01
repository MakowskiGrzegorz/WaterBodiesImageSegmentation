# WaterBodiesImageSegmentation



## TODO LIST
1. config for model creation             <-- partialy done
2. cleanup in code
- have wgan and simple gan in config
* for this we should think about what are the differences and try to exclude them into different files 
- two different block with conv2dtranspose and upsample
- fit this blocks in config
- historical averaging features etc in config
- wgan on different branch so we have to merge it after cleanup
---------------------------------------------------------
generator_blocks.py
discriminator_blocks.py
generator.py
discriminator.py
GAN.py   <-- this is framework
loss.py?

I think that WGAN and DCGAN have differences in last layer and how loss calculation happens




3. script for training models with different configs
4. use better GAN
- find which gan is the best in zoo
- download it and try to use

5. inference
- cleanup in inference code
- try to create inference notebook because it will fit best there

