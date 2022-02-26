# Importing everything we need
from logging import raiseExceptions
from sys import Path
import tensorflow as tf
import argparse,os
#Creating argparse argument parser
arguments_parser = argparse.ArgumentParser()
#Adding parsing arguments to argument parser
arguments_parser.add_argument("--input_dir",help="input dir for dataset")
arguments_parser.add_argument("--mode",help="specify working mode")
#Parsing arguments from argument parser
arguments=arguments_parser.parse_args()
#Specifying standard image size and data
IMG_WIDTH=256
IMG_HEIGHT=256
OUTPUT_CHANNELS=3
#Specifying batch size of 1 because of better performance for the U-Net described in pix2pix paper
BATCH_SIZE=1
#Initializing buffer size
BUFFER_SIZE=1
#Specifying lambda for calculating loss:
LAMBDA=100
#Initializing keras binary crossentropy loss object
binarycrossentropy_loss_object=tf.keras.losses.BinaryCrossEntropy(from_logits=True)
#Singe combined image file loading function
def load_images(image_file):
    #Reading image file and decoding it to a uint8 tensor
    image=tf.io.read_file(image_file)
    image=tf.io.decode_jpeg(image)
    #Splitting each image tensor into two tensors: with input and real images
    width=tf.shape(image)[1]//2
    input_image=image[:, width:, :]
    real_image=image[:, :width, :]
    #Convert both images to float32 tensors
    input_image=tf.cast(input_image,tf.float32)
    real_image=tf.cast(real_image,tf.float32)
    #Return loaded images
    return input_image, real_image
#Image processing functions
def resize_images(input_image,real_image,height,width):
    #Resize both input and real images to specified height and width
    input_image=tf.resize(input_image, [height,width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image=tf.resize(real_image, [height,width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #Return resized images
    return input_image, real_image
def random_crop_images(input_image,real_image):
    #Stack images for random cropping
    stacked_image=tf.stack([input_image,real_image],axis=0)
    #Random crop both images
    cropped_image=tf.image.random_crop(stacked_image,size=[2,IMG_HEIGHT, IMG_HEIGHT,3])
    #return random cropped images
    return cropped_image[0],cropped_image[1]
def normalize_images(input_image,real_image):
    #Normalizing the images to [-1,1]
    input_image=(input_image/127.5)-1
    real_image=(real_image/127.5)-1
    #Return normalized images
    return input_image, real_image
#Applying random jittering and mirroring for preprocessing training dataset as described in the pix2pix paper
@tf.function()
def random_jitter(input_image,real_image):
    #Resizing both images to higher resolution (286x286)
    input_image,real_image=resize_images(input_image,real_image,286,286)
    #Random cropping back to defined standard resolution (256x256)
    input_image,real_image=random_crop_images(input_image,real_image)
    #Random mirroring images
    if tf.random.uniform(())>0.5:
        input_image=tf.image.flip_left_right(input_image)
        real_image=tf.image.flip_left_right(real_image)
    #Return preprocessed images
    return input_image, real_image
#Images loading and processing functions
#Loading images for training function
def load_images_train(image_file):
    #Load images from combined image file
    input_image,real_image=load_images(image_file)
    #Preprocess both images
    input_image,real_image=random_jitter(input_image,real_image)
    #Normalize both images
    input_image,real_image=normalize_images(input_image,real_image)
    #Return processed images
    return input_image,real_image
#Loading images for testing function
def load_images_test(image_file):
    #Load images from combined image file
    input_image,real_image=load_images(image_file)
    #Resize both images to defined standard size
    input_image,real_image=resize_images(input_image,real_image,IMG_HEIGHT,IMG_WIDTH)
    #Normalize both images
    input_image,real_image=normalize_images(input_image,real_image)
    #Return processed images
    return input_image,real_image
#U-net keras downsampling(encoding) model generation function
def downsample(filters,size,apply_batchnormalization=True) -> tf.keras.Model: #Function outputs U-net keras downsampling(encoding) model with specified filters and size
    #Creating initializer that generates tensors with a normal distribution
    initializer=tf.random_normal_initializer(0.,0.02)
    #Creating empty tf.keras.Model
    downsample_model=tf.keras.Sequential()
    #Adding 2D convolution layer to keras model
    downsample_model.add(tf.keras.layers.Conv2D(filters,size,strides=2,padding='same',kernel_initializer=initializer,use_bias=False))
    #Applying batch normalization if not specified else:
    if apply_batchnormalization:
        #Adding normalization layer to keras model
        downsample_model.add(tf.keras.layers.BatchNormalization())
    #Adding activation layer with LeakyReLU activation function
    downsample_model.add(tf.keras.layers.LeakyReLU())
    #Return downsampling model
    return downsample_model
#U-net keras upsampling(decoding) model generation function
def upsample(filters,size,apply_dropout=False) -> tf.keras.Model: #Function outputs U-net keras upsampling(decoding) model with specified filter and size
    #Creating initializer that generates tensors with a normal distribution
    initializer=tf.random_normal_initializer(0.,0.02)
    #Creating empty tf.keras.Model
    upsample_model=tf.keras.Sequential()
    #Adding 2D deconvolution(Transposed convolution) layer to keras model
    upsample_model.add(tf.keras.layers.Conv2DTranspose(filters,size,strides=2,padding='same',kernel_initializer=initializer,use_bias=False))
    #Adding normalization layer to keras model
    upsample_model.add(tf.keras.layers.BatchNormalization())
    #Applying dropout if specified:
    if apply_dropout:
        #Adding dropout layer to the input of keras model
        #Dropout layer is used during training to prevent overtraining
        upsample_model.add(tf.keras.layers.Dropout(0.5)) #Dropout layer randomly sets input units to 0 with a frequency of 0.5. Inputs not set to 0 are scaled up by 2 such that the sum over all inputs is unchanged.
    #Adding activation layer with ReLU activation function
    upsample_model.add(tf.keras.layers.ReLU())
    #Return upsampling model
    return upsample_model
#U-net keras generator model generation function
def Generator() -> tf.keras.Model: #Function outputs generator keras model builded on U-net chain
    #Creating keras tensor instance
    inputs=tf.keras.layers.Input(shape=[256,256,3]) #tensor size 256x256x3
    #Creating downsample(encoding) models stack(encoding part of U-net)
    down_stack=[
        downsample(64,4,apply_batchnormalization=False), #output tensor size 128x128x64 (batch_size,128,128,64)
        downsample(128,4), #output tensor size 64x64x128 (batch_size,64,64,128)
        downsample(256,4), #output tensor size 32x32x256 (batch_size,32,32,256)
        downsample(512,4), #output tensor size 16x16x512 (batch_size,16,16,512)
        downsample(512,4), #output tensor size 8x8x512 (batch_size,8,8,512)
        downsample(512,4), #output tensor size 4x4x512 (batch_size,4,4,512)
        downsample(512,4), #output tensor size 2x2x512 (batch_size,2,2,512)
        downsample(512,4), #output tensor size 1x1x512 (batch_size,1,1,512)
    ]
    #Creating upsample(decoding) models stack(decoding part of U-net)
    up_stack=[
        upsample(512,4,apply_dropout=True), #output tensor size 2x2x1024 (batch_size,2,2,1024)
        upsample(512,4,apply_dropout=True), #output tensor size 4x4x1024 (batch_size,4,4,1024)
        upsample(512,4,apply_dropout=True), #output tensor size 8x8x1024 (batch_size,8,8,1024)
        upsample(512,4), #output tensor size 16x16x1024 (batch_size,16,16,1024)
        upsample(256,4), #output tensor size 32x32x512 (batch_size,32,32,512)
        upsample(128,4), #output tensor size 64x64x256 (batch_size,64,64,256)
        upsample(64,4), #output tensor size 128x128x128 (batch_size,128,128,128)
    ]
    #Creating initializer that generates tensors with a normal distribution
    initializer=tf.random_normal_initializer(0.,0.02)
    #Creating last 2D deconvolution(Transposed convolution) keras layer
    last_layer=tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS,4,strides=2,padding='same',kernel_initializer=initializer,activation='tanh') #output tensor size 256x256x3 (batch_size,256,256,3)
    #Creating and initializing variable for processed(output) tensor
    processed_tensor=inputs
    #Downsampling through the model
    #Creating list for processed_tensor instances for building path from encoder with ability to skip some encoding-decoding models(layers) and pass through to opposite decoding model(layer) 
    skips=[]
    #Going through each downsample model in down_stack
    for downsample_model in down_stack:
        #Applying current downsample model to processed_tensor
        processed_tensor=downsample_model(processed_tensor)
        #Adding current processed_tensor to skips list
        skips.append(processed_tensor)
    #Getting iterator for skips value
    skips=reversed(skips[:-1])
    #Upsampling through the model and estabilishing the skips connections
    for upstack_model,skip in zip(up_stack,skips):
        #Applying current upstack model to processed_tensor
        processed_tensor=upstack_model(processed_tensor)
        #Concatenating processed_tensor and skip to a single keras layer to add a skip connection in chain
        processed_tensor=tf.keras.layers.Concatenate([processed_tensor,skip])
    #Applying last 2D deconvolution(Transposed convolution) keras layer to processed_tensor
    processed_tensor=last_layer(processed_tensor)
    #Returning keras model that transforms input tensor to processed_tensor using U-net chain
    return tf.keras.Model(inputs=inputs, outputs=processed_tensor)
#Generator loss calculating function
def generator_loss(discriminator_generated_output,generator_output,target):
    #Getting sigmoid binary crossentropy loss between discriminator output and tensor of all ones
    gan_loss=binarycrossentropy_loss_object(tf.ones_like(discriminator_generated_output),discriminator_generated_output)
    #Getting mean absolute error between target and generator output
    mae_l1_loss=tf.reduce_mean(tf.abs(target-generator_output))
    #Calculating total generator loss
    total_generator_loss=gan_loss+(LAMBDA*mae_l1_loss)
    #Return losses
    return total_generator_loss,gan_loss,mae_l1_loss
#Main function
def main():
    #Reading input directory from arguments
    input_dir=arguments.input_dir
    #Reading mode from arguments
    mode=arguments.mode
    #Checking for existing of input directory
    if input_dir is None or not os.path.exists(input_dir):
        raiseExceptions("Input directory does not exist or is not specified")
    BUFFER_SIZE=len(os.listdir(path=str(Path(input_dir)/'train')))
    #If started in train mode
    if mode.lower()=='train':
        #Building an input pipeline
        #Creating tensorflow dataset of all images files
        train_dataset=tf.data.Dataset.list_files(str(Path(input_dir)/'train/*.jpg'))
        #Loading every image in dataset
        train_dataset=train_dataset.map(load_images_train,num_parallel_calls=tf.data.AUTOTUNE)
        #Shuffling dataset to randomize iteration order
        train_dataset=train_dataset.shuffle(BUFFER_SIZE)
        #Converting dataset to batch
        train_dataset=train_dataset.batch(BATCH_SIZE)
    #If started in test mode
    if mode.lower()=='test':
        #Building an input pipeline
        #Creating tensorflow dataset of all images files
        try:
            test_dataset=tf.data.Dataset.list_files(str(Path(input_dir)/'test/*.jpg'))
        except tf.errors.InvalidArgumentError:
            test_dataset=tf.data.Dataset.list_files(str(Path(input_dir)/'val/*.jpg'))
        #Loading every image in dataset
        test_dataset=test_dataset.map(load_images_train,num_parallel_calls=tf.data.AUTOTUNE)
        #Converting dataset to batch
        test_dataset=test_dataset.batch(BATCH_SIZE)
generator=Generator()
tf.keras.utils.plot_model(generator,show_shapes=True,dpi=64)