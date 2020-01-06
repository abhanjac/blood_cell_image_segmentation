# Objective: 
Segmentation of different blood cells in a digitized blood smear image.

This project is to create a semantic segmentation map of digitized blood smear images containing different blood cells using a convolutional neural networks.
This is also done to get a hands on with segmentation networks.

The neural network used here is a modified version of the [U-Net](https://arxiv.org/abs/1505.04597) segmentation network.
This network takes in a **224 x 224** image as input. This is a simple RGB colored image obtained from a digital microscope showing the different blood cells in a blood smear on a slide at **40x** magnification.
The network produces an output image which is also **224 x 224** pixels in size and shows the different **Red Blood Cells (RBC)**, **White Blood Cells (WBC)**, and **Platelets or Thrombocytes (THR)** regions in different colors.
With enough training the neural network will be able to predict and color code the different regions of the input image with different colors thereby producing a semantic segmentation map of the input image.
The network has to be first trained using a training dataset and validated with a validation dataset. After that it has to tested on a completely unseen test dataset to check its performance.
The images in the dataset used here has the following types of cells: **2** types of RBCs: **Infected RBC** (RBCs infected with malarial parasite), **Healthy RBC**; 
**5** types of WBCs: **Eosinophil**, **Basophil**, **Neutrophil**, **Lymphocytes**, and **Monocytes**; 
**2** types of Platelets: **Thrombocytes** (individual platelet cells) and **Platelet Clumps** (bunch of platelets appearing as a cluster).
So overall there are **9** objects and background, so **10** classes in the dataset. And in the output predicted segmentation map will have the following colors for the different objects.

* **1. Eosinophil** ![](https://placehold.it/20/00ffff?text=+)   `(color: cyan)`
* **2. Basophil** ![](https://placehold.it/20/008080?text=+)   `(color: teal)`
* **3. Neutrophil** ![](https://placehold.it/20/fbbebe?text=+)   `(color: pink)`
* **4. Lymphocytes** ![](https://placehold.it/20/808000?text=+)   `(color: olive)`
* **5. Monocytes** ![](https://placehold.it/20/000080?text=+)   `(color: navy)`
* **6. Thrombocytes** ![](https://placehold.it/20/ffff00?text=+)   `(color: yellow)`
* **7. Platelet Clumps** ![](https://placehold.it/20/ff00ff?text=+)   `(color: magenta)`
* **8. Infected RBC** ![](https://placehold.it/20/ff0000?text=+)   `(color: red)`
* **9. Healthy RBC** ![](https://placehold.it/20/00ff00?text=+)   `(color: green)`
* **10. background** ![](https://placehold.it/20/000000?text=+)   `(color: black)`

# Dataset Creation:
The images used for creating the training, testing and validation datasets are obtained from four different databases: 
* **Leukocyte Images for Segmentation and Classification (LISC) database**: This contains images of five types of WBCs on a background of RBCs. The images are labeled by the type of WBC in them, and each image also has a binary mask that indicates the pixels representing the WBC region.
* **Isfahan University of Medical Science (IUMC) database**: This has labeled images of individual WBCs with their binary masks. However, this database does not have Basophil images.
* **MAMIC database**: It has large blood smear images of healthy RBCs, THRs, Platelet clumps and Malaria infected RBCs. Occasionally, WBCs also appear in the MAMIC images, but they are not labelled. Every image contains multiple cells, without any binary masks to separate them.
* **KAGGLE database**: This contains images of individual healthy and infected RBCs, but without any binary masks. All the Malarial infection images in the last two databases are with Plasmodium Falciparum pathogen.

The main reason to combine all these different databases is the unavailability of a single annotated database that contains all types of blood cells (mentioned earlier) along with malaria infected RBCs.

For a robust training of the CNN, the training dataset should have a wide variety of combinations of the different blood cells. 
For example, there should be images with an Eosinophil and a Basophil with healthy RBCs in the background, images with a Monocyte and Platelet clumps on a background containing both healthy and infected RBCs, images containing only Lymphocytes on a background of infected RBCs, etc. None of the databases mentioned earlier has this much variety. 
Additionally, total number of WBC images over all the databases is around **391**, which is not sufficient for a good training. Hence, a fresh dataset was created which has the desired variations, using images from the previously mentioned databases as building blocks.

As a first step, a set of images is created that has only one kind of cell in them along with their binary masks. This is done for the LISC, KAGGLE, and MAMIC images. IUMC images are already in this format. 
The region of WBCs in the LISC images are cropped off using their masks to create individual images of WBCs. LISC and IUMC provides all the required WBC samples. 
One set of infected and healthy RBCs are obtained from KAGGLE. THRs, Platelet clumps and another set of infected and healthy RBCs are cropped out manually from several MAMIC images. 
The binary masks of the samples obtained from KAGGLE and MAMIC are created using simple image thresholding technique. 
Finally, all these newly created samples are resized such that they are similar in size to cells seen under a microscope with **40x** magnification. Some of these final samples are shown in Fig.~\ref{fig:modified_images}. 
The total number of samples obtained in this manner for different cells is given below: 

| Cell Types | LISC | IUMC | MAMIC | KAGGLE |
|:----------:|:----:|:----:|:-----:|:------:|
| Eosinophil | 37 | 42 | - | - |
| Basophil | 50 | - | - | - |
| Neutrophil | 47 | 38 | - | - |
| Lymphocyte | 45 | 32 | - | - |
| Monocyte | 48 | 36 | - | - |
| Thrombocyte | - | - | 82 | - |
| Platelet clump | - | - | 36 | - |
| Infected RBC | - | - | 407 | 13779 |
| Healthy RBC | - | - | 3539 | 13779 |


**The following flowchart shows how the training, testing and validation datasets are created.**

![](images/dataset_creation_flowchart.png)


First, all of the different types of image samples shown in the above table are separated into three groups namely: **training samples** (comprising **80%** of all the samples), **testing samples** (comprising **10%** of all the samples) and **validation samples** (comprising **10%** of all the samples). Only images from the training samples set are used to create the synthetic training dataset. Similarly, only images from the testing and validation samples sets are used to create the images for testing and validation datasets, respectively. This is done so that there are no common samples between the three datasets created and the neural networks never see any testing samples during training.

The size of the images in these datasets are **224 x 224** pixels. At first, some **1000 x 1000** background images are created that contain only RBCs in them. This is done by affixing randomly selected RBC samples on a blank image at random places. These locations are also recorded in a separate list. Altogether, **1500** such background images are created. **500** of these have only infected RBCs, **500** have only healthy RBCs, and **500** have a mix of both. Then, **224 x 224** blocks are cropped out of these images from random locations and WBC, THR and Platelet clump samples are affixed in them randomly. For each such image, the class names of the objects and the position and size of their bounding boxes are recorded in a separate list. The samples are also rotated at random angles while affixing them. 3 sample images obtained are shown in following figure.

![](images/final_images_2.png)


The total number of images in the final training, testing and validation sets are **65350**, **6560**, and **6560** respectively. All possible combinations of cells are present in among these images. But the datasets being too big are not added to this github repository.






# Training with weights:


# Current Framework: 
* Tensorflow 1.7.0 (with GPU preferred). 
* Opencv libraries, Ubuntu 16.04, Python 3.6.3 (Anaconda).
* This training does not necessarily needs GPUs, but they will make it much faster. This model is trained on one **NVIDIA P6000 Quadro GPU** in the [**Paperspace**](https://www.paperspace.com/) cloud platform.

# Modifications from original U-Net:

# Requirements: 
* The training set and testing set of images are created by combining can be downloaded from the [kaggle website](https://www.kaggle.com/c/dogs-vs-cats).
* The training and testing sets have to be de-compressed into two separate folders called **train** and **test** respectively.
* The training set has **25000** images out of which **5000** will be used to create a validation set and rest will be used for training. So, after de-compressing the training and testing sets, running the [utils.py](codes/utils.py) once, can create the validation set.
* Testing set has **12500** images.
* Training, validation and testing images are to be placed in folders named **train**, **valid** and **test** in the same directory that has the codes [train_classifier.py](codes/train_classifier.py).


# Data Preprocessing, Hyperarameter and Code Settings:
**[NOTE] All these settings are specified in the [config.py](codes/config.py) file.**
* The mean and standard deviation of the training set is used to normalize the images during training. And the same is used to normalize during validation and testing.
* Images are all resized into **224 x 224 x 3** size before feeding into the network.
* **Batch size: 100**
* **Epochs: 15**
* **Learning rate: 0.001 (upto epoch 1 - 10), 0.0003 (epoch 11), 0.00013 (epoch 12 - 13), 0.00003 (epoch 14 - 15)**. 
The learning rate is varied based on what fits best for increasing the validation accuracy.
* **Number of Classes ( nClasses ): 2 ('dog', 'cat')**
* A record of the **latest maximum validation accuracy** is also kept separately in a variable.
* The trained neural network model is saved if the validation accuracy in the current epoch is **better** than the latest maximum validation accuracy. 
* Only the 5 latest such saved models or checkpoints are retained inside the [temp](codes/temp) directory.
* The training logs are saved in the [logs](codes/logs) directory.

# Scripts and their Functions:
* [**config.py**](codes/config.py): All important parameters are defined here.
* [**utils.py**](codes/utils.py): All important functions used for the training or testing process are defined here. There are also some extra functions as well.
* [**train_classifier.py**](codes/train_classifier.py): The network model and training process are defined in this script.
* [**application.py**](codes/application.py): Evaluates the output on fresh images and also shows the localization ability of the Global Max-Pooling (GMP) layer of the network.

# Network Architecture:

### Layer 1:
**Conv --> Relu --> Batch-Norm --> Max-pool**

| Input | Conv Kernel | Filters | Output | Activation | Max-pool Kernel | Max-pool Stride | Max-pool Output |
|:-----:|:-----------:|:-------:|:------:|:----------:|:---------------:|:---------------:|:---------------:|
| 224 x 224 x 3 | 3 x 3 | 32 | 224 x 224 x 32 | Relu | 2 x 2 | 2 | 112 x 112 x 32 |

### Layer 2:
**Conv --> Relu --> Batch-Norm --> Max-pool**

| Input | Conv Kernel | Filters | Output | Activation | Max-pool Kernel | Max-pool Stride | Max-pool Output |
|:-----:|:-----------:|:-------:|:------:|:----------:|:---------------:|:---------------:|:---------------:|
| 112 x 112 x 32 | 3 x 3 | 64 | 112 x 112 x 64 | Relu | 2 x 2 | 2 | 56 x 56 x 64 |

### Layer 3:
**Conv --> Relu --> Batch-Norm --> Max-pool**

| Input | Conv Kernel | Filters | Output | Activation | Max-pool Kernel | Max-pool Stride | Max-pool Output |
|:-----:|:-----------:|:-------:|:------:|:----------:|:---------------:|:---------------:|:---------------:|
| 56 x 56 x 64 | 3 x 3 | 128 | 56 x 56 x 128 | Relu | 2 x 2 | 2 | 28 x 28 x 128 |

### Layer 4:
**Conv --> Relu --> Batch-Norm --> Max-pool**

| Input | Conv Kernel | Filters | Output | Activation | Max-pool Kernel | Max-pool Stride | Max-pool Output |
|:-----:|:-----------:|:-------:|:------:|:----------:|:---------------:|:---------------:|:---------------:|
| 28 x 28 x 128 | 3 x 3 | 256 | 28 x 28 x 256 | Relu | 2 x 2 | 2 | 14 x 14 x 256 |

### Layer 5:
**Conv --> Relu --> Batch-Norm**

| Input | Conv Kernel | Filters | Output | Activation |
|:-----:|:-----------:|:-------:|:------:|:----------:|
| 14 x 14 x 256 | 3 x 3 | 512 | 14 x 14 x 512 | Relu |

### Layer 6:
**Conv --> Relu --> Batch-Norm**

| Input | Conv Kernel | Filters | Output | Activation |
|:-----:|:-----------:|:-------:|:------:|:----------:|
| 14 x 14 x 512 | 1 x 1 | 256 | 14 x 14 x 256 | Relu |

### Layer 7:
**Conv --> Relu --> Batch-Norm --> Global-Max-Pool (GMP)**

| Input | Conv Kernel | Filters | Output | Activation | GMP Kernel | GMP Stride | GMP Output |
|:-----:|:-----------:|:-------:|:------:|:----------:|:----------:|:----------:|:----------:|
| 14 x 14 x 256 | 3 x 3 | 512 | 14 x 14 x 512 | Relu | 14 x 14 | 1 | 1 x 1 x 512 |

### Layer 8:
**Dense --> Dropout --> Softmax**

The output from the Layer 7 is flattened from 1 x 1 x 512 to the shape of 512 and fed into a dense layer. The dense layer has 2 output nodes, as there are only two (dog and cat) classification categories.

| Input | Output | Keep-probablity of Dropout | Activation |
|:-----:|:------:|:--------------------------:|:----------:|
| 512 | 2 (nClasses) | 0.5 | Softmax |

# Short Description of Training:
The network architecture is defined in the [train_classifier.py](codes/train_classifier.py) script.
The training process is also defined in the same script. Several functions and parameters that are used by the training process are defined in the [utils.py](codes/utils.py) and [config.py](codes/config.py) scripts.
Training happens in batches and after every epoch the model evaluated on the validation set. The training and validation accuracy are recorded in the log files (which are saved in the [logs](codes/logs) directory) and then the model is saved as a checkpoint. 
Another **json** file is also saved along with the checkpoint, which contains the following details:

* Epoch, Batch size and Learning rate
* Mean and Standard deviation of the training set.
* Latest maximum validation accuracy.
* A statistics of the epoch, learning rate, training loss, training accuracy and validation accuracy upto the current epoch.

These information from this json file are reloaded into the model to restart the training, in case the training process got stopped or interrupted because of any reason.

According to the [Weakly-supervised learning with convolutional neural networks paper](http://leon.bottou.org/publications/pdf/cvpr-2015.pdf), the **Global Max-Pooling (GMP)** layer is able to localize the parts of the image which the network emphasizes on to classify objects. 
In this network as well the GMP layer is used. 

GMP layers are used to reduce the spatial dimensions of a three-dimensional tensor in the final layers of the newtork. A tensor with dimensions **hxwxd** is reduced in size to have dimensions **1x1xd** by the GMP layers. It reduces each hxw feature map to a single number by simply taking the maximum of all points in the hxw feature map. The following figure will make it more clear.

![](images/global_max_pooling.png)

Once the 1x1xd tensor is formed, its output is flattened and is fed to another dense layer that does the final classification in its output. For this case, the GMP layer forms a 1x1x512 tensor which is flattened into a 512 tensor and then converted by a dense layer (Layer 8) into a tensor of size 2 (equal to the number of output classes) which gives the final classification output.

The dense layer (Layer 8) has a weight coming from each of the nodes of the 512 layer to the final 2 node layer. These weights are collected and multiplied to the 14x14 feature maps of Layer 7 (from which the nodes of Layer 8 are created by the GMP layer). All these multiplied feature maps are then combined together to form a **class activation map** as shown in the following figure.

![](images/class_activation_mapping.png)

This class activation map shows how the region of the object is localized by the use of the GMP layer.

After classification the localzation ability of the GMP layers are tested and the results are shown in the result section.

# Results:
### The final accuracies of the model are as follows:

| Training Accuracy | Validation Accuracy | Testing Accuracy |
|:-----------------:|:-------------------:|:----------------:|
| 99.99 % | 93.57 % | 93.48 % |

### Prediction label superimposed on the input image fed to the network.

![cat_image_1_with_prediction](images/cat_image_1_with_prediction.png)
![dog_image_1_with_prediction](images/dog_image_1_with_prediction.png)

Next, the regions where the network focusses to find the most important features to classify the object is found out.
This is represented by the heat map shown below, obtained from the GMP layer.
This is found out in the same manner as explained in the [Weakly-supervised learning with convolutional neural networks paper](http://leon.bottou.org/publications/pdf/cvpr-2015.pdf).

### Heat map showing the regions where the network focusses to classify the objects.

![cat_image_1_gmp_layer_heat_map](images/cat_image_1_gmp_layer_heat_map.png)
![dog_image_1_gmp_layer_heat_map](images/dog_image_1_gmp_layer_heat_map.png)

### Heat map superimposed on the actual image

![cat_image_1_gmp_layer_superimposed](images/cat_image_1_with_gmp_layer_superimposed.png)
![dog_image_1_gmp_layer_superimposed](images/dog_image_1_with_gmp_layer_superimposed.png)

# Observations and Discussions:

The heat map does not always engulfs the complete object as is seen in the next set of figures. 

![cat_image_2_with_prediction](images/cat_image_2_with_prediction.png)
![cat_image_2_gmp_layer_superimposed](images/cat_image_2_with_gmp_layer_superimposed.png)

![dog_image_2_with_prediction](images/dog_image_2_with_prediction.png)
![dog_image_2_gmp_layer_superimposed](images/dog_image_2_with_gmp_layer_superimposed.png)

This is because the most important features required to classify the object is often only a part of the object and not its complete body.
In case of cats for most of the images, the most red part of the heat map was near the face of the cat. As that is the most significant feature to identify it (as per the networks judgement). The same is true for dogs as well.

However, in cases where the face of the cat or dog is not sufficient to identify it, the network looks for other features from other part of the object (as seen in the first set of figures).
This is also evident from the following image of the dog. In this image because the face of the dog is not highlighted properly (due to a dark environment), so the network focusses on the legs of the dog to classify it.

![dog_image_3_with_prediction](images/dog_image_3_with_prediction.png)
![dog_image_3_gmp_layer_superimposed](images/dog_image_3_with_gmp_layer_superimposed.png)

### An overall look of the images with the superimposed heat maps is shown below.

![cat](images/cat.gif)
![dog](images/dog.gif)

The video of this detection can also be found on [Youtube](https://www.youtube.com/) at this [link](https://www.youtube.com/watch?v=gws5meW1_o0).





