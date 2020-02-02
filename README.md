# Objective: 
Segmentation of different blood cells in a digitized blood smear image.

**A *Trailer* of Final Result:**

![](images/blood_cell_segmentation_resized.gif)

[**YouTube Link**](https://www.youtube.com/watch?v=iXx_U7ga3FU&feature=youtu.be)

---

This project is to create a semantic segmentation map of digitized blood smear images containing different blood cells using a convolutional neural networks.
This is also done to get a hands on with segmentation networks.

The neural network used here is a modified version of the [U-Net](https://arxiv.org/abs/1505.04597) segmentation network. The pdf of the paper is also present [here](extra_files/Unet.pdf).
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
* [**Leukocyte Images for Segmentation and Classification (LISC) database**](http://users.cecs.anu.edu.au/~hrezatofighi/publications.htm): This contains images of five types of WBCs on a background of RBCs. The images are labeled by the type of WBC in them, and each image also has a binary mask that indicates the pixels representing the WBC region.
* [**Isfahan University of Medical Science (IUMC) database**](https://misp.mui.ac.ir/fa): This has labeled images of individual WBCs with their binary masks. However, this database does not have Basophil images.
* [**MAMIC database**](http://fimm.webmicroscope.net/Research/Momic): It has large blood smear images of healthy RBCs, THRs, Platelet clumps and Malaria infected RBCs. Occasionally, WBCs also appear in the MAMIC images, but they are not labelled. Every image contains multiple cells, without any binary masks to separate them.
* [**KAGGLE database**](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria): This contains images of individual healthy and infected RBCs, but without any binary masks. All the Malarial infection images in the last two databases 
are with Plasmodium Falciparum pathogen.

The main reason to combine all these different databases is the unavailability of a single annotated database that contains all types of blood cells (mentioned earlier) along with malaria infected RBCs.

For a robust training of the CNN, the training dataset should have a wide variety of combinations of the different blood cells. 
For example, there should be images with an Eosinophil and a Basophil with healthy RBCs in the background, images with a Monocyte and Platelet clumps on a background containing both healthy and infected RBCs, images containing only Lymphocytes on 
a background of infected RBCs, etc. None of the databases mentioned earlier has this much variety. 
Additionally, total number of WBC images over all the databases is around **391**, which is not sufficient for a good training. Hence, a fresh dataset was created which has the desired variations, using images from the previously mentioned databases as building blocks.

As a first step, a set of images is created that has only one kind of cell in them along with their binary masks. This is done for the LISC, KAGGLE, and MAMIC images. IUMC images are already in this format. 
The region of WBCs in the LISC images are cropped off using their masks to create individual images of WBCs. LISC and IUMC provides all the required WBC samples. 
One set of infected and healthy RBCs are obtained from KAGGLE. THRs, Platelet clumps and another set of infected and healthy RBCs are cropped out manually from several MAMIC images. 
The binary masks of the samples obtained from KAGGLE and MAMIC are created using simple image thresholding technique. 
Finally, all these newly created samples are resized such that they are similar in size to cells seen under a microscope with **40x** magnification. Some of these final samples are shown in Fig.~\ref{fig:modified_images}. 
The total number of samples obtained in this manner for different cells is given below: 

| Cell Types | LISC | IUMC | MAMIC | KAGGLE |
