## Vanilla Autoencoder Implementation
This repository is about Vanilla AutoEncoder in Tensorflow 2 , I used tf.keras.Model and tf.layers.Layer instead of tf.keras.models.Sequential.  This allows us to customize and have full control of the model, I also used custom training instead of relying on the fit() function.  
In case we have very huge dataset, I applied online loading (by batch) instead of loading the data completely at the beginning. This will eventually not consume the memory.  

#### AutoEncoder Architecrure      
<p></p>
<center>
<img src="img/1111.png" align="center" width="700" height="300"/>
</center>

#### The Architecrure of Vanilla Autoencoder
<center>   
<img src="img/vanilla1.png" width="700" height="300"/>   
</center>

### Training on MNIST
<p></p>
<center>
<img src="img/mnist.png" width="400" height="350"/>
</center>

### Requirement
```
python==3.7.0
numpy==1.18.1
```
### How to use
Training & Prediction can be run as follows:    
`python train.py train`  
`python train.py predict img.png`  


### More information
* Please refer to the original paper of Vanilla AutoEncoder [here](https://web.stanford.edu/class/psych209a/ReadingsByDate/02_06/PDPVolIChapter8.pdf) and [here](https://www.aaai.org/Papers/AAAI/1987/AAAI87-050.pdf) for more information.

### Implementation Notes
* **Note 1**:   
Since datasets are somehow huge and painfully slow in training ,I decided to make number of units variable. If you want to run it in your PC, you can reduce or increase the number of units into any number you like. (512 is by default). For example:  
`model = vanilla_ae.Vanilla_Ae((None,height, width, channel), latent = 100, units = 512)`

* **Note 2** :   
You can also make the size of images smaller, so that it can be ran faster and doesn't take too much memories.

### Result for MNIST:   
* Learning rate = 0.0001
* Batch size = 16  
* Optimizer = Adam   
* units = 512

Epoch | Training Loss |  Validation Loss  |
:---: | :---: | :---:
1 | 0.0176 | 0.0105
10 | 0.0075 | 0.0074
20 | 0.0068| 0.0067

Epoch | True image and predicted image
:---: | :---:
1 | <img src="img/vanilla_1.png" />
10 | <img src="img/vanilla_10.png" />
20 |<img src="img/vanilla_20.png" />
