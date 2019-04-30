# iRSpot-SPI(Sequence Physio-chemical Integrator)
## iRSpot-SPI : Meotic Recombination (hotspots/coldspots) prediction model

A classification predictive model for descrimination of meotic recombination hotspot and coldspots.
## Getting Started
Following are list of todo, before making run of the propose model.

### Prerequisites
#####  Following are the lib need to be installed....
What things you need to install the software and how to install them
- python3.6  [follow](https://www.python.org/downloads/release/python-367/)
- keras 2.2.4 [follow](https://keras.io/)
- Flask 1.0.2 [follow](http://flask.pocoo.org/docs/0.12/installation/)
- scikit-learn 0.19.1 [follow](https://scikit-learn.org/stable/install.html)
- scipy 1.1.0 [follow](https://scipy.org/install.html)
- numpy1.15.4 [follow](https://docs.scipy.org/doc/numpy/user/install.html)
- matplotlib3.0.2 [follow](https://matplotlib.org/users/installing.html#building-on-windows/)
- tensorflow 1.12.0 [follow](https://www.tensorflow.org/hub/installation)

```
$pip install <lib_name>
```

### Training a model.

```
Training.py
```
training.py is used for training the model, and saving the model and model weights for future pretrained model prediction via model.py


### Making prediction.

```
model.py
```
model.py is used to use the pretrained model, and make prediction on the unseen .fasta sequence.
These prediction are saved in a file with a sequence header.

![prediction](https://user-images.githubusercontent.com/29139858/48424825-7941bb00-e79e-11e8-9567-45e386d02c3d.JPG)



# License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Inspiration
* etc.. and all manuscript proofreading, and other encouragment... 

:EMOJICODE:
@All :+1: Farman Ali (njust), Yasir, Izhar

