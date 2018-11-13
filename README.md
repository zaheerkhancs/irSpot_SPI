# iRSpot-SPI(Sequence Physio-chemical Integrator)
## iRSpot-SPI : Meotic Recombination (hotspots/coldspots) prediction model

A classification predictive model for descrimination of meotic recombination hotspot and coldspots.
## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

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

### Runing a model.

```
Training.py
```
training.py is used for training the model, and saving the model and model weights for future pretrained model prediction via model.py



```
model.py
```
model.py is used to use the pretrained model, and make prediction on the unseen .fasta sequence.
These prediction are saved in a file with a sequence header.

![prediction](https://user-images.githubusercontent.com/29139858/48424825-7941bb00-e79e-11e8-9567-45e386d02c3d.JPG)


```
app.py
```
app.py is used to initiate and launch the webserver

![prediction](https://user-images.githubusercontent.com/29139858/48427960-f708c500-e7a4-11e8-9e81-75ee2a4f63b1.JPG)






```
Copy this URL to the browser, and launch... 
```

![prediction](https://user-images.githubusercontent.com/29139858/48427967-f839f200-e7a4-11e8-81ae-129c65afc661.JPG)







```
Copy and past .fasta sequence and paste here in the textarea , and click on the prediction button.
```

![prediction](https://user-images.githubusercontent.com/29139858/48427965-f7a15b80-e7a4-11e8-8548-029a2995d92f.JPG)

And repeat

```
until finished
```
# License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc.. and all manuscript proofreading, and other encouragment... 
### Special thanks to
```
Farman Ali (njust), Yasir, Izhar
```

