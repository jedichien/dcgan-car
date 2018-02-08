# WGAN - mimic automobile
This tutorial refers to [ganbook]. I love car personaly, and then decided to use GAN to generate it autonomously. I try to use normal GAN to train onece, but the result is poor, and then I give WGAN a shot, however as results seems not bad. In normal GAN will be facing gradient is the same despite the different distance between two distribution(fake and realistic) which is so called gradient vanished, however, WGAN can tackle this due to put some constraint like `1-Lipschitz` and `gradient-penalty`, and then in this tutorial we apply `1-Lipschitz` method(actually is K-Lipschitz). Enjoy this tutorial :D

## Results
![](https://media.giphy.com/media/3ohs4zS1i5ehRPny9O/giphy.gif)

## Dataset
dataset comes from [stanford cars dataset]([stanford_cars_dataset]).
* [train]([data_1])
* [test]([data_2])

## Environment (python 2.7)
1. keras-2.0 (or above
2. tensorflow-gpu


[ganbook]: https://github.com/tjwei/GANotebooks
[stanford_cars_dataset]: http://ai.stanford.edu/~jkrause/cars/car_dataset.html
[data_1]: http://imagenet.stanford.edu/internal/car196/cars_train.tgz
[data_2]: http://imagenet.stanford.edu/internal/car196/cars_test.tgz

