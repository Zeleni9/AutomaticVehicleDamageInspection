from fastai import *
from fastai.vision import *
from fastai.metrics import accuracy
from fastai.widgets import *
import torch


def train_classifier(plot_learning_rate_graph=False):
    path = Path('data/vehicle')
    classes = ['vehicle_damaged', 'vehicle_not_damaged']

    # Setting up seed for repetability
    np.random.seed(42)
    data = ImageDataBunch.from_folder(path, train='.', valid_pct=0.2, ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)

    # Verify dataset
    print(data.classes, data.c, len(data.train_ds), len(data.valid_ds))

    # Model training
    learn = cnn_learner(data, models.resnet34, metrics=error_rate)

    # Train for 4 epochs
    learn.fit_one_cycle(4)

    # Save weights, so we don't need to retrain (this matters when training is time consuming)
    learn.save('stage-1')

    if (plot_learning_rate_graph):
        # Must be done before calling lr_find, because we need to unfreeze layers to get learning rates for the whole model  
        learn.unfreeze()
        # Plot function for finding the best learning rate
        learn.lr_find()

        # Visualize graph where loss is depending on picked learning rate
        # The best tool to pick a good learning rate for our models
        # Here we are taking value of learning rate with the biggest fall in loss
        # in this example it would be [1e-04, 1e-03]
        learn.recorder.plot()

    # Training model 8 epochs more with learning rates ranging from 1e-04 to 1e-03
    # The idea here is that we train lower layers of the model with lower learning rates because they are pretrained on Imagenet
    # and higher layers with higher learning rate to fine tune the model for our dataset
    learn.fit_one_cycle(8, max_lr=slice(1e-4, 1e-3))
    learn.save('stage-2')

    # Show results
    learn.show_results()



if __name__ == "__main__":
    # Setup GPU to run with cuda
    torch.cuda.set_device(0)

    # Train image classifier for damage vehicle inspection
    train_classifier()
