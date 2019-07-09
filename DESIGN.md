# Design Document

## Naming

*Sjaandi* means seer in Icelandic. Since we are trying to see what happens inside a neural net, I chose this name. I name projects by first picking a word to desribe what the project will be doing, and then putting it through Google Translate into some other language until I think that the name looks cool enought. That's why the project is called *sjaandi*, which is a working name that can be changed later.

## Functionality

After doing some research, I propose the following list of features:

1. Visualizing convolutional filters/kernels:
    - first layer filters. Since the first-layer filters are 3-channels deep, this is a very basic and simple feature.
    - deeper layer filters. This visualization is also straightforward, but not very good for providing an intuition about what the network is looking for.
    - doesn't require any training, only access to model weights
    - (1) and (2) in the PDF doc
2. Working with last fully connected layer's activations. Vector representations of each image at the end of the network can be used to cluster semantically and visually similar images together.
    - can use PCA or t-SNE to visualize fieature representation in 2D space
    - can search for similar images / duplicates
    - can search for outliers / mislabelled data
    - can cluster and plot multi-modal intraclass variability, when images labeled with the same class (e.g. *cat*) exhibit visually / semantically discitnct subgroups of images (e.g. cats in the wild vs. cats on the lap of a person)
    - doesn't require model trainign, only activations after layers we need access to for a given image set
    - (3) and (3*) in the PDF doc
3. Visualizing activations of deep layers.
    - plot activations after each conv layer
    - can also work with video / real-time in browser - if the model is small enough
    - (4) in PDF doc
4. Finding maximally activating patches for a given convolutional kernel
    - can search for images with a given feature
    - requires some math to calculate receptive field
    - helps explore what each convolutional kernel is actually looking for in the images
    - need access to layer activations
    - (5) in the PDF doc
5. Performing occlusion experiments.
    - masking a certain part of an input image and recording how that affects probability of detection of correct class
    - produce nice heatmaps
    - can work with any network, doesn't need access to activations, nor training
    - (6) in the PDF doc
6. Visualizing saliency maps
    - kind of meh visualization
    - still helps see if the model is looking at the right areas of image
    - requires backprop onto the input image space
    - (7) in the PDF doc
7. Guided backpropagation:
    - backpropagate onto image space, but with modified ReLU
    - needs access to layer activations
    - needs custom layer implementation
    - produces nice images / especially combined with maximally activating regions
    - (8) in the PDF doc
8. Gradient ascent and DeepDream-like implementations
    - makes nice pictures
    - has a lot of web apps online that do this
    - I am reluctant to focus on this as it will not stand out
    - (9) and until the end in the PDF doc

## Considerations

I could implement all features and apply them to some specific network architecture, and it would probably be a nice project, but not too useful in terms of extending to different networks.

After talking with you, I got the impression that it is important to focus on building something that can be useful to other people.

Therefore, the functionality above should be implemented as a pip-installable library. It should include the possibility of easily supporting and of the features for any custom model. Therefore, target audience would be engineers who would like to apply the features to their own model, as easily as possible. Therefore, implementation should focus on an API that would be easily incorporated in their own code.

I am not sure how to do it yet, but I think I would dig into the direction of using some kind of hook that hooks inside the model layers. I will also consider decorators that would allow to mark parts of the model and thus apply functionality described above.

## Web-App

The purpose of the web app is to provide an interactive demonstration of my skills. I will pick some of the functions described above and let users explore a model, or maybe find / group their own images in terms of visual similarity (something along the lines of feature 2 from Functionality section).

**Note**: I have very limited experience building web apps, expecially ones that are not static, so this building this would be much harder to do than I think.

## Proposed Workflow

1. Implementing selected features for a given model, bulding core code base for the features themselves
2. Implementing code for applying the features to any custom model, developing library API, packaging code, publishing to pip
3. Writing a blog post about the library, with examples / case study of a given model
4. Building a web app with interactive selected features

## Questions / Doubts

1. I don't think I should try to implement all the features listed above, it looks like some are not too relevant / interesting
2. I am not sure about the web app functionality yet
3. I think that some features described (like the feature 2) can be productized. Visually similar images search is a commercially valuable product. I know that companies work on this. Maybe I should only focus on that one instead, making it into a product and showcasing it online. For example, I have a friend who does visually similar images search and they sell it to law enforcement to help search for info on publicly available data. I also taled to a researher at Unity 3D who told me they are building this functionality in their 3D asset store to help users find visually similar assets for their projects.
    - if I pursue this path, I will need data and labels
    - one way would be to crawl Instagram and use hash tags as labels to bootstrap the data collection process and cheat a little on the labeling
4. Another product I have been thinking about that is somewhat related is visual search for products in videos that would automatically generate affiliate links. Let's say we ahve a YouTube video and the system would find all Amazon products in the video and give links for that. I think this can have huge success, but I am not sure where to start in terms of data collection and labeling. If I could build a small scale prototype that would be a very strong project to show to potential employers I think.