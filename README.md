# Flogo

[![Skill Icons](https://skillicons.dev/icons?i=py,pytorch&perline=2)](https://skillicons.dev)

## Overview

Flogo is a versatile deep learning framework and Domain-Specific Language (DSL) developed on top of PyTorch. It can be used with various libraries, offering flexibility and compatibility. Our team comprises developers with prior experience in different deep learning frameworks. With Flogo, we provide an alternative approach to neural network development, infused with our unique perspective.

## Key Features

- **Expressive**: Flogo emphasizes expressiveness, allowing you to effectively convey complex model architectures and ideas in your code.
- **User-Friendly**: Flogo offers an intuitive and consistent interface, making the construction and adjustment of models straightforward.
- **Efficient**: At its core, Flogo utilizes the power and efficiency of PyTorch, ensuring optimized performance.

## Preprocessing

This section will discuss how to prepare your data for model training using Flogo. This platform utilizes a columnar data structure known as a dataframe for streamlined data manipulation. With various mappers like `OneHotMapper`, `StandardizationMapper`, `NormalizationMapper`, `GrayScaleMapper`, or `ResizeMapper`, you can transform data with ease.

Flogo has the capability to import data from different sources, including image and file formats, directly into the dataframe. This feature promotes efficient data processing and manipulation in the preprocessing stage.

A unique feature of Flogo is its timeline handling capabilities. It can import data from itl files, and perform operations like altering granularity using the using the `group_by` function. Also, you could transform a timeline into a dataframe with specified window and offset using the `to_dataframe` function, as shown below:

```python
timeline.group_by(1, DAY).to_dataframe(window=5, offset=1)
```

After shaping the dataframe to your requirements, you can convert it into a dataset structure. This structure changes from a column-centric to an entry-centric database, where each entry is composed of a list of inputs and a list of outputs. You can also set the batch size for the dataset.

To create a dataset, Flogo provides the DatasetBuilder class which is used along with a suitable caster, like `PytorchCaster()`. For instance, to create a dataset from a dataframe, set input and output columns, establish a batch size, use the following code:

```python
DatasetBuilder(PytorchCaster()).build(dataframe, input=["price_input_0'", "price_input_1'"], output=["price_output'"], batch_size=1)
```
Once you have your dataset, it is common practice to split it into three separate datasets: a training dataset, a validation dataset, and a testing dataset. The training dataset is used to train the model, the validation dataset is used to tune hyperparameters and evaluate the model's performance during training, and the testing dataset is used to assess the final model's performance after training.

Through these preprocessing methods, you can optimally ready your data for model training in Flogo.

## Structure

One of the unique aspects of Flogo is its flexible and modular structure for defining neural networks. The neural network model in Flogo is composed of sections, which can be either preprocessing sections or link sections.

### Sections
In Flogo, a neural network is composed of sections. A section represents a distinct component of a neural network that performs specific functions and transformations. It serves as a fundamental building block for constructing the overall architecture of the neural network. Sections in Flogo provide a modular and organized approach to designing neural network models. Each section is responsible for a particular aspect of the model's operations. There two types of sections:

* Processing Sections: Processing sectionsare responsible for performing specific computations and transformations on the input data. They process the data to extract features, model relationships, and capture relevant patterns. These sections can take various forms, such as convolutional sections, linear sections, residual sections, or recurrent sections.

* Link Sections: Link sections in Flogo have two primary tasks: connecting two processing sections and serving as the output of the neural network. These sections establish the connections between different parts of the model, linking the processed information from one section to the next. They play a crucial role in shaping the flow of information within the neural network. Additionally, link sections can serve as the final output layer of the model. They transform the processed data into the desired format based on the task at hand. For example, in classification tasks, a softmax link section is commonly used to produce class probabilities as the output.


### Blocks and layers

Within each section, there are blocks that encapsulate specific functionality. Each section has its own set of blocks, allowing for modular design and easy customization. For example, a convolutional section may consist of multiple convolutional blocks. Similarly, a link section may include blocks such as fully connected layers, activation layers, or other specialized layers.

Within each block, you can define various layers that perform specific operations. These layers can include linear layers, pooling layers, activation layers, and more. You can stack multiple layers within a block to create complex transformations and computations.

The modular nature of Flogo's structure allows for great flexibility and adaptability. It enables you to build architectures that suit your specific requirements and experiment with different combinations of preprocessing sections, link sections, blocks, and layers. By leveraging this unique structure, you can easily design and configure neural networks in Flogo that efficiently handle a wide range of deep learning tasks.

## Architecture

In Flogo, the ForwardArchitecture plays a crucial role in defining how information flows through the different layers of the model. It is a key component that determines the computation and transformation of data as it passes forward through the neural network.

## Training

In this phase, we will train our Flogo architecture by defining various parameters and executing the training task. During training, we need to specify the optimizer, loss function, early stopping criteria, and a validator. Once these parameters are defined, we can pass our model along with the training and validation datasets to train it for a specific number of epochs.

### Optimizer
The optimizer determines the optimization algorithm used to update the model's parameters during training. In Flogo, you can choose from various optimizers such as Adam, SGD, or others, depending on your requirements. For example, to use SGD with a learning rate of 0.01, you can define the optimizer as follows:

```python
optimizer = Optimizer(PytorchOptimizer("SGD", architecture.parameters(), 0.01))
```

### Loss function

The loss function quantifies the discrepancy between the predicted outputs and the true labels, guiding the model to minimize this discrepancy during training. Flogo provides different loss functions such as Mean Squared Error (MSE), Cross Entropy Loss, and more. For instance, to use the Mean Squared Error loss function, you can define it as follows:

```python
loss_function = Loss(PytorchLoss("MSELoss"))
```

### Early Stopping

Early stopping is a technique used to prevent overfitting by monitoring a certain metric (e.g., accuracy or loss) on a validation set during training. In Flogo, you can configure early stopping based on your preferred metric. For example, to use early stopping based on precision, you can define it as follows:

```python
early_stopping = EarlyStopping(PrecisionMonitor(90))
```
### Validator

The validator evaluates the model's performance during training by measuring specific metrics on the validation set. In Flogo, you can define a validator with a specific metric measurement. For example, to measure the loss during validation, you can define the validator as follows:

```python
validator = PytorchValidator(LossMeasurer())
```
### Training task

Once the necessary parameters are defined, we can execute the training using the `TrainingTask` class. This task combines the optimizer, loss function, validator, and early stopping criteria to train the model. We pass the desired number of epochs, the model architecture, and the training and validation datasets to the execute method. Here's an example of how to execute the training task:

```python
TrainingTask(PytorchTrainer(Optimizer(PytorchOptimizer("SGD", architecture.parameters(), 0.01)),
                                    Loss(PytorchLoss("MSELoss"))), 
                                    PytorchValidator(LossMeasurer()),
                                    EarlyStopping(PrecisionMonitor(0))).execute(epochs, 
                                                                                architecture, 
                                                                                train_dataset, 
                                                                                validation_dataset)
```

In the above example, we create an instance of the TrainingTask class with the defined optimizer, loss function, validator, and early stopping criteria. We then execute the task by providing the number of epochs, the model architecture (architecture), and the training and validation datasets (train_dataset and validation_dataset, respectively).

By customizing these parameters, you can train your Flogo architecture effectively and monitor its performance during the training process.


# Contact

(c) 2023 José Juan Hernández Gálvez 
<br>Github: https://github.com/josejuanhernandezgalvez<br> 
(c) 2023 Juan Carlos Santana Santana 
<br>Github: https://github.com/JuanCarss<br>
(c) 2023 Joel del Rosario Pérez
<br>Github: https://github.com/Joeel71<br>
