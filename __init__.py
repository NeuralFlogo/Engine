import compiled.model.layers as compiled_layers
from compiled.model.blocks.classification import CompiledClassificationBlock
from compiled.model.blocks.flatten import CompiledFlattenBlock
from compiled.model.blocks.linear import CompiledLinearBlock
from pytorch.model.sections.link.classification import Classification
from pytorch.model.sections.link.flatten import Flatten
from pytorch.model.sections.processing.feed_forward import FeedForward

feed_forward = [
    CompiledLinearBlock(compiled_layers.linear.Linear(100, 10), compiled_layers.activation.Activation("ReLU")),
    CompiledLinearBlock(compiled_layers.linear.Linear(10, 2), compiled_layers.activation.Activation("ReLU"))]
FeedForward(feed_forward).build()

flatten = CompiledFlattenBlock(compiled_layers.flatten.Flatten(10, 8))
Flatten(flatten).build()

classification = CompiledClassificationBlock(compiled_layers.classification.Classification("Softmax", 10))
print(Classification(classification).build())

# architecture_residual = [{Kernel.Convolutional: (4, 5), Channel.In: 10, Channel.Out: 40, Activation.name: "ReLU",
#                          Kernel.Pool: (2, 2), Stride.Pool: (1, 1), Padding.Pool: (1, 1), Pooling.type: "Max",
#                          Stride.Convolutional: (2, 2), Padding.Convolutional: (1, 1)},
#                         {Kernel.Convolutional: (5, 5), Channel.In: 40, Channel.Out: 40, Activation.name: "ReLU",
#                          Block.HiddenSize: 6, Stride.Convolutional: (1, 1), Padding.Convolutional: (1, 1)},
#                          {Stride.Pool: (2, 2), Padding.Pool: (1, 1), Pooling.type: "Avg",
#                          Stride.Convolutional: (1, 1), Kernel.Pool: (2, 2)}]

# pytorch_architecture = ResNet(architecture_residual).build()
# ResidualModule(pytorch_architecture).forward(torch.zeros((10, 100, 100)))


# architecture_conv = [[{Kernel.Convolutional: (), Channel.In: 10, Channel.Out: 7, Layers.Type: "Convolutional",
#                        Stride.Convolutional: (), Padding.Convolutional: ()},
#                       {Kernel.Convolutional: (), Channel.In: 7, Channel.Out: 5, Layers.Type: "Convolutional",
#                        Stride.Convolutional: (), Padding.Convolutional: ()},
#                       {Activation.name: "ReLU", Layers.Type: "Activation"},
#                       {Kernel.Pool: (), Stride.Pool: (), Padding.Pool: (), Pooling.Type: "Max", Layers.Type: "Pool"}]]
#
# print(SimpleModel(Convolutional(architecture_conv).build()))


# architecture_recurrent = {Layers.Size: 3, Block.HiddenSize: 10, Channel.In: 4, Block.Type: "LSTMCell",
#                           Activation.name: "ReLU", Layers.Bias: True, Channel.Out: 2}
# pytorch_architecture = recurrent(architecture_recurrent).build()
# model = RnnModel(pytorch_architecture)
# model = model.forward(torch.zeros(4))
# print(mode[0])

# PATH = "/Users/jose_juan/Desktop/training_set_test"
# data_loader = images_source_type(224, 0, 1, path=PATH, batch_size=64)
#
#
# architecture_feedforward = [{Channel.In: 4096, Channel.Out: 2000, Activation.name: "ReLU"},
#                             {Channel.In: 2000, Channel.Out: 2, Activation.name: "ReLU"}]
#
# architecture_classification = {Activation.name: "Softmax", Activation.dimension: 1}
#
# architecture_link = {Dimension.Start: 1, Dimension.End: 3}
#
# pytorch_architecture = Convolutional(architecture_conv).build()
# pytorch_architecture += [Flatten(architecture_link).build()]
# pytorch_architecture += FeedForward(architecture_feedforward).build()
# pytorch_architecture += [Classification(architecture_classification).build()]
# model = SimpleModel(pytorch_architecture)
# print(model)
#
# # Training(model, data_loader, data_loader, torch.nn.CrossEntropyLoss(),
# # torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9), 10).train()
#
# num_epochs = 10
# optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
# criterion = torch.nn.CrossEntropyLoss()
#
# for epoch in range(num_epochs):  # I decided to train the model for 50 epochs
#     loss_var = 0
#
#     for idx, (images, labels) in enumerate(data_loader):
#         labels = torch.tensor(one_hot_encode(labels))
#         optimizer.zero_grad()
#         scores = model(images)
#         loss = criterion(scores, labels)
#         loss.backward()
#         optimizer.step()
#         loss_var += loss.item()
#         if idx % 64 == 0:
#             print(
#                 f'Epoch [{epoch + 1}/{num_epochs}] || Step [{idx + 1}/{len(data_loader)}] || Loss:{loss_var / len(data_loader)}')
#     print(f"Loss at epoch {epoch + 1} || {loss_var / len(data_loader)}")
#
