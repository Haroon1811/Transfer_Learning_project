We are going to use EfficientNet.B0 model pretrained on ImageNet and using ImageNet weights. 
Customize the outputs of a pre-trained model by changing the output layer(s) to suit out problem.
Original EfficientNet_b0() comes with out_features=1000 (because 1000 classes in ImageNet), 
however our problem is thar only 3 classes are there-> so out_features=3. 
We can freeze all of the layers/parameters in the features section by setting the attribute requires_grad = False. 
(PyTorch doesn't track gradient update and in turn these parameters won't be changed by our optimizer during training). 
After training the model, we test predictions on a custom image. 
Loss function=torch.nn.CrossEntropyLoss()-> for multiclass classification problem 
Optimizer = torch.optim.Adam() with learning rate, lr=1e-3
