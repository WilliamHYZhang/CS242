The DREAM Framework: Dynamic Responsibility and Evolutionary Adaptive Modularization for State-of-the-Art Model Distillation

To run the code:

1. Install `torch`, `torchvision`, and `tdqm`.
2. Run the Python notebooks. 

`DREAM_ResNet_18.ipynb` tests the dynamic teacher/TA weighting algorithm against no distillation, standard distillation using ResNet 101 as the teacher, standard TA distillation using ResNet34 as the TA, and an equal teacher/TA weighting algorithm. We use the torchvision implementation of ResNet18.

`DREAM_ResNet_10.ipynb` tests the dynamic teacher/TA weighting algorithm against no distillation, standard distillation using ResNet 101 as the teacher, standard TA distillation using ResNet34 as the TA, and an equal teacher/TA weighting algorithm. We use a custom implementation of ResNet10.

`DREAM_ResNet_UltraMini.ipynb` tests the full DREAM framework (dynamic teacher/TA weighting algorithm and section-wise bottom-up distillation) against no distillation, standard distillation using ResNet 101 as the teacher, standard TA distillation using ResNet10 as the TA, an equal teacher/TA weighting algorithm, and the dynamic teacher/TA weighting algorithm alone.

Use the `best_model.pth` (ResNet 101), `resnet_10_tf` (ResNet 10), and `resnet_34_tf` (ResNet 34) as preloaded teacher and TA models. For best results, we recommend hosting these notebooks on a Colab A100 instance.