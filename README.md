## Usecase :
This is my personal `faster-rcnn` module with almost all components copied from `torchvision.model.detection` module.

I just added as much comments as possible with other uncessary code removed. Hope this will help the intereseted ones understand the `faster-rcnn` model better.
  Just copy the `faster_rcnn` folder in your project folder and import model as :
```(python)
from faster_rcnn.faster_rcnn_model import fasterrcnn_resnet50_fpn
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
```
  Instead of regular : 
  `from torchvision.models.detection import fasterrcnn_resnet50_fpn`


## General Layout :
- Resnet backbone with a feature-pyramid-network (FPN)
  -  Rather than just using features from intermedite layer, FPN utilizes
    features from different intermediate layers of a backbone (in this case, resnet)
  to create more meaningful features.
- ROI Network
- RPN Network


## Setup for Development :
- git clone repo 
- pip install pre-commit
- pre-commit install
