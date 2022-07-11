## About :
This is my personal `faster-rcnn` module with almost all components copied from `torchvision.model.detection` module.
Goal is to disect the components and understand the implementation details.  

I added more explicit comments with other uncessary code removed. Hope this will help the intereseted ones understand the `faster-rcnn` model better.

Just copy the `faster_rcnn` folder in your project folder and import model as :
```(python)
from faster_rcnn.faster_rcnn_model import fasterrcnn_resnet50_fpn
```
Instead of regular : 
```
from torchvision.models.detection import fasterrcnn_resnet50_fpn
```
NOTE : This implementation is slightly different original `faster-rcnn` as it utilizes a `Feature-Pyramid-Network` after the `backbone`.
Than just using feature-maps of `backbone`. 
Reference Paper : https://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf


## Setup for Development :
- git clone repo 
- pip install pre-commit
- pre-commit install
