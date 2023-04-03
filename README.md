# CKIM
This is the Python code used to implement the CKIM assisted object detectors as described in the paper:

[**Commonsense Knowledge Assisted Deep Learning for Resource-constrained and Fine-grained Object Detection**  
Pu Zhang, Bin Liu](https://arxiv.org/abs/2303.09026)



## Abstract
In this paper, we consider fine-grained image object detection in resource-constrained cases such as edge computing. Deep learning (DL), namely learning with deep neural networks (DNNs), has become the dominating approach to object detection. To achieve accurate fine-grained detection, one needs to employ a large enough DNN model and a vast amount of data annotations, which brings a challenge for using modern DL object detectors in resource-constrained cases. To this
end, we propose an approach, which leverages commonsense knowledge to assist a coarse-grained object detector to get accurate fine-grained detection results. Specifically, we introduce a commonsense knowledge inference module (CKIM) to process coarse-grained lables given by a benchmark DL detector to produce fine-grained lables. We consider both crisp-rule and fuzzy-rule based inference in our CKIM; the latter is used to handle ambiguity in the target semantic labels. We implement our method based on several modern DL detectors, namely YOLOv4, Mobilenetv3-SSD and YOLOv7-tiny. Experiment results show that our approach outperforms benchmark detectors remarkably in terms of accuracy, model size and processing latency.


## Dependencies
Please install following essential dependencies:  
scipy==1.2.1  
numpy==1.17.0  
matplotlib==3.1.2  
opencv_python==4.1.2.30  
torch==1.2.0  
torchvision==0.4.0  
tqdm==4.60.0  
Pillow==8.2.0  
h5py==2.10.0  


## Dataset and pre-process
Please download the [CLEVR dataset](https://cs.stanford.edu/people/jcjohns/clevr/) and move image files to ```/Data```.   
Dataset with middle size objects can be generated following [clevr-dataset-gen](https://github.com/facebookresearch/clevr-dataset-gen).  
The annotations necessary for training object detection models can be found in ```/Data```.



## CKIM learning
You can derive CKIM with crisp and fuzzy implementations as follows: 
1. Run ```python /CKIM_generation/crisp_rule.py ``` for crisp-CKIM generation. The parameters of obtained rules are saved in crisp_rule.txt.  
2. Run ```python /CKIM_generation/fuzzy_rule.py ``` for fuzzy-CKIM generation. The parameters of obtained rules are saved in fuzzy_rule.txt.  


## Training 
You can train your own model using the following command:  

```python train.py --class=<grained> --CKIM=<CKIM_type> --logs=<model_path>```    

```<grained>``` can be 'fine' or 'coarse', representing training the fine-grained-model without CKIM or coarse-grained model with CKIM, seperately.  
```<CKIM-type>``` can be 'crisp' or 'fuzzy', means crisp-CKIM and fuzzy-CKIM, respectively.  
```<model_path>``` is the path where you wan to save your trained models.


## Testing
In ```/YOLO-CKIM/yolo.py```, please modify the ```model_path``` to the path of the trained model you want to test, and the ```class_path ``` to the path of your ground turth. Then, you can test your model by running:  

```python get_map.py --data=<data_path> --CKIM=<CKIM_type>```.  

```<data_path>``` means the path to your testing data  
```<CKIM-type>``` can be 'crisp' or 'fuzzy', representing crisp-CKIM and fuzzy CKIM, respectively.   
Testing results will be save in ```/YOLO-CKIM/map_out```.

Trained CKIM assisted YOLOv7-tiny model can be found in ```/YOLO-CKIM/checkpoint/coarse-grained/best_epoch_weights.pth```.



## Citation
If you find this code useful in your research then please cite  

@article{zhang2023commonsense,

  title={Commonsense Knowledge Assisted Deep Learning for Resource-constrained and Fine-grained Object Detection},
  
  author={Zhang, Pu and Liu, Bin},
  
  journal={arXiv preprint arXiv:2303.09026},
  
  year={2023}
  
}


## Acknowledgement
This code is adapted from [YOLOv4](https://github.com/bubbliiiing/yolov4-tiny-pytorch) and [YOLOv7](https://github.com/WongKinYiu/yolov7).
