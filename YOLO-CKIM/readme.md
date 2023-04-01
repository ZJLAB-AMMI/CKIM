# YOLO-CKIM


## Training 
You can train your own model using the following command:  
```python train.py --class=<grained> --CKIM=<CKIM_type> --logs=<model_path>```    

```<grained>``` can be 'fine' or 'coarse', representing training the fine-grained-model without CKIM or coarse-grained model with CKIM, seperately.  
```<CKIM-type>``` can be 'crisp' or 'fuzzy', means crisp-CKIM and fuzzy-CKIM, respectively.  
```<model_path>``` is the path where you wan to save your trained models.


## Testing
In ```/YOLO-CKIM/yolo.py```, please modify the ```model_path``` to the path of the trained model you want to test, and the ```class_path ``` to the path of your ground turth.  
Then, you can test your model by running ```python get_map.py --data=<data_path> --CKIM=<CKIM_type>```.  

```<data_path>``` means the path to your testing data  
```<CKIM-type>``` can be 'crisp' or 'fuzzy', representing crisp-CKIM and fuzzy CKIM, respectively.  

Results will be save in ```/YOLO-CKIM/map_out```.

Trained fine-grained model and corase-grained model can be found in ```/YOLO-CKIM/checkpoint/```.
