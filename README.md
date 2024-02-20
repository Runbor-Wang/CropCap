<<<<<<< HEAD
# CropCap: Embedding Visual Cross-Partition Dependency for Image Captioning
## Embedding Visual Cross-Partition Dependency
### Dependencies
- python 3.6.9
- torch 1.7.0+cuda110
- json5 0.9.5
- tqdm 4.52.0
- h5py 3.1.0
- numpy 1.19.4
- pillow 8.0.1
- torchvision 0.8.1+cuda110
- scikit-image 0.17.2
- tensorboardX 2.1
- opencv-python 4.4.0.46
### Datasets
- [MSCOCO](https://cocodataset.org/#captions-2015)
### Usage
#### Prepare dataset:
Please download spice-1.0.jar firstly, and split the data following the Karpathy’s splits for offline evaluation. 
Then, resizing all images of the MS COCO dataset into 384 × 384 for Swin-L Transformer backbone. To train the 
captioning model stably and efficiently, the dictionary of descriptive words is built by collecting the words 
that occur more than 5 times and then ends up with a vocabulary of 9, 487 words. We convert all the training 
captions to lowercase in the training process.
#### Implementation details:
The embedding size to 512 and set the number of multi-heads in all self-attention modules to 8. For training, 
cross-entropy loss for 12 epochs with a mini-batch size of 8, and an Adam optimizer whose learning rate is initialized
at 4e−4 and the warmup step is set to 20, 000. The learning rate is decayed 0.1 times and starts from the 8-th epoch. 
We decay the learning rate every 2 epoch during cross-entropy loss training. We increase the scheduled sampling 
probability by 0.05 for every 3 epochs. After the cross-entropy loss training, we train our model by optimizing 
the CIDEr score with a self-critical training strategy for another 15 epochs with an initial learning rate of 4e−5, 
which is decayed 0.1 times every 4 epochs. For testing, we use the beam search for our model with a beam size of 2. 
The default random seed is set to 42.
#### XE Train:
`python train.py --input_json --input_images --input_label_h5 --cached_tokens --batch_size beam_size --learning_rate --learning_rate_decay_every --learning_rate_decay_rate --checkpoint_path --noamopt_warmup`

#### CIDEr Train:
`python train.py --input_json --input_images --input_label_h5 --cached_tokens --batch_size --self_critical_after --beam_size --learning_rate --learning_rate_decay_every --learning_rate_decay_rate --checkpoint_path --noamopt_warmup`

#### Evaluation:
`python eval.py --model --infos_path --batch_size --language_eval --beam_size --input_json --image_root --input_label_h5 --split test `

**Note: all the path in the code should to be changed for user.**
