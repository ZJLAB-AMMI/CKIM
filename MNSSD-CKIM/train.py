import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
#from model import SSD300, MultiBoxLoss
from MobilenetV3_Small import SSD300, MultiBoxLoss
from datasets import PascalVOCDataset
from coco_utils import *
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
# for visualization
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
import torchvision.transforms.functional as FT
from PIL import ImageDraw

# Data parameters
data_folder = './'  # folder with data files
keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


# Learning parameters
checkpoint =None #  path to model checkpoint, None if none './checkpoint_ssd300.pth.tar'
batch_size = 4  # batch size 
# iterations = 120000  # number of iterations to train  120000
workers = 4  # number of workers for loading data in the DataLoader 4
print_freq = 1000  # print training status every __ batches
lr =1e-3  # learning rate
#decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

cudnn.benchmark = True

# use for visualization
writer=SummaryWriter('runs/ssd')
writer_count = 0
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def main():
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at
    # Initialize model or load checkpoint
    if checkpoint is None:
        print("checkpoint none")
        start_epoch = 0
        model = SSD300(n_classes=n_classes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():#参数迭代器
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)

        # differnet optimizer           
        # optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
        #                             lr=lr, momentum=momentum, weight_decay=weight_decay)

        # 这里会生成optimizer.param_groups,默认是一组，是一个dict,如果添加了 @params 就会变成多组，这样可以分别给不同层的参数设置不同的学习率
        # 每个dict是一个6个key的字典 {'params': [], 'lr': 0.001, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0.0005, 'nesterov': False}
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr':  lr}, {'params': not_biases}],# 这里是给biases用lr, not_biases用默认1e-2
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)  #但是这里又给所有的都设置了lr,前边那句就没用了

        #optimizer = torch.optim.SGD(params=[{'params':model.parameters(), 'lr': 2 * lr}, {'params': model.parameters}],  lr=lr, momentum=momentum, weight_decay=weight_decay) 


    else:
        print("checkpoint load")
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']


    


    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Custom dataloaders 这个是他自己写的
    train_dataset = PascalVOCDataset(data_folder,
                                     split='train',
                                     keep_difficult=keep_difficult)
                                     
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                collate_fn=train_dataset.collate_fn, num_workers=workers,#
                                               pin_memory=True)  # note that we're passing the collate function here

    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # now it is mobilenet v3,VGG paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations,
    epochs = 300
    print("epochs:",epochs)

    # # 这句可以给不同group设置初始值，但是现在都是一样的
    # for param_group in optimizer.param_groups:
    #     optimizer.param_groups[1]['lr']=lr

    print("The starting LR is %f\n" % (optimizer.param_groups[1]['lr'],))

    # 这里可以选用不同的学习率调整方法 pytorch 自带14种 https://pytorch.org/docs/stable/optim.html
    #different scheduler six way you could try
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = (epochs // 7) + 1) 
    # scheduler = ReduceLROnPlateau(optimizer,mode="min",factor=0.8,patience=15,verbose=True, threshold=0.00001, threshold_mode='rel', cooldown=2, min_lr=0, eps=1e-08)#
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, T_mult=2, eta_min=0, last_epoch=- 1, verbose=True) 

    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        # if epoch in decay_lr_at:
        #     adjust_learning_rate_epoch(optimizer,epoch)

        
        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)
        print("epoch loss:",train_loss.item())
        # scheduler.step(train_loss) # train loss 不下降了，就降低学习率     
        
        scheduler.step()
        # Save checkpoint
        
        if epoch%10==0:
            save_checkpoint(epoch, model, optimizer)


def train(train_loader, model, criterion, optimizer, epoch):

    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()
    global train_loss
    global conf_loss
    global loc_loss
    # Batches
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # if(i%200==0):
        #     adjust_learning_rate_iter(optimizer,epoch)
        #     print("batch id:",i)#([8, 3, 300, 300])
        #N=8
        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]
        
        # Forward prop.
        # 把每一层的feature也返回
        predicted_locs, predicted_scores, feats_list = model(images)  # (N, anchor_boxes_size, 4), (N, anchor_boxes_size, n_classes)

        # Loss 这里把标注信息和预测信息作为输入，算出Loss 就是MultiBoxLoss的forward,但是这部分不在模型里的
        #                 预测出来的额       预测出来的       标签     标签
        # loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar MultiBoxLoss
        # 把分开的也返回出来，可视化用
        loss, conf_loss, loc_loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar MultiBoxLoss
        train_loss = loss
        #print("training",train_loss)

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()
        # print(" loss : %d"%loss.item())
        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()
        
        # Print status
        if i % print_freq == 0:#200个iter打印一次
            print('Epoch: [{0}][{1}/{2}][{3}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),optimizer.param_groups[1]['lr'],
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
            global writer_count
            writer.add_scalar('learning rate', optimizer.param_groups[1]['lr'], global_step=writer_count)
            writer.add_scalar('loss/train loss', train_loss, global_step=writer_count)
            writer.add_scalar('loss/conf_loss', conf_loss, global_step=writer_count)
            writer.add_scalar('loss/loc_loss', loc_loss, global_step=writer_count)

            tensor_for_show = images[0]
            global mean
            global std
            tensor_for_show[0] = tensor_for_show[0]*std[0] + mean[0]
            tensor_for_show[1] = tensor_for_show[1]*std[1] + mean[1]
            tensor_for_show[2] = tensor_for_show[2]*std[2] + mean[2]
            image_for_show = transforms.ToPILImage()(tensor_for_show)
            image_for_show_det = transforms.ToPILImage()(tensor_for_show)
            draw =ImageDraw.Draw(image_for_show)
            det_draw =ImageDraw.Draw(image_for_show_det)
            # 都是300x300
            # print(image_for_show.size[0])# width
            # print(image_for_show.size[1])# height
            # 原始标签是(xmin,ymin,xmax,ymax)
            # print(len(boxes))# batch_size
            boxes_for_show = boxes[0]#第一个图片里有几个box
            labels_for_show = labels[0]
            # print(boxes_for_show.shape)
            # print(box[0,0].item())#xmin
            # print(box[0,1].item())#ymin
            # print(box[0,2].item())#xmax
            # print(box[0,3].item())#ymax
            for i in range(boxes_for_show.size(0)):#每一个box都画出来
                # 先找到类别
                label_idx = labels_for_show[i].item()
                label = rev_label_map[label_idx]
                color = label_color_map[label]
                # 画box
                box = boxes_for_show[i]
                tl_x = box[0].item() * image_for_show.size[0]
                tl_y = box[1].item() * image_for_show.size[1]
                dr_x = box[2].item() * image_for_show.size[0]
                dr_y = box[3].item() * image_for_show.size[1]
                draw.rectangle((tl_x, tl_y, dr_x, dr_y), fill=None, outline=color,width=1)
                # 画标签
                draw.text((tl_x, tl_y),label,color)

            # 使用当前模型检测
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.1, max_overlap=0.7,
                                                                                       top_k=10)
            det_boxes_for_show = det_boxes_batch[0]# 第一个图片的boxes
            det_labels_for_show = det_labels_batch[0]# 第一个图片的boxes
            det_scores_for_show = det_scores_batch[0]# 第一个图片的boxes
            for i in range(len(det_boxes_for_show)):
                det_box = det_boxes_for_show[i]
                det_label_idx = det_labels_for_show[i].item()
                det_label = rev_label_map[det_label_idx]
                det_color = label_color_map[det_label]
                det_score = det_scores_for_show[i].item()
                # 画box
                tl_x = det_box[0].item() * image_for_show.size[0]
                tl_y = det_box[1].item() * image_for_show.size[1]
                dr_x = det_box[2].item() * image_for_show.size[0]
                dr_y = det_box[3].item() * image_for_show.size[1]
                det_draw.rectangle((tl_x, tl_y, dr_x, dr_y), fill=None, outline=det_color,width=1)
                # 画标签
                det_draw.text((tl_x, tl_y),det_label,det_color)
                # 画得分
                det_draw.text((dr_x, dr_y),str(det_score),det_color)
            
            # print(len(feats_list))
            for i in range(len(feats_list)):#6个尺度的features
                layer_features = feats_list[i]
                layer_feat = layer_features[0]# 第一个图片的features
                for j in range(layer_feat.shape[0]):
                    layer_feature_0 = layer_feat[j:j+1,:,:]
                # layer_feature_0 = layer_feat[:1,:,:]
                    writer.add_image("layer_"+str(i)+"_features/"+str(j), layer_feature_0, global_step=writer_count, walltime=None, dataformats='CHW')

            tensor_for_show = FT.to_tensor(image_for_show)
            det_tensor_for_show = FT.to_tensor(image_for_show_det)
            writer.add_image("results/ground_truth", tensor_for_show, global_step=writer_count, walltime=None, dataformats='CHW')
            writer.add_image("results/detection", det_tensor_for_show, global_step=writer_count, walltime=None, dataformats='CHW')
            writer_count+=1
        #break #test
    del predicted_locs, predicted_scores, images, boxes, labels, image_for_show, tensor_for_show  # free some memory since their histories may be stored


# 这两个参数没有用到
def adjust_learning_rate_epoch(optimizer,cur_epoch):
    """
    Scale learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param scale: factor to multiply learning rate with.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.1
    print("DECAYING learning rate. The new LR is %f\n" % (optimizer.param_groups[1]['lr'],))

#warmup ,how much learning rate.
def adjust_learning_rate_iter(optimizer,cur_epoch):

    if(cur_epoch==0 or cur_epoch==1 ):
        for param_group in optimizer.param_groups:
            param_group['lr'] =param_group['lr'] +  0.0001  
            print("DECAYING learning rate iter.  The new LR is %f\n" % (optimizer.param_groups[1]['lr'],))

      


if __name__ == '__main__':
    main()
