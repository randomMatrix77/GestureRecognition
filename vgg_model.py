import torch
import numpy as np
import os

# To change download location of pretrained models
# To check the current loc :
# os.getenv("TORCH_HOME",os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch'))
# To change use : os.environ['TORCH_HOME'] = "new_download_path"

os.environ['TORCH_HOME'] = "D:/Softwares/miniconda/torch_models"
print('current location : {}'.format(os.getenv("TORCH_HOME",os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'),
                                                                         'torch'))))


################################# TRAINING BLOCK ####################################################################
#####################################################################################################################

# import import_images
# import sklearn
# import torch.nn as nn
# from torchvision import models

# train, lab_train, val, lab_val = import_images.import_data()
# lr_net = 0.0001
# epochs = 10
# vgg = models.vgg16(pretrained=True)
# 
# for name, child in vgg.features.named_children():
#     for param in child.parameters():
#         param.requires_grad = False
#
#
# in_features = vgg.classifier[6].in_features
# linear_layers = list(vgg.classifier.children())[:-1]
# linear_layers.extend([nn.Linear(in_features, 3)])
# vgg.classifier = nn.Sequential(*linear_layers)
#
# del name, child, param, linear_layers

# Check if layer is frozen or not
# for name, child in vgg.features.named_children():
#     for param in child.parameters():
#         print(child, param.requires_grad)
#
# for name, child in vgg.classifier.named_children():
#     for param in child.parameters():
#         print(child, param.requires_grad)
#
# crit_net = nn.CrossEntropyLoss()
# optim_net = torch.optim.Adam(vgg.parameters(), lr = lr_net)
#
# last_val = 0
#
# for i in range(epochs):
#
#     loss_epoch = 0
#     train_accuracy = 0
#     val_accuracy = 0
#
#     train, lab_train = sklearn.utils.shuffle(train, lab_train)
#     val, lab_val = sklearn.utils.shuffle(val, lab_val)
#
#     for j, (inp, lab) in enumerate(zip(train, lab_train)):
#
#         inp = np.expand_dims(inp, 0)
#         inp = np.hstack((inp, inp, inp))
#         inp = torch.tensor(inp, dtype= torch.float32)
#
#         out = vgg(inp)
#
#         # if lab == 3:
#         #     lab = np.array([2])
#         # if lab == 7:
#         #     lab = np.array([0])
#
#         loss = crit_net(out, torch.LongTensor(lab))
#
#         loss_epoch += loss.item()
#
#         loss.backward()
#         optim_net.step()
#
#         pred = out.argmax().detach().numpy()
#
#         if pred == lab:
#             train_accuracy += 1
#
#         if j % 300 == 0:
#             print('Loss current epoch : {}, accuracy : {}'.format(loss_epoch/(j + 1), train_accuracy/(j + 1)))
#
#         if loss_epoch/(j + 1) > 15000:
#             loss_epoch = 0
#             print('Loss value was reset!')
#
#     for (inp, lab) in zip(val, lab_val):
#
#         inp = np.expand_dims(inp, 0)
#         inp = np.hstack((inp, inp, inp))
#         inp = torch.tensor(inp, dtype=torch.float32)
#         out = vgg(inp)
#         pred = out.argmax().detach().numpy()
#
#         # if lab == 3:
#         #     lab = 2
#         # if lab == 7:
#         #     lab = 0
#
#         if pred == lab:
#             val_accuracy += 1
#
#     print('Epoch {}, Loss : {}, Train acc : {}, Val acc: {}'.format(i+1, loss_epoch/len(train),
#                                                                     train_accuracy/len(train), val_accuracy/len(val)))
#
#     if val_accuracy > last_val:
#         torch.save(vgg, 'vgg16_epoch_' + str(i+1)+'.ckpt')
#
#     last_val = val_accuracy

###################################################################################################################
###################################################################################################################


#################### RUNTIME BLOCK ################################################################################

vgg = torch.load('D:/Study/thesis/SelfDriving/vgg16_epoch_3.ckpt')

def runtime_model(img):

    img = np.expand_dims(img, 0)
    img = np.hstack((img, img, img))
    img = torch.tensor(img, dtype=torch.float32)
    out = vgg(img)
    pred = out.argmax().detach().numpy()

    return pred

####################################################################################################################

# import cv2
# def prep_img(img):
#
#     img = cv2.resize(img, (224, 224))
#     img = np.expand_dims(img, 0)
#     img = np.vstack((img, img, img))
#     img = np.expand_dims(img, 0)
#
#     return img
