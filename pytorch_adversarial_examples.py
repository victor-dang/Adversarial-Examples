%matplotlib inline
%config InlineBackend.figure_format = 'retina'
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
img = Image.open('vjai_1.jpg')

preprocess = transforms.Compose([transforms.Resize((224,224),interpolation=Image.NEAREST),
                                 #transforms.Resize(256),
                                 #transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                ])
img_tensor = preprocess(img)[None,:,:,:]
plt.imshow(img_tensor[0].detach().numpy().transpose(1,2,0))
import json 

with open('imagenet_class_index.json') as f:
    imagenet_classes = {int(i):x[1] for i,x in json.load(f).items()}
    
import torch
import torch.nn as nn
from torchvision.models import resnet152
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    
    def forward(self, x):
        return (x - self.mean.type_as(x)[None,:,None,None])/ self.std.type_as(x)[None,:,None,None]

norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def plot_figure(input_img, predict, correct_class=None, target_class=None, pars=None, img_name='Image', fig_show=1):
    #p = sess.run(probs, feed_dict={image: img})[0]
    img = input_img[0].detach().numpy().transpose(1,2,0)
    p = nn.Softmax(dim=1)(predict)[0,:]
    p = p.detach().numpy()
    predicted_id = int(p.argsort()[-1:][::-1])
    predicted_probability = p[predicted_id]
    if(fig_show):
        plt.figure()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        fig.sca(ax1)

        ax1.imshow(img)
        fig.sca(ax1)
        plt.title(img_name)
        topk = list(p.argsort()[-5:][::-1])

        #print(predicted_probability)
        topprobs = p[topk]
        barlist = ax2.bar(range(5), topprobs)
        if target_class in topk:
            barlist[topk.index(target_class)].set_color('r')
        if correct_class in topk:
            barlist[topk.index(correct_class)].set_color('g')

        plt.sca(ax2)
        plt.ylim([0, 1.1])
        plt.xticks(range(5),
                    [imagenet_classes[i][:16] for i in topk],
                    rotation=80)


        fig.subplots_adjust(bottom=0.2)
        plt.show()
        for rect in barlist:
            height = rect.get_height()
            ax2.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,'%0.3f' % float(height), ha='center', va='bottom')
        if pars:
            plt.title('epsilon: '+str(pars[0])+', learning rate: '+str(pars[1])+', iteration: '+str(pars[2])+', Elapsed time (s): '+str(pars[3]))

    return predicted_id,predicted_probability
   
  
