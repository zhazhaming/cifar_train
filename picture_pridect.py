from model.nn_model import *
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
from model.nn_model import *



# 设置预测设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using {}".format(device))

# 加载模型
model = Mynn().to(device)

if device == 'cpu':
    model.load_state_dict(torch.load("2022-05-27-16-54-14.pth",map_location=torch.device('cpu')))
else:
    model.load_state_dict(torch.load("2022-05-27-16-54-14.pth"))

# 加载图片
image_path = r'dataset\bird.jpg'
picture_size = 32

#  图片标准化
transform_BZ = transforms.Normalize(
    mean=[0.5,0.5,0.5],
    std=[0.5,0.5,0.5]
)

val_tf = transforms.Compose([transforms.Resize([picture_size,picture_size]),
                             transforms.ToTensor(),
                             transform_BZ
                             ])


def padding_black(img):
    w, h = img.size
    scale = picture_size / max(w, h)
    img_fg = img.resize([int(x) for x in [w * scale, h * scale]])
    size_fg = img_fg.size
    size_bg = picture_size
    img_bg = Image.new("RGB", (size_bg, size_bg))
    img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                          (size_bg - size_fg[1]) // 2))
    img = img_bg
    return img

img = Image.open(image_path)
img = img.convert('RGB')
# img = padding_black(img)
# print(type(img))  # 打印输出图片的类型

img_tensor = val_tf(img)
# print(type(img_tensor))  #打印输出标准化后的图片类型

# 增加图片的维度，之前是三维，模型要求四维
img_tensor = Variable(torch.unsqueeze(img_tensor,dim=0).float(),requires_grad=False).to(device)
# print(img_tensor)

# 进行数据输入和模型转换
model.eval()
with torch.no_grad():
    output_tensor = model(img_tensor)
    print(output_tensor)

    # 将输出通过softmax变为概率值
    output = torch.softmax(output_tensor,dim=1)
    # print(output)

    # 输出可能性最大的那位
    pred_value,pred_index = torch.max(output,1)
    print(pred_value)
    print(pred_index)

    # 将数据从cuda转回cpu
    if  torch.cuda.is_available() == False:
        pred_value = pred_value.detach().cpu().numpy()
        pred_index = pred_index.detach().cpu().numpy()

    # 类别标签
    # classes = ['Five','good','low','Yeah', 'Three', 'Stop', 'WoQuan', 'Love', 'click', 'Thour', 'ok']
    # classes = ['apple', 'banana', 'bitter_gourd', 'capsicum', 'orange', 'tomato']
    classes = ['plane','jiuhuche','bird','cat','deer','dog',"frog",'horse','ship','truck']
    # 输出预测
    print("预测类别：",classes[pred_index[0]],"概率为:",pred_value[0]*100,"%")