import operator
import pickle
import numpy as np
import megengine.module as M
import megengine.functional as F
from megengine import jit, tensor


class Module(M.Module):

    def forward(self, inputs):
        return super().forward(inputs)

class Helper:

    @staticmethod
    def transpose_pat(ndim, a, b):
        pat = list(range(ndim))
        pat[a], pat[b] = pat[b], pat[a]
        return pat
    

root = Module()
root.conv1 = M.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), dilation=(1, 1), groups=1, bias=False)
root.bn1 = M.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
root.relu = M.ReLU()
root.maxpool = M.MaxPool2d(kernel_size=3, stride=2, padding=1)
root.layer1 = Module()
setattr(root.layer1, "0", Module())
getattr(root.layer1, "0").conv1 = M.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, bias=False)
getattr(root.layer1, "0").bn1 = M.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
getattr(root.layer1, "0").relu = M.ReLU()
getattr(root.layer1, "0").conv2 = M.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, bias=False)
getattr(root.layer1, "0").bn2 = M.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
setattr(root.layer1, "1", Module())
getattr(root.layer1, "1").conv1 = M.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, bias=False)
getattr(root.layer1, "1").bn1 = M.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
getattr(root.layer1, "1").relu = M.ReLU()
getattr(root.layer1, "1").conv2 = M.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, bias=False)
getattr(root.layer1, "1").bn2 = M.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
root.layer2 = Module()
setattr(root.layer2, "0", Module())
getattr(root.layer2, "0").conv1 = M.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1), groups=1, bias=False)
getattr(root.layer2, "0").bn1 = M.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
getattr(root.layer2, "0").relu = M.ReLU()
getattr(root.layer2, "0").conv2 = M.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, bias=False)
getattr(root.layer2, "0").bn2 = M.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
getattr(root.layer2, "0").downsample = Module()
setattr(getattr(root.layer2, "0").downsample, "0", M.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1, bias=False))
setattr(getattr(root.layer2, "0").downsample, "1", M.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
setattr(root.layer2, "1", Module())
getattr(root.layer2, "1").conv1 = M.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, bias=False)
getattr(root.layer2, "1").bn1 = M.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
getattr(root.layer2, "1").relu = M.ReLU()
getattr(root.layer2, "1").conv2 = M.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, bias=False)
getattr(root.layer2, "1").bn2 = M.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
root.layer3 = Module()
setattr(root.layer3, "0", Module())
getattr(root.layer3, "0").conv1 = M.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1), groups=1, bias=False)
getattr(root.layer3, "0").bn1 = M.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
getattr(root.layer3, "0").relu = M.ReLU()
getattr(root.layer3, "0").conv2 = M.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, bias=False)
getattr(root.layer3, "0").bn2 = M.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
getattr(root.layer3, "0").downsample = Module()
setattr(getattr(root.layer3, "0").downsample, "0", M.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1, bias=False))
setattr(getattr(root.layer3, "0").downsample, "1", M.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
setattr(root.layer3, "1", Module())
getattr(root.layer3, "1").conv1 = M.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, bias=False)
getattr(root.layer3, "1").bn1 = M.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
getattr(root.layer3, "1").relu = M.ReLU()
getattr(root.layer3, "1").conv2 = M.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, bias=False)
getattr(root.layer3, "1").bn2 = M.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
root.layer4 = Module()
setattr(root.layer4, "0", Module())
getattr(root.layer4, "0").conv1 = M.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1), groups=1, bias=False)
getattr(root.layer4, "0").bn1 = M.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
getattr(root.layer4, "0").relu = M.ReLU()
getattr(root.layer4, "0").conv2 = M.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, bias=False)
getattr(root.layer4, "0").bn2 = M.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
getattr(root.layer4, "0").downsample = Module()
setattr(getattr(root.layer4, "0").downsample, "0", M.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1, bias=False))
setattr(getattr(root.layer4, "0").downsample, "1", M.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
setattr(root.layer4, "1", Module())
getattr(root.layer4, "1").conv1 = M.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, bias=False)
getattr(root.layer4, "1").bn1 = M.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
getattr(root.layer4, "1").relu = M.ReLU()
getattr(root.layer4, "1").conv2 = M.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, bias=False)
getattr(root.layer4, "1").bn2 = M.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
root.avgpool = M.AdaptiveAvgPool2d(oshp=(1, 1))
root.fc = M.Linear(in_features=512, out_features=1000, bias=True)

@jit.trace(capture_as_const=True)
def forward(x):
    conv1=root.conv1(x)
    bn1=root.bn1(conv1)
    relu_1=root.relu(bn1)
    maxpool=root.maxpool(relu_1)
    layer1_0_conv1=getattr(root.layer1, "0").conv1(maxpool)
    layer1_0_bn1=getattr(root.layer1, "0").bn1(layer1_0_conv1)
    layer1_0_relu=getattr(root.layer1, "0").relu(layer1_0_bn1)
    layer1_0_conv2=getattr(root.layer1, "0").conv2(layer1_0_relu)
    layer1_0_bn2=getattr(root.layer1, "0").bn2(layer1_0_conv2)
    add_1=F.add(layer1_0_bn2, maxpool)
    layer1_0_relu_1=getattr(root.layer1, "0").relu(add_1)
    layer1_1_conv1=getattr(root.layer1, "1").conv1(layer1_0_relu_1)
    layer1_1_bn1=getattr(root.layer1, "1").bn1(layer1_1_conv1)
    layer1_1_relu=getattr(root.layer1, "1").relu(layer1_1_bn1)
    layer1_1_conv2=getattr(root.layer1, "1").conv2(layer1_1_relu)
    layer1_1_bn2=getattr(root.layer1, "1").bn2(layer1_1_conv2)
    add_2=F.add(layer1_1_bn2, layer1_0_relu_1)
    layer1_1_relu_1=getattr(root.layer1, "1").relu(add_2)
    layer2_0_conv1=getattr(root.layer2, "0").conv1(layer1_1_relu_1)
    layer2_0_bn1=getattr(root.layer2, "0").bn1(layer2_0_conv1)
    layer2_0_relu=getattr(root.layer2, "0").relu(layer2_0_bn1)
    layer2_0_conv2=getattr(root.layer2, "0").conv2(layer2_0_relu)
    layer2_0_bn2=getattr(root.layer2, "0").bn2(layer2_0_conv2)
    layer2_0_downsample_0=getattr(getattr(root.layer2, "0").downsample, "0")(layer1_1_relu_1)
    layer2_0_downsample_1=getattr(getattr(root.layer2, "0").downsample, "1")(layer2_0_downsample_0)
    add_3=F.add(layer2_0_bn2, layer2_0_downsample_1)
    layer2_0_relu_1=getattr(root.layer2, "0").relu(add_3)
    layer2_1_conv1=getattr(root.layer2, "1").conv1(layer2_0_relu_1)
    layer2_1_bn1=getattr(root.layer2, "1").bn1(layer2_1_conv1)
    layer2_1_relu=getattr(root.layer2, "1").relu(layer2_1_bn1)
    layer2_1_conv2=getattr(root.layer2, "1").conv2(layer2_1_relu)
    layer2_1_bn2=getattr(root.layer2, "1").bn2(layer2_1_conv2)
    add_4=F.add(layer2_1_bn2, layer2_0_relu_1)
    layer2_1_relu_1=getattr(root.layer2, "1").relu(add_4)
    layer3_0_conv1=getattr(root.layer3, "0").conv1(layer2_1_relu_1)
    layer3_0_bn1=getattr(root.layer3, "0").bn1(layer3_0_conv1)
    layer3_0_relu=getattr(root.layer3, "0").relu(layer3_0_bn1)
    layer3_0_conv2=getattr(root.layer3, "0").conv2(layer3_0_relu)
    layer3_0_bn2=getattr(root.layer3, "0").bn2(layer3_0_conv2)
    layer3_0_downsample_0=getattr(getattr(root.layer3, "0").downsample, "0")(layer2_1_relu_1)
    layer3_0_downsample_1=getattr(getattr(root.layer3, "0").downsample, "1")(layer3_0_downsample_0)
    add_5=F.add(layer3_0_bn2, layer3_0_downsample_1)
    layer3_0_relu_1=getattr(root.layer3, "0").relu(add_5)
    layer3_1_conv1=getattr(root.layer3, "1").conv1(layer3_0_relu_1)
    layer3_1_bn1=getattr(root.layer3, "1").bn1(layer3_1_conv1)
    layer3_1_relu=getattr(root.layer3, "1").relu(layer3_1_bn1)
    layer3_1_conv2=getattr(root.layer3, "1").conv2(layer3_1_relu)
    layer3_1_bn2=getattr(root.layer3, "1").bn2(layer3_1_conv2)
    add_6=F.add(layer3_1_bn2, layer3_0_relu_1)
    layer3_1_relu_1=getattr(root.layer3, "1").relu(add_6)
    layer4_0_conv1=getattr(root.layer4, "0").conv1(layer3_1_relu_1)
    layer4_0_bn1=getattr(root.layer4, "0").bn1(layer4_0_conv1)
    layer4_0_relu=getattr(root.layer4, "0").relu(layer4_0_bn1)
    layer4_0_conv2=getattr(root.layer4, "0").conv2(layer4_0_relu)
    layer4_0_bn2=getattr(root.layer4, "0").bn2(layer4_0_conv2)
    layer4_0_downsample_0=getattr(getattr(root.layer4, "0").downsample, "0")(layer3_1_relu_1)
    layer4_0_downsample_1=getattr(getattr(root.layer4, "0").downsample, "1")(layer4_0_downsample_0)
    add_7=F.add(layer4_0_bn2, layer4_0_downsample_1)
    layer4_0_relu_1=getattr(root.layer4, "0").relu(add_7)
    layer4_1_conv1=getattr(root.layer4, "1").conv1(layer4_0_relu_1)
    layer4_1_bn1=getattr(root.layer4, "1").bn1(layer4_1_conv1)
    layer4_1_relu=getattr(root.layer4, "1").relu(layer4_1_bn1)
    layer4_1_conv2=getattr(root.layer4, "1").conv2(layer4_1_relu)
    layer4_1_bn2=getattr(root.layer4, "1").bn2(layer4_1_conv2)
    add_8=F.add(layer4_1_bn2, layer4_0_relu_1)
    layer4_1_relu_1=getattr(root.layer4, "1").relu(add_8)
    avgpool=root.avgpool(layer4_1_relu_1)
    flatten_1=F.flatten(avgpool, 1)
    fc=root.fc(flatten_1)
    return fc

with open("state.pkl", "rb") as f:
    state = pickle.load(f)
tstate = root.state_dict()
for k in tstate.keys():
    state[k] = state[k].reshape(tstate[k].shape)
root.load_state_dict(state, strict=False)
data = tensor(np.random.random([1, 3, 224, 224]).astype(np.float32))

root.eval()
ret = forward(data)
forward.dump("model.mgb", arg_names=["data"])
