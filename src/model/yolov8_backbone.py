'''
This module is for YOLOv8 Backbone
This module includes ...
ConvBNSiLU
Bottleneck
SPPF
C2f
YOLOv8Backbone
'''
import torch
from torch import nn

class ConvBNSiLU(nn.Module):
    """
    Conv ->  BatchNorm ->  SiLU
    (fundamental unit of backbone)
    """

    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1, p: int = None):

        super().__init__()

        p = k // 2 if p is None else p

        self.block = nn.Sequential(
              nn.Conv2d(c_in, c_out, k, s, p, bias=False),
              nn.BatchNorm2d(c_out),
              nn.SiLU(inplace=True),
              )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Bottleneck(nn.Module):
    '''Bottleneck'''
    def __init__(self, c_in : int, c_out : int, shortcut : bool = True):
        super().__init__()

        self.cv1 = ConvBNSiLU(c_in, c_out, 3)
        self.cv2 = ConvBNSiLU(c_in, c_out, 3)
        self.use_skip = shortcut and (c_in == c_out)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))

        return x + out if self.use_skip else out


class SPPF(nn.Module):
    '''Spatial Pyramid Pooling Fast'''

    def __init__(self, c_in : int, c_out : int, k : int = 5):
        super().__init__()
        c_h = c_in // 2
        self.cv1 = ConvBNSiLU(c_in, c_h, 1, 1, 0)
        self.cv2 = ConvBNSiLU(c_h * 4, c_out, 1, 1, 0)
        self.pool = nn.MaxPool2d(k, stride=1, padding=k//2)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))

class C2f(nn.Module):
    '''YOLOv8 C2f block (cross stage partial + bottleneck)'''

    def __init__(self, c_in : int, c_out : int, n : int = 1, shortcut : bool = True):
        super().__init__()

        self.c = max(c_out // 2, 1)
        self.cv1 = ConvBNSiLU(c_in, 2 * self.c, 1, 1, 0)
        self.cv2 = ConvBNSiLU((2 + n) * self.c, c_out, 1, 1, 0)
        self.bottlenecks = nn.ModuleList(
                [Bottleneck(self.c, self.c, shortcut) for _ in range(n)]
                )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(b(y[-1]) for b in self.bottlenecks)
        return self.cv2(torch.cat(y, dim=1))



class YOLOv8Backbone(nn.Module):
    '''
    Backbone based on YOLOv8

    you can resize model by width_mult and depth_mult

    output : [P3, P4, P5] (stride 8, 16, 32)
    '''

    def __init__(self,
                 c_in : int = 3,
                 width_mult : float = 0.25,
                 depth_mult : float = 0.33,
                 ):
        super().__init__()

        def make_divisible(v : float, divisor : int) -> int:
            '''
            Scale a channel value to the nearest multiple of `divisor`.

            This keeps convolution and attention dimensions hardware-friendly and
            prevents invalid small channel counts by enforcing a minimum of `divisor`.

            채널 수를 `divisor`의 배수로 보정한다.

            Conv 연산과 attention 차원이 안정적으로 동작하도록 채널 수를 정렬하며,
            너무 작은 값이 들어와도 최소 `divisor` 이상이 되도록 보장한다.
           
            '''
            return max(int(round(v / divisor) * divisor), divisor)

        def ch(x : int) -> int:
            '''
            Compute the scaled channel width for a backbone layer.

            The base channel count `x` is multiplied by `width_mult`, then rounded to
            a valid divisible channel size so downstream modules such as multi-head
            attention can safely consume the feature maps.

            백본 각 레이어의 최종 채널 수를 계산한다.

            기본 채널 수 `x`에 `width_mult`를 적용한 뒤, downstream 모듈에서 사용할 수
            있도록 유효한 배수 채널 수로 보정한다.
            '''

            return make_divisible(x * width_mult, divisor=8)

        def dep(x : int) -> int:
            '''
            Compute the scaled repeat count for a backbone block.

            The base repeat count `x` is multiplied by `depth_mult`, then rounded to
            an integer while keeping at least one repeat so each stage retains its
            intended representation capacity.

            백본 블록의 반복 횟수를 계산한다.

            기본 반복 횟수 `x`에 `depth_mult`를 적용한 뒤 정수로 반올림하며,
            각 stage가 최소 한 번은 block을 수행하도록 1 이상의 값을 보장한다.
            '''
            return max(round(x * depth_mult), 1)

        self.stem = ConvBNSiLU(c_in, ch(64), 3, 2)

        self.stage1 = nn.Sequential(
                ConvBNSiLU(ch(64), ch(128), 3, 2),
                C2f(ch(128), ch(128), dep(3), True),
                )
        
        # P3
        self.stage2 = nn.Sequential(
                ConvBNSiLU(ch(128), ch(256), 3, 2),
                C2f(ch(256), ch(256), dep(6), True),
                )

        # P4
        self.stage3 = nn.Sequential(
                ConvBNSiLU(ch(256), ch(512), 3, 2),
                C2f(ch(512), ch(512), dep(6), True),
                )

        # P5
        self.stage4 = nn.Sequential(
                ConvBNSiLU(ch(512), ch(1024), 3, 2),
                C2f(ch(1024), ch(1024), dep(3), True),
                SPPF(ch(1024), ch(1024)),
                )

        self.out_channels : list[int] = [ch(256), ch(512), ch(1024)]

    def forward(self, x : torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return [p3, p4, p5]

