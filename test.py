import struct
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import binascii # 用于十六进制字符串到字节串的转换
import pandas as pd # 引入 pandas 库用于数据保存，如果未安装请先 pip install pandas
asin_arg = 15 / (64 / 2)
asin_arg = max(-1, min(1, asin_arg)) # 钳制到 [-1, 1]
print(math.degrees(math.asin(asin_arg)))