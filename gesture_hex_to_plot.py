import struct
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import binascii # 用于十六进制字符串到字节串的转换
import pandas as pd # 引入 pandas 库用于数据保存，如果未安装请先 pip install pandas
plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题
# --- 协议常量定义 ---
ANGLE_SCALE = 64 # 协议中定义的角度比例因子
VELOCITY_SCALE = 8
PI = math.pi
FRAME_HEAD_VALUE = 0xAA55 # 帧头的期望值 (小端序为 0x55AA)
EXTINFO_HEAD = 0XCDAB
# 主要帧头部分 (FrameHead 到 Reserved3) 的字节大小
MAIN_FRAME_HEADER_SIZE = 12 
# 扩展信息头部分 (ExtInfo_HEAD 到 ExtInfo_ADDON) 的字节大小
EXT_INFO_HEADER_SIZE = 14 # 2字节 ExtInfo_HEAD + 4字节 ExtInfo_Version到ExtInfo_VEL + 4字节 ExtInfo_RANGE_RES到ExtInfo_VEL_RES + 4字节 ExtInfo_ADDON
TARGET_NUM_SIZE = 2 # ExData_Target_Num 的字节大小

# --- CRC16 校验函数 ---
def crc16_xmodem(data):
    """
    计算 CRC16-XMODEM (也称为 ZMODEM) 校验码。
    根据协议中指定的参数：Poly=0x1021, Init=0xC6C6, XorOut=0x0000, RefIn=True, RefOut=True。
    这个实现尝试匹配这些参数。
    """
    poly = 0x1021
    crc = 0xC6C6  # 初始值
    
    for byte in data:
        crc ^= (byte << 8) # 将字节与 CRC 的高位异或
        for _ in range(8):
            if crc & 0x8000: # 如果最高位是 1
                crc = (crc << 1) ^ poly
            else:
                crc <<= 1
        crc &= 0xFFFF # 保持 CRC 为 16 位
    
    # 协议指明 RefOut=True，但当 XorOut 为 0x0000 时，最终结果通常不需要额外位反转。
    # 这里直接返回计算出的 CRC 值。
    return crc

# --- 点数据解析函数 (格式 0) ---
def parse_point_format_0(data_buffer, ext_info):
    """
    解析格式 0 的点数据 (8 字节)。
    :param data_buffer: 包含点数据的字节串
    :param ext_info: 包含 ExtInfo 字段 (如 RANGE_RES, VELOCITY_RES) 的字典
    :return: 包含解析后的点数据 (range, velocity, angle, power_abs, x, y, z) 的字典
             如果数据不足则返回 None。
    """
    if len(data_buffer) < 8:
        print(f"错误: 格式 0 点数据不足。需要 8 字节，实际 {len(data_buffer)} 字节。")
        return None

    try:
        # '<HBBI' 表示：小端序，一个无符号短整型 (Idx1, 2字节), 两个无符号字符 (Idx2, Idx3, 各1字节), 一个无符号整型 (PowABS, 4字节)。总计 2+1+1+4 = 8 字节。
        idx1, idx2, idx3, pow_abs_raw = struct.unpack('<HBbI', data_buffer[:8])
    except struct.error as e:
        print(f"解析格式 0 点数据时发生错误: {e}。缓冲区长度: {len(data_buffer)}。")
        return None

    point_data = {}
    
    # 安全获取分辨率值，将毫米转换为米，毫米/秒转换为米/秒。如果未找到，默认为 1。
    range_res = ext_info.get('ExtInfo_RANGE_RES', 1) / 10.0 
    velocity_res = ext_info.get('ExtInfo_VEL_RES', 1) / 10.0
    velocity_scale = VELOCITY_SCALE # 协议规定 VELOCITY_SCALE 固定为 8

    # 距离计算 (Range Calculation)
    point_data['range'] = idx1 * range_res

    # 速度计算 (Velocity Calculation)
    if idx2 < (velocity_scale / 2):
        point_data['velocity'] = idx2 * velocity_res
    else:
        point_data['velocity'] = (idx2 - velocity_scale) * velocity_res

    # 角度计算 (Angle Calculation - 度)
    # 限制 asin 函数的输入范围在 [-1, 1] 之间，以避免数学域错误。

    if idx3 < (ANGLE_SCALE / 2):
        asin_arg =float(idx3/(ANGLE_SCALE/2))
        asin_arg = max(-1, min(1, asin_arg))
        point_data['angle'] = math.degrees(math.asin(asin_arg))
    else:
        asin_arg = float((idx3-ANGLE_SCALE)/(ANGLE_SCALE/2))
        asin_arg = max(-1, min(1, asin_arg))
        point_data['angle'] = math.degrees(math.asin(asin_arg))
    # asin_arg = idx3 / (ANGLE_SCALE / 2)
    # asin_arg = max(-1, min(1, asin_arg)) # 钳制到 [-1, 1]
    # point_data['angle'] = math.degrees(math.asin(asin_arg))
    
    # PowABS: U(32,16) - 高 16 位为整数部分，低 16 位为小数部分 (2^(-16))
    #point_data['power_abs'] = float(pow_abs_raw) / (2**16)

    point_data['power_abs'] = pow_abs_raw
    # 转换为笛卡尔坐标 (假设距离是径向距离，角度是方位角，Z 轴为 0 以简化)
    # 这是一种常见的简化，适用于单角度雷达且未明确提供仰角的情况。
    x_val = point_data['range'] * math.cos(math.radians(point_data['angle']))
    y_val = point_data['range'] * math.sin(math.radians(point_data['angle']))

    z_val = 0.0 # 默认 Z 轴为 0.0，因为单个角度通常意味着在 2D 平面上的投影。
    
    point_data['x'] = x_val
    point_data['y'] = y_val
    point_data['z'] = z_val
    
    return point_data

# --- 点数据解析函数 (格式 1 或 2) ---
def parse_point_format_1_or_2(data_buffer, ext_info, format_type):
    """
    解析格式 1 或 2 的点数据 (16 字节)。
    :param data_buffer: 包含点数据的字节串
    :param ext_info: 包含 ExtInfo 字段 (如 RANGE_RES, VELOCITY_RES) 的字典
    :param format_type: 1 或 2 (指示具体是哪种格式)
    :return: 包含解析后的点数据 (range, velocity, angle, power_abs, x, y, z) 的字典
             如果数据不足则返回 None。
    """
    point_size = 16 # Rangeldx, Velldx, Angleldx, PowABS (4 * 4 = 16 字节)
    if len(data_buffer) < point_size:
        # print(f"错误: 格式 {format_type} 点数据不足。需要 {point_size} 字节，实际 {len(data_buffer)} 字节。")
        return None

    try:
        # '<iiiI' 表示：小端序，三个有符号整型 (Rangeldx, Velldx, Angleldx, 各4字节), 一个无符号整型 (PowABS, 4字节)。总计 4*4 = 16 字节。
        rangeldx, velldx, angleldx, pow_abs_raw = struct.unpack('<iiiI', data_buffer[:point_size])
    except struct.error as e:
        print(f"解析格式 {format_type} 点数据时发生错误: {e}。缓冲区长度: {len(data_buffer)}。")
        return None
    
    point_data = {}
    range_res = ext_info.get('ExtInfo_RANGE_RES', 1) / 10.0 # 将毫米转换为厘米
    velocity_res = ext_info.get('ExtInfo_VEL_RES', 1) / 10.0 # 将毫米/秒转换为厘米/秒
    
    # 根据格式类型确定缩放因子
    scale_factor_range_vel_angle = 4096.0 if format_type == 1 else 409600.0

    # 距离计算 (RangeMean Calculation)
    point_data['range'] = rangeldx / scale_factor_range_vel_angle * range_res

    # 速度计算 (VelocityMean Calculation)
    point_data['velocity'] = (velldx / scale_factor_range_vel_angle) * velocity_res

    # 角度计算 (AngleMean Calculation - 度)
    asin_arg = (angleldx / scale_factor_range_vel_angle) / (ANGLE_SCALE / 2)
    asin_arg = max(-1, min(1, asin_arg)) # 钳制到 [-1, 1]
    point_data['angle'] = math.degrees(math.asin(asin_arg))

    # PowABS: U(32,16)
    # point_data['power_abs'] = float(pow_abs_raw) / (2**16)
    point_data['power_abs'] = pow_abs_raw
    # 转换为笛卡尔坐标 (假设角度是方位角，Z 轴为 0 以简化)
    x_val = point_data['range'] * math.cos(math.radians(point_data['angle']))
    y_val = point_data['range'] * math.sin(math.radians(point_data['angle']))
    z_val = 0.0 # 默认 Z 轴为 0.0
    
    point_data['x'] = x_val
    point_data['y'] = y_val
    point_data['z'] = z_val

    return point_data

# --- 帧解析主函数 ---

def parse_frame(frame_data_buffer):

    """
    从字节缓冲区中解析单个点云数据帧。
    :param frame_data_buffer: 包含潜在帧数据的字节串，应从帧头开始。
    :return: 元组 (frame_info, points, frame_total_length) 或 (None, None, None) 如果解析失败。
    """
    frame_info = {}
    points = []
    
    # 确保缓冲区足够长以获取初始帧头信息
    if len(frame_data_buffer) < MAIN_FRAME_HEADER_SIZE:
        return None, None, None

    # 解析主帧头
    try:
        frame_head, frame_length, frame_interval, reserved1, reserved2, reserved3 = \
            struct.unpack('<HIHBBH', frame_data_buffer[0:MAIN_FRAME_HEADER_SIZE])
        
        if frame_head != FRAME_HEAD_VALUE:
            # 如果帧头不匹配，则说明这不是一个有效的帧。
            return None, None, None

        frame_info['FrameHead'] = frame_head
        frame_info['FrameLength'] = frame_length
        frame_info['FrameInterval'] = frame_interval
        frame_info['Reserved1'] = reserved1
        frame_info['Reserved2'] = reserved2
        frame_info['Reserved3'] = reserved3
        
        # 计算完整帧的总长度 (包括 FrameHead 和 FrameLength)
        total_frame_len = frame_length + 6 # 2字节 FrameHead + 4字节 FrameLength
        if len(frame_data_buffer) < total_frame_len:
            # 数据不足以构成一个完整的帧
            return None, None, None 

    except struct.error as e:
        print(f"解析主帧头时发生错误: {e}")
        return None, None, None

    offset = MAIN_FRAME_HEADER_SIZE # 扩展信息头开始的偏移量
    
    # 解析扩展信息头 (ExtInfo_HEAD)
    if len(frame_data_buffer) < offset + EXT_INFO_HEADER_SIZE:
        return None, None, None
    
    try:
        ext_info_head = struct.unpack('<H', frame_data_buffer[offset:offset+2])[0]
        if ext_info_head != 0XCDAB: # ABh CDh -> CDABh 小端序
            return None, None, None
        offset += 2
        
        ext_version, ext_rsv, ext_range, ext_vel = struct.unpack('<BBBB', frame_data_buffer[offset:offset+4])
        offset += 4

        ext_range_res, ext_vel_res = struct.unpack('<HH', frame_data_buffer[offset:offset+4])
        offset += 4
        
        ext_addon = struct.unpack('<I', frame_data_buffer[offset:offset+4])[0]
        offset += 4

        frame_info['ExtInfo_HEAD'] = ext_info_head
        frame_info['ExtInfo_Version'] = ext_version
        frame_info['ExtInfo_RSV'] = ext_rsv
        frame_info['ExtInfo_RANGE'] = ext_range
        frame_info['ExtInfo_VEL'] = ext_vel
        frame_info['ExtInfo_RANGE_RES'] = ext_range_res
        frame_info['ExtInfo_VEL_RES'] = ext_vel_res
        frame_info['ExtInfo_ADDON'] = ext_addon
        
    except struct.error as e:
        print(f"解析扩展信息头时发生错误: {e}")
        return None, None, None

    # 解析目标点数据
    if len(frame_data_buffer) < offset + TARGET_NUM_SIZE:
        return None, None, None
    
    try:
        exdata_target_num = struct.unpack('<H', frame_data_buffer[offset:offset+TARGET_NUM_SIZE])[0]
        offset += TARGET_NUM_SIZE
        
        frame_info['ExData_Target_Num'] = exdata_target_num
        
        for i in range(exdata_target_num):
            # 每个点至少需要 2 字节用于 Info1 和 Info2
            if len(frame_data_buffer) < offset + 2:
                print(f"错误: 目标点 {i} 的信息 (Info1/Info2) 数据不足。偏移量 {offset}，总长度 {len(frame_data_buffer)}。")
                break # 停止处理当前帧的剩余点
            
            target_info1 = struct.unpack('<B', frame_data_buffer[offset:offset+1])[0]
            offset += 1
            target_info2 = struct.unpack('<B', frame_data_buffer[offset:offset+1])[0]
            offset += 1

            point_format = (target_info1 >> 3) & 0x07 # 提取点格式 (Info1 的第 3-5 位)

            point_data = None
            if point_format == 0:
                point_size = 8
                # 检查是否有足够的字节来解析当前点
                if len(frame_data_buffer) < offset + point_size:
                    print(f"错误: 格式 0 点数据不足。需要 {point_size} 字节，实际 {len(frame_data_buffer) - offset} 字节。")
                    break
                point_data = parse_point_format_0(frame_data_buffer[offset:], frame_info)
                if point_data:
                    offset += point_size
            elif point_format == 1 or point_format == 2:
                point_size = 16 # Rangeldx, Velldx, Angleldx, PowABS (4*4=16 字节)
                # 检查是否有足够的字节来解析当前点
                if len(frame_data_buffer) < offset + point_size:
                    print(f"错误: 格式 {point_format} 点数据不足。需要 {point_size} 字节，实际 {len(frame_data_buffer) - offset} 字节。")
                    break
                point_data = parse_point_format_1_or_2(frame_data_buffer[offset:], frame_info, point_format)
                if point_data:
                    offset += point_size
            else:
                print(f"警告: 不支持的点格式: {point_format} (第 {i} 个点)。跳过该点。")
                # 如果遇到不支持的格式，无法可靠地推进偏移量，因此停止解析当前帧的剩余点。
                break 

            if point_data:
                points.append(point_data)
            else:
                # 如果点解析失败 (例如数据不足)，也停止当前帧的解析。
                break 

    except struct.error as e:
        print(f"解析目标点数据时发生错误: {e}")
        return None, None, None

    # --- CRC 校验 (可选但推荐) ---
    # CRC 涵盖整个帧数据，但不包括 CRC 本身。
    # total_frame_len 已经包含了 FrameHead (2) 和 FrameLength (4)。
    # 如果 FrameCRC 存在，它将是 total_frame_len 的最后 2 个字节。
    # 协议中 FrameCRC 标记为“可选”。
    
    # 检查缓冲区是否足够长，以包含 FrameCRC（如果存在的话）

    # 这里选择不进行校验，因为校验最后两位总是00；
    """
    if len(frame_data_buffer) >= total_frame_len:
        # 假设如果 FrameCRC 存在，它会占据帧的最后 2 字节
        # 数据用于 CRC 计算的部分是从 FrameHead 开始，直到 CRC 字段之前。
        data_for_crc = frame_data_buffer[:total_frame_len - 2]
        
        # 实际的 CRC 值
        actual_crc = struct.unpack('<H', frame_data_buffer[total_frame_len-2:total_frame_len])[0]
        
        calculated_crc = crc16_xmodem(data_for_crc)
        
        if calculated_crc != actual_crc:
            print(f"CRC 校验失败! 计算值: {hex(calculated_crc)}, 实际值: {hex(actual_crc)}。帧头: {hex(frame_head)}。")
            # 如果 CRC 校验失败，可以选择丢弃该帧。
            # return None, None, None # 取消注释此行以丢弃 CRC 失败的帧
        else:
            print(f"CRC 校验通过。帧头: {hex(frame_head)}。")
    """
    return frame_info, points, total_frame_len

# --- 可视化函数 ---
def visualize_points(all_points_data, title="点云数据可视化"):
    """
    在 3D 散点图中可视化解析后的点云数据。
    :param all_points_data: 字典列表，每个字典代表一个解析后的点，包含 'x', 'y', 'z' 键。
    :param title: 图表的标题。
    """
    if not all_points_data:
        print("没有点数据可供可视化。")
        return

    # 过滤掉可能没有 x,y,z 坐标的点
    valid_points = [p for p in all_points_data if 'x' in p and 'y' in p and 'z' in p]
    if not valid_points:
        print("解析后没有有效的 3D 点数据可供可视化。")
        return

    xs = [p['x'] for p in valid_points]
    ys = [p['y'] for p in valid_points]
    zs = [p['z'] for p in valid_points]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xs, ys, zs, c='blue', marker='o', s=5, alpha=0.6) # s 是标记大小，alpha 是透明度

    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    ax.set_title(title)
    ax.view_init(elev=90, azim=0)
    ax.grid(True)
    
    # 自动调整坐标轴的限制，使其适应所有数据点
    if xs: # 确保列表不为空
        max_coord = max(max(xs), max(ys), max(zs), abs(min(xs)), abs(min(ys)), abs(min(zs)))
        ax.set_xlim([-max_coord, max_coord])
        ax.set_ylim([-max_coord, max_coord])
        ax.set_zlim([-max_coord, max_coord])
        # 设置等比例尺，防止图形变形
        ax.set_box_aspect([1,1,1]) # This makes axis lengths equal

    plt.show()

# --- 逐帧数据变化的可视化函数_根据一帧中功率最大的点 ---
def visualize_frame_changes_absmax(frame_by_frame_data):

    """
    绘制每一帧的平均距离和平均角度随帧数的变化。
    :param frame_by_frame_data: 一个列表，列表中的每个元素都是代表一帧的点数据列表。
                                例如: [[point1_frame1, point2_frame1], [point1_frame2, ...]]
    """
    if not frame_by_frame_data:
        print("没有逐帧数据可供可视化。")
        return

    frame_numbers = []
    ranges_of_max_abs = []  # 存储abs最大值对应的range
    angles_of_max_abs = []  # 存储abs最大值对应的angle  
    velocity_of_max_abs = []  # 存储abs最大值对应的angle  
    power_of_max_abs = []
    # avg_ranges = []  #存储距离均值
    # avg_angles = []  #存储角度均值

    # 遍历每一帧的数据
    for i, frame_points in enumerate(frame_by_frame_data):
        # 确保当前帧有数据点
        if not frame_points:
            continue

        # 1. 找到 power_abs 值最大的那个数据点
        # 使用 max() 函数和一个 lambda 表达式作为 key
        # key=lambda p: p.get('power_abs', 0) 的意思是：对于 frame_points 中的每个点 p，
        # 使用 p 的 'power_abs' 值（如果不存在则默认为0）来进行比较，从而找到最大的那个点。
        point_with_max_abs = max(frame_points, key=lambda p: p.get('power_abs', 0))

        # 2. 从找到的这个点中提取 range 和 angle
        target_velocity = point_with_max_abs.get('velocity',0)
        if(target_velocity<500):
            target_range = point_with_max_abs.get('range', 0)
            target_angle = point_with_max_abs.get('angle', 0)
            target_power = point_with_max_abs.get('power_abs',0)
            # 3. 将结果存入列表
            frame_numbers.append(i + 1)  # 帧数从 1 开始
            ranges_of_max_abs.append(target_range)
            angles_of_max_abs.append(target_angle)
            velocity_of_max_abs.append(target_velocity)
            power_of_max_abs.append(target_power)
    
    # 开始绘图
    # 创建一个包含两个子图的图窗
    fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    # sharex=True 让两个子图共享同一个 X 轴（帧数）

    # 绘制第一个子图：距离 vs. 帧数
    ax1.plot(frame_numbers, ranges_of_max_abs, marker='o', linestyle='-', color='b')
    ax1.set_ylabel('距离 (cm)')
    ax1.set_title('每一帧的平均距离和角度变化')
    ax1.grid(True)

    # 绘制第二个子图：角度 vs. 帧数
    ax2.plot(frame_numbers, angles_of_max_abs, marker='s', linestyle='--', color='r')
    ax2.set_xlabel('帧数')
    ax2.set_ylabel('角度 (度)')
    ax2.grid(True)

    # 绘制第三个子图：速度 vs. 帧数
    ax3.plot(frame_numbers, velocity_of_max_abs, marker='s', linestyle='--', color='g')
    ax3.set_xlabel('帧数')
    ax3.set_ylabel('速度（cm/s）')
    ax3.grid(True)

    # 绘制第四个子图：最大能量 vs. 帧数
    ax4.plot(frame_numbers, power_of_max_abs, marker='s', linestyle='--', color = 'purple')
    ax4.set_xlabel('点数')
    ax4.set_ylabel('功率')
    ax4.grid(True)

    # 调整布局以防止标签重叠
    plt.tight_layout()
    # 显示图表
    plt.show()
# --- 拿出所有点，排除离谱速度的点---
def visualize_frame_changes(frame_by_frame_data):
    if not frame_by_frame_data:
        print("没有逐帧数据可供可视化。")
        return
    all_points_frame_numbers = [] # 用作 X 轴
    all_range_coords = []
    all_angle_coords = []
    all_power_coords = []
    all_velocity_coords = []

    frame_num = 1
 # 遍历每一帧
    for i,frame_points in enumerate(frame_by_frame_data) :

        # 遍历当前帧中的每一个数据点
        for point in frame_points:
            target_velocity = point.get('velocity',0)
            if(target_velocity<500):
                # 安全地获取 range 和 angle，如果不存在则默认为0
                target_range = point.get('range', 0)
                target_angle = point.get('angle', 0)
                target_power = point.get('power_abs',0)
                all_points_frame_numbers.append(frame_num)
                all_range_coords.append(target_range)
                all_angle_coords.append(target_angle)
                all_velocity_coords.append(target_velocity)
                all_power_coords.append(target_power)
                frame_num += 1
              
    # 开始绘图
    # 创建一个包含两个子图的图窗
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    # sharex=True 让两个子图共享同一个 X 轴（帧数）

    # 绘制第一个子图：距离 vs. 点数
    ax1.plot(all_points_frame_numbers, all_range_coords,marker='o', alpha=0.6, color='b')
    ax1.set_ylabel('距离 (cm)')
    ax1.set_title('每一帧的平均距离和角度变化')
    ax1.grid(True)

    # 绘制第二个子图：角度 vs. 点数
    ax2.plot(all_points_frame_numbers, all_angle_coords, marker='o',  alpha=0.6, color='r')
    ax2.set_xlabel('点数')
    ax2.set_ylabel('角度 (度)')
    ax2.grid(True)

    # 绘制第三个子图：速度 vs. 点数
    ax3.plot(all_points_frame_numbers, all_velocity_coords, marker='o',  alpha=0.6, color='g')
    ax3.set_xlabel('点数')
    ax3.set_ylabel('速度（cm/s）')
    ax3.grid(True)

    # 绘制第四个子图：功率 vs. 点数
    ax4.plot(all_points_frame_numbers, all_power_coords, marker='o',  alpha=0.6, color='purple')
    ax4.set_xlabel('点数')
    ax4.set_ylabel('功率')
    ax4.grid(True)
    # 调整布局以防止标签重叠
    plt.tight_layout()
    # 显示图表
    plt.show()
# --- 数据保存函数 ---
    """
    # range (浮点数): 点的径向距离，单位为米。
    # velocity (浮点数): 点的径向速度，单位为米/秒。
    # angle (浮点数): 点的方位角，单位为度。
    # power_abs (浮点数): 点的反射强度或功率，一个归一化后的浮点值。
    # x (浮点数): 点在笛卡尔坐标系中的 X 坐标，单位为米。
    # y (浮点数): 点在笛卡尔坐标系中的 Y 坐标，单位为米。
    # z (浮点数): 点在笛卡尔坐标系中的 Z 坐标，单位为米。在你的当前代码中，z 坐标被简化地设置为 0.0，这意味着点被假定在一个二维平面上。
    """
def save_points_to_csv(points_data, file_path="parsed_points.csv"):
    """
    将解析后的点数据保存到 CSV 文件。
    :param points_data: 字典列表，每个字典代表一个解析后的点。
    :param file_path: 保存 CSV 文件的路径。
    """
    if not points_data:
        print("没有点数据可保存。")
        return

    # 将字典列表转换为 pandas DataFrame
    df = pd.DataFrame(points_data)
    
    # 重新组织列的顺序，让 x, y, z 靠前，更直观
    # 获取所有列名
    cols = df.columns.tolist()
    # 移除 'x', 'y', 'z'
    for coord in ['x', 'y', 'z']:
        if coord in cols:
            cols.remove(coord)
    # 将 'x', 'y', 'z' 放到最前面
    new_cols_order = [c for c in ['x', 'y', 'z'] if c in df.columns] + cols
    df = df[new_cols_order]

    try:
        df.to_csv(file_path, index=False, encoding='utf-8')
        print(f"解析后的 {len(points_data)} 个点数据已成功保存到 '{file_path}'。")
    except Exception as e:
        print(f"保存数据到 CSV 文件时发生错误: {e}")

def visualize_comparison(file_path1, file_path2, title="点云数据比较可视化"):
    """
    加载两个 CSV 文件中的点数据，并在一个 3D 图中进行比较可视化。
    :param file_path1: 第一个 CSV 文件的路径。
    :param file_path2: 第二个 CSV 文件的路径。
    :param title: 图表的标题。
    """
    try:
        df1 = pd.read_csv(file_path1)
        df2 = pd.read_csv(file_path2)
    except FileNotFoundError as e:
        print(f"错误: 未找到文件 '{e.filename}'。请检查文件路径是否正确。")
        return
    except Exception as e:
        print(f"读取 CSV 文件时发生错误: {e}")
        return

    print(f"数据集 1 ('{file_path1}') 包含 {len(df1)} 个点。")
    print(f"数据集 2 ('{file_path2}') 包含 {len(df2)} 个点。")

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制第一个数据集 (例如，蓝色)
    if not df1.empty and 'x' in df1.columns and 'y' in df1.columns and 'z' in df1.columns:
        ax.scatter(df1['x'], df1['y'], df1['z'], c='blue', marker='o', s=5, alpha=0.6, label=f'数据集 1 ({file_path1})')
    else:
        print(f"警告: 数据集 1 ('{file_path1}') 没有有效的 'x','y','z' 列或为空。")

    # 绘制第二个数据集 (例如，红色)
    if not df2.empty and 'x' in df2.columns and 'y' in df2.columns and 'z' in df2.columns:
        ax.scatter(df2['x'], df2['y'], df2['z'], c='red', marker='^', s=5, alpha=0.6, label=f'数据集 2 ({file_path2})')
    else:
        print(f"警告: 数据集 2 ('{file_path2}') 没有有效的 'x','y','z' 列或为空。")

    ax.set_xlabel('X (米)')
    ax.set_ylabel('Y (米)')
    ax.set_zlabel('Z (米)')
    ax.set_title(title)
    ax.grid(True)
    ax.legend() # 显示图例

    # 自动调整坐标轴的限制
    all_xs = pd.concat([df1['x'], df2['x']]).dropna()
    all_ys = pd.concat([df1['y'], df2['y']]).dropna()
    all_zs = pd.concat([df1['z'], df2['z']]).dropna()

    if not all_xs.empty:
        max_coord = max(all_xs.abs().max(), all_ys.abs().max(), all_zs.abs().max())
        ax.set_xlim([-max_coord, max_coord])
        ax.set_ylim([-max_coord, max_coord])
        ax.set_zlim([-max_coord, max_coord])
        ax.set_box_aspect([1,1,1])

    plt.show()

# --- 主程序入口 ---
def main(txt_file_path,output_csv_path):
# def main(txt_file_path):
 
    """
    主函数：读取 TXT 文件，解析点云数据，并进行可视化。
    :param txt_file_path: 您 TXT 文件的路径。
    :param output_csv_path: 输出 CSV 文件的路径。
    """
    all_parsed_points = []
    all_frames_data = [] 
    try:
        # 打开 TXT 文件并读取内容。
        # .replace(' ', '').replace('\n', '') 是为了去除文件中的所有空格和换行符，
        # 因为 binascii.unhexlify 要求十六进制字符串是连续的。
        with open(txt_file_path, 'r', encoding='utf-8') as f: # 显式指定编码，通常是 utf-8
            hex_string_data = f.read().replace(' ', '').replace('\n', '') 
        
        # 将十六进制字符串转换为字节串
        byte_data = binascii.unhexlify(hex_string_data)
        
    except FileNotFoundError:
        print(f"错误: 未找到文件 '{txt_file_path}'。请检查文件路径是否正确。")
        return
    except binascii.Error as e:
        print(f"错误: 十六进制字符串转换为字节串时发生错误: {e}。请检查 TXT 文件内容是否为有效的十六进制字符。")
        return
    except Exception as e:
        print(f"读取或转换数据时发生意外错误: {e}")
        return

    current_offset = 0
    frame_count = 0
    
    # 帧头的字节模式 (0xAA55 的小端序表示: b'\x55\xaa')
    frame_head_pattern = struct.pack('<H', FRAME_HEAD_VALUE) 

    print(f"正在尝试从 '{txt_file_path}' 解析 {len(byte_data)} 字节数据...")

    while current_offset < len(byte_data):
        # 在字节数据中搜索帧头模式
        start_index = byte_data.find(frame_head_pattern, current_offset)
        
        if start_index == -1:
            # print(f"从偏移量 {current_offset} 开始未找到更多帧头。数据结束。")
            break # 未找到更多帧，退出循环

        if start_index > current_offset:
            # 如果帧头不在当前偏移量，说明跳过了一些“垃圾”数据。
            print(f"跳过 {start_index - current_offset} 字节的冗余数据，在偏移量 {start_index} 找到下一个帧头。")

        # 尝试解析从 `start_index` 开始的帧
        frame_data_candidate = byte_data[start_index:]
        
        frame_info, points_in_frame, frame_total_length = parse_frame(frame_data_candidate)
        
        if frame_info and points_in_frame is not None and frame_total_length is not None:
            # 成功解析了一个帧
            frame_count += 1
            print(f"成功解析第 {frame_count} 帧 (起始偏移量: {start_index})。总长度: {frame_total_length} 字节。包含 {len(points_in_frame)} 个点。")
                        # === 2. 修改这里：同时填充两个列表 ===
            all_parsed_points.extend(points_in_frame) # 继续保留，用于3D可视化和CSV保存
            all_frames_data.append(points_in_frame) # 新增，保存当前帧的数据

            current_offset = start_index + frame_total_length # 更新偏移量到当前帧的末尾
            
        else:
            # 如果解析失败，则将偏移量前进一个字节，继续搜索下一个帧头。
            # 这是为了防止卡在一个无效的帧头候选上，或者处理部分损坏的帧。
            # print(f"未能解析从偏移量 {start_index} 开始的帧。前进 1 字节继续搜索。")
            current_offset = start_index + 1
            
    print(f"\n解析完成。总共解析了 {frame_count} 帧。")
    print(f"总共收集了 {len(all_parsed_points)} 个点。")
    # === 新增：保存解析后的数据到 CSV 文件 ===
    save_points_to_csv(all_parsed_points, output_csv_path)
    # 调用可视化函数
    # visualize_points(all_parsed_points, title="点云数据可视化")
    # === 调用新加的函数来显示变化图 ===
    visualize_frame_changes(all_frames_data)
                                                                                                                                                                                                                                                     
if __name__ == "__main__":

        # 示例路径，请根据您的实际情况修改！
    your_actual_txt_file_path = 'D:/Data/Origin/30°_1m.txt' 
    main(your_actual_txt_file_path,"D:/Data/CSV/30°_1m.csv") # 运行程序，使用虚拟数据文件进行测试
    # --- 【重要修改处 1】 ---
    # 请将这里的 'your_data.txt' 替换为您的实际 TXT 文件路径。
    # 例如：
    # txt_file_path = 'C:/Users/YourUser/Documents/点云数据.txt'
    # 或者如果文件在脚本同目录下：
    # txt_file_path = 'my_point_cloud_data.txt'
    # 请确保路径中的反斜杠 (\\) 使用双反斜杠 (\\) 或者使用正斜杠 (/)，
    # 或者在字符串前加 r (原始字符串)，例如 r'C:\Users\...'
    

    # --- 【可选修改处 2】 ---
    # 以下代码块用于生成一个虚拟的 TXT 文件，方便您测试代码。
    # 当您有自己的真实数据文件时，可以注释掉（或删除）这个 `with open(...)` 部分。
    # 如果您想用我的示例数据进行测试，请确保注释掉 `your_actual_txt_file_path` 的赋值行，
    # 并将 `txt_file_path` 设置为 'dummy_data.txt'。
    
    # 构造一个包含 2 个点的帧，并重复 3 次，中间加入一些随机“垃圾”数据，模拟真实串口数据
    # 这个帧包含一个格式 0 的点: Idx1=100, Idx2=32, Idx3=16, PowABS=1.0
    # Range = 100 * 0.109m = 10.9m
    # Angle = asin(16/(64/2)) = asin(16/32) = asin(0.5) = 30 degrees
    # dummy_frame_content_hex = (
    #     "55 AA " +         # FrameHead (0xAA55)
    #     "12 00 00 00 " +   # FrameLength (18 bytes payload) -> Total Frame = 18+6=24 bytes
    #     "05 00 " +         # FrameInterval (5ms)
    #     "01 01 " +         # Reserved1, Reserved2
    #     "00 00 " +         # Reserved3
    #     "AB CD " +         # ExtInfo_HEAD (0xABCD)
    #     "12 00 " +         # ExtInfo_Version (v12), ExtInfo_RSV
    #     "20 08 " +         # ExtInfo_RANGE (32), ExtInfo_VEL (8)
    #     "6D 00 " +         # ExtInfo_RANGE_RES (109mm)
    #     "5E 01 " +         # ExtInfo_VEL_RES (350mm/s)
    #     "00 00 00 00 " +   # ExtInfo_ADDON
    #     "01 00 " +         # ExData_Target_Num (1 point)
    #     "08 00 " +         # Target_UNIT_Info1 (Format 0), Target_UNIT_Info2
    #     "00 64 20 10 00 00 01 00" # Point Data (Format 0: Idx1=100, Idx2=32, Idx3=16, PowABS=1.0)
    # )
    
    # # 模拟数据流，包含重复帧和一些无关字节
    # test_data_for_dummy_file = (
    #     "FF FF FF " + # 一些“垃圾”数据
    #     dummy_frame_content_hex + # 第一个完整帧
    #     "01 02 03 04 05 " + # 更多“垃圾”数据
    #     dummy_frame_content_hex + # 第二个完整帧
    #     "EE EE EE EE " + # 更多的“垃圾”数据
    #     dummy_frame_content_hex + # 第三个完整帧
    #     "00 00 " # 末尾的一些字节
    # )
    
    # dummy_file_name = 'dummy_data.txt'
    # with open(dummy_file_name, 'w', encoding='utf-8') as f:
    #     f.write(test_data_for_dummy_file)
    # print(f"已创建虚拟数据文件 '{dummy_file_name}' 用于测试。")
