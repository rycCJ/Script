import serial
import struct
import os
from serial.tools import list_ports
from scipy.io import savemat
import numpy as np
# 雷达参数
#  UDFDEF_CHIRP_NUM                    ( 8 )
#  UDFDEF_VEL_NUM                      ( 16 )
#  UDFDEF_RANGE_NUM                      ( 32 30)
# 从数据上来看，速度维似乎是256个点，距离还不确定

# 从2dfft中获取的数据是滤过噪声的了
DEBUG_MODE = False  # 调试点这里

def select_serial_port():
    ports = list(list_ports.comports())
    if not ports:
        input("未检测到可用串口，按回车键退出...")
        return None
    print("\n可用串口列表：")
    for i, port in enumerate(ports):
        print(f"[{i+1}] {port.device}: {port.description}")
    while True:
        try:
            choice = int(input("\n请选择要连接的串口序号 (输入0退出): "))
            if choice == 0:
                return None
            selected = ports[choice-1]
            print(f"已选择：{selected.device}")
            return selected.device
        except (ValueError, IndexError):
            print("错误：无效的输入，请重新选择！")

def select_baudrate():
    common_rates = [9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600]
    print("\n常用波特率列表：")
    for i, rate in enumerate(common_rates):
        print(f"[{i+1}] {rate}")
    print("[0] 自定义波特率")
    while True:
        try:
            choice = int(input("\n请选择波特率序号 (输入0自定义): "))
            if choice == 0:
                rate = int(input("请输入自定义波特率: "))
                return rate if rate > 0 else 115200
            return common_rates[choice-1]
        except (ValueError, IndexError):
            print("错误：无效的输入，请重新选择！")

def get_output_filename():
    default_name = "output.mat"
    print(f"\n默认保存文件名：{default_name}")
    while True:
        filename = input("请输入保存文件名（直接回车使用默认）: ").strip()
        if not filename:
            return default_name
        if not filename.lower().endswith('.mat'):
            filename += '.mat'
        if os.path.basename(filename) == filename:
            return filename
        print("错误：文件名不能包含路径符号！")

def parse_frame(data):
    try:
        # 帧头校验
        if data[0:2] != b'\x55\xAA':
            return None, 0
        
        # 解析帧长度 (4字节小端)
        frame_len = struct.unpack('<I', data[2:6])[0]
        total_frame_size = 6 + frame_len + 5 # 头6B + 数据长度
        
        if DEBUG_MODE:
            print(f"[DEBUG] 帧头位置: 0 | 帧长度值: {frame_len} | 预期总长度: {total_frame_size} | 实际数据长度: {len(data)}")
        
        # 数据完整性检查
        if len(data) < total_frame_size:
            return None, 0
        
        # 帧尾校验 (最后5字节)
        footer_start = total_frame_size - 5
        if data[footer_start:footer_start+5] != b'\x34\x34\x34\x34\x0A':
            if DEBUG_MODE:
                print(f"[DEBUG] 帧尾校验失败 @ {footer_start}: {data[footer_start:footer_start+5].hex()}")
            return None, total_frame_size
        
        # 解析点数
        num_points = struct.unpack('<H', data[26:28])[0]
        if DEBUG_MODE:
            print(f"[DEBUG] 解析到点数: {num_points}")
        
        # 解析点数据
        points = []
        pos = 28  # 数据起始位置
        
        for _ in range(num_points):
            if pos + 10 > len(data):
                break
                
            # 检查点标记
            if data[pos:pos+2] != b'\x82\x00':
                pos += 10
                continue
                
            try:
                x = struct.unpack('<h', data[pos+2:pos+4])[0]
                y = data[pos+4]
                z = struct.unpack('<i', data[pos+6:pos+10])[0]
                points.append([x, y, z])
                if DEBUG_MODE:
                    print(f"[DEBUG] 点位 {len(points)}: X={x}, Y={y}, Z={z}")
            except struct.error as e:
                if DEBUG_MODE:
                    print(f"[WARN] 坐标解析错误 @ {pos}: {str(e)}")
            
            pos += 10
        
        return points, total_frame_size
    
    except Exception as e:
        if DEBUG_MODE:
            print(f"[ERROR] 解析异常: {str(e)}")
        return None, 0

def main():
    port = select_serial_port()
    if not port: return
    
    baudrate = select_baudrate()
    if not baudrate: return
    
    try:
        ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=8,
            parity='N',
            stopbits=1,
            timeout=1
        )
        print(f"\n成功连接 {port} @ {baudrate} bps")
        print("正在等待数据... (Ctrl+C 停止)")
    except Exception as e:
        input(f"连接失败: {str(e)}\n按回车退出...")
        return

    buffer = bytearray()
    frames = []
    
    try:
        while True:
            data = ser.read(4096)
            if data:
                buffer += data
                if DEBUG_MODE:
                    print(f"[DEBUG] 收到数据块: {len(data)} 字节 | 缓冲区累计: {len(buffer)} 字节")
                
                while True:
                    # 查找帧头
                    header_pos = buffer.find(b'\x55\xAA')
                    if header_pos == -1:
                        break
                        
                    # 移除无效前缀
                    if header_pos > 0:
                        if DEBUG_MODE:
                            print(f"[DEBUG] 丢弃前置数据 {header_pos} 字节")
                        buffer = buffer[header_pos:]
                    
                    # 解析帧
                    result, consumed = parse_frame(buffer)
                    if result is None:
                        break
                        
                    # 保存有效数据
                    frames.append(np.array(result))
                    if DEBUG_MODE:
                        print(f"[DEBUG] 成功解析一帧，消耗 {consumed} 字节")
                    
                    # 移除已处理数据
                    buffer = buffer[consumed:]
                        
            else:
                if DEBUG_MODE:
                    print("[DEBUG] 无新数据...")
                
    except KeyboardInterrupt:
        pass
    
    finally:
        if frames:
            filename = get_output_filename()
            try:
                # 转换为MATLAB兼容的cell array
                cell_array = np.empty((len(frames),), dtype=object)
                for i, arr in enumerate(frames):
                    cell_array[i] = arr.astype(np.float32)
                
                # 创建MATLAB兼容的字典结构并保存
                mdic = {
                    'frames': cell_array,  # MATLAB中将看到的变量名
                    '__globals__': [],         # 必要的MATLAB元数据
                    '__header__': b'Created by Python', 
                    '__version__': '1.0'
                }
                savemat(filename, mdic)
                print(f"\n成功保存 {len(frames)} 帧数据到 {filename}")
            except Exception as e:
                print(f"保存失败: {str(e)}")
        else:
            print("\n未接收到有效数据")
        ser.close()
        input("按回车键退出...")

if __name__ == '__main__':
    main()