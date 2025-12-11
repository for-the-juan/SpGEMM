#!/usr/bin/env python3
"""
36x36矩阵CSV文件转Bitmask工具
用法: python matrix_bitmask.py <csv文件> [选项]
"""

import numpy as np
import csv
import os
import sys
import argparse

def read_csv_to_matrix(csv_file):
    """
    读取CSV文件并返回36x36矩阵
    
    参数:
    csv_file: CSV文件路径
    
    返回:
    numpy array: 36x36矩阵
    """
    print(f"正在读取CSV文件: {os.path.basename(csv_file)}")
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
    except FileNotFoundError:
        print(f"错误: 文件 '{csv_file}' 不存在")
        sys.exit(1)
    except Exception as e:
        print(f"错误: 读取CSV文件失败 - {e}")
        sys.exit(1)
    
    if not rows:
        print("错误: CSV文件为空")
        sys.exit(1)
    
    # 判断是否包含标题行
    first_cell = rows[0][0] if rows[0] else ""
    
    if first_cell.startswith('C') or first_cell == '':
        # 包含标题，从第二行开始是数据
        print("检测到CSV文件包含标题行")
        data_rows = rows[1:]
        # 移除每行的第一个元素（行标题）
        data = [row[1:] if len(row) > 1 else [] for row in data_rows]
    else:
        # 不包含标题，直接使用所有行
        print("CSV文件不包含标题行")
        data = rows
    
    # 检查数据完整性
    if not data or len(data) == 0:
        print("错误: CSV文件中没有数据")
        sys.exit(1)
    
    # 转换为numpy数组
    try:
        matrix = np.array(data, dtype=float)
    except Exception as e:
        print(f"错误: 转换数据为矩阵失败 - {e}")
        print("请确保CSV文件包含36行36列的数值数据")
        sys.exit(1)
    
    # 验证矩阵大小
    if matrix.shape != (36, 36):
        print(f"警告: 矩阵大小是 {matrix.shape}, 期望的是 (36, 36)")
        print("程序将继续处理，但bitmask可能不是预期的36位格式")
    
    print(f"✓ 成功读取 {matrix.shape[0]}x{matrix.shape[1]} 矩阵")
    print(f"✓ 非零元素数量: {np.count_nonzero(matrix)}")
    print(f"✓ 矩阵密度: {np.count_nonzero(matrix)/(matrix.shape[0]*matrix.shape[1]):.2%}")
    
    return matrix

def generate_bitmask(matrix, threshold=1e-10, reverse_bits=False):
    """
    从矩阵生成bitmask
    
    参数:
    matrix: numpy数组
    threshold: 判断为非零元素的阈值
    reverse_bits: 是否反转位顺序（True: LSB=列0, False: MSB=列0）
    
    返回:
    bitmask_array: uint64数组
    hex_strings: 16进制字符串列表
    binary_strings: 二进制字符串列表
    """
    n_rows, n_cols = matrix.shape[0], matrix.shape[1]
    
    # 创建bitmask数组
    bitmask_array = np.zeros(n_rows, dtype=np.uint64)
    
    # 逐行处理
    for i in range(n_rows):
        row_mask = 0
        for j in range(n_cols):
            if abs(matrix[i, j]) > threshold:
                if reverse_bits:
                    # LSB对应第0列
                    row_mask |= (1 << j)
                else:
                    # MSB对应第0列（默认）
                    row_mask |= (1 << (n_cols - 1 - j))
        
        bitmask_array[i] = row_mask
    
    # 转换为16进制字符串
    hex_strings = [f"0x{val:016X}" for val in bitmask_array]
    
    # 转换为二进制字符串
    binary_strings = []
    for val in bitmask_array:
        bin_str = format(val, f'0{n_cols}b')
        # 每4位加一个空格，便于阅读
        bin_spaced = ' '.join([bin_str[i:i+4] for i in range(0, n_cols, 4)])
        binary_strings.append(bin_spaced)
    
    return bitmask_array, hex_strings, binary_strings

def print_bitmask(bitmask_array, hex_strings, binary_strings, matrix, 
                  show_binary=True, show_details=True, format="c"):
    """
    打印bitmask信息
    
    参数:
    bitmask_array: bitmask数组
    hex_strings: 16进制字符串列表
    binary_strings: 二进制字符串列表
    matrix: 原始矩阵
    show_binary: 是否显示二进制表示
    show_details: 是否显示详细信息
    format: 输出格式 ("c", "python", "hex", "raw")
    """
    n_rows, n_cols = matrix.shape[0], matrix.shape[1]
    
    if show_details:
        print("\n" + "="*80)
        print("BITMASK 输出")
        print("="*80)
        
        # 统计信息
        total_elements = n_rows * n_cols
        non_zero_count = np.count_nonzero(matrix)
        bitmask_ones = sum(bin(val).count('1') for val in bitmask_array)
        
        print(f"矩阵: {n_rows} × {n_cols} = {total_elements} 元素")
        print(f"非零元素: {non_zero_count} ({non_zero_count/total_elements:.2%})")
        print(f"Bitmask 1的位数: {bitmask_ones}")
        
        if non_zero_count != bitmask_ones:
            print(f"警告: 矩阵非零元素数 ({non_zero_count}) 与 bitmask 1的位数 ({bitmask_ones}) 不匹配")
            print(f"可能由于阈值设置或浮点精度问题")
        
        print(f"\n位顺序说明:")
        print(f"  MSB(最高位) ←→ LSB(最低位)")
        print(f"  列 0 ←→ 列 {n_cols-1}")
        print(f"  1 = 非零元素, 0 = 零元素")
    
    # 根据格式输出
    if format == "c":
        print_c_format(hex_strings)
    elif format == "python":
        print_python_format(hex_strings)
    elif format == "hex":
        print_hex_only(hex_strings)
    elif format == "raw":
        print_raw_hex(hex_strings)
    
    # 显示二进制表示（如果请求）
    if show_binary and show_details:
        print(f"\n二进制表示 (前{min(5, n_rows)}行):")
        print("-"*80)
        print(f"{'行':<4} {'16进制':<18} {'二进制 (列 0→' + str(n_cols-1) + ')':<50}")
        print(f"{'-'*4} {'-'*18} {'-'*50}")
        
        for i in range(min(5, n_rows)):
            print(f"{i:<4} {hex_strings[i]:<18} {binary_strings[i]:<50}")
        
        if n_rows > 5:
            print(f"... (仅显示前5行，共{n_rows}行)")
    
    if show_details:
        print("\n" + "="*80)

def print_c_format(hex_strings):
    """以C语言数组格式输出"""
    print(f"\n// C语言数组格式:")
    print(f"// 大小: {len(hex_strings)} 个 uint64_t")
    print(f"uint64_t bitmask[{len(hex_strings)}] = {{")
    
    for i in range(0, len(hex_strings), 4):
        row_hex = []
        for j in range(4):
            if i + j < len(hex_strings):
                row_hex.append(hex_strings[i + j])
        
        if row_hex:
            hex_line = ", ".join(row_hex)
            if i + 4 < len(hex_strings):
                print(f"    {hex_line},")
            else:
                print(f"    {hex_line}")
    
    print(f"}};")

def print_python_format(hex_strings):
    """以Python列表格式输出"""
    print(f"\n# Python列表格式:")
    print(f"# 大小: {len(hex_strings)} 个整数")
    print(f"bitmask = [")
    
    for i in range(0, len(hex_strings), 4):
        row_hex = []
        for j in range(4):
            if i + j < len(hex_strings):
                row_hex.append(hex_strings[i + j])
        
        if row_hex:
            hex_line = ", ".join(row_hex)
            if i + 4 < len(hex_strings):
                print(f"    {hex_line},")
            else:
                print(f"    {hex_line}")
    
    print(f"]")

def print_hex_only(hex_strings):
    """只输出16进制值，每行一个"""
    print(f"\n# 16进制值 (每行一个):")
    for hex_str in hex_strings:
        print(hex_str)

def print_raw_hex(hex_strings):
    """输出原始16进制值（不含0x前缀）"""
    print(f"\n# 原始16进制值 (不含0x前缀):")
    for hex_str in hex_strings:
        print(hex_str[2:])  # 移除"0x"前缀

def save_bitmask_to_file(hex_strings, output_file, format="c"):
    """
    保存bitmask到文件
    
    参数:
    hex_strings: 16进制字符串列表
    output_file: 输出文件路径
    format: 输出格式
    """
    try:
        with open(output_file, 'w') as f:
            if format == "c":
                f.write(f"// 从CSV文件生成的Bitmask\n")
                f.write(f"// 包含 {len(hex_strings)} 个 uint64_t 值\n")
                f.write(f"// 每行一个值，表示该行的非零元素位置\n")
                f.write(f"// MSB对应第0列，LSB对应最后一列\n\n")
                f.write(f"#include <stdint.h>\n\n")
                f.write(f"const uint64_t bitmask[{len(hex_strings)}] = {{\n")
                
                for i in range(0, len(hex_strings), 4):
                    row_hex = []
                    for j in range(4):
                        if i + j < len(hex_strings):
                            row_hex.append(hex_strings[i + j])
                    
                    if row_hex:
                        hex_line = ", ".join(row_hex)
                        if i + 4 < len(hex_strings):
                            f.write(f"    {hex_line},\n")
                        else:
                            f.write(f"    {hex_line}\n")
                
                f.write(f"}};\n")
                
            elif format == "python":
                f.write(f"# 从CSV文件生成的Bitmask\n")
                f.write(f"# 包含 {len(hex_strings)} 个整数\n")
                f.write(f"# 每行一个值，表示该行的非零元素位置\n")
                f.write(f"# MSB对应第0列，LSB对应最后一列\n\n")
                f.write(f"bitmask = [\n")
                
                for i in range(0, len(hex_strings), 4):
                    row_hex = []
                    for j in range(4):
                        if i + j < len(hex_strings):
                            row_hex.append(hex_strings[i + j])
                    
                    if row_hex:
                        hex_line = ", ".join(row_hex)
                        if i + 4 < len(hex_strings):
                            f.write(f"    {hex_line},\n")
                        else:
                            f.write(f"    {hex_line}\n")
                
                f.write(f"]\n")
            
            elif format == "hex":
                f.write(f"# 从CSV文件生成的Bitmask\n")
                f.write(f"# 包含 {len(hex_strings)} 个16进制值\n")
                for hex_str in hex_strings:
                    f.write(f"{hex_str}\n")
            
            elif format == "raw":
                f.write(f"# 从CSV文件生成的Bitmask\n")
                f.write(f"# 包含 {len(hex_strings)} 个原始16进制值\n")
                for hex_str in hex_strings:
                    f.write(f"{hex_str[2:]}\n")  # 移除"0x"前缀
        
        print(f"✓ Bitmask已保存到: {output_file}")
        
    except Exception as e:
        print(f"错误: 保存文件失败 - {e}")

def main():
    """主函数：解析命令行参数并执行转换"""
    
    parser = argparse.ArgumentParser(
        description='将36x36矩阵CSV文件转换为bitmask表示',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  %(prog)s matrix.csv                    # 基本使用
  %(prog)s matrix.csv -f python          # Python格式输出
  %(prog)s matrix.csv -o output.txt      # 保存到文件
  %(prog)s matrix.csv -t 0.01            # 设置阈值
  %(prog)s matrix.csv --reverse          # 反转位顺序
  %(prog)s matrix.csv --no-binary        # 不显示二进制
  %(prog)s matrix.csv --quiet            # 简洁输出
        '''
    )
    
    parser.add_argument(
        'csv_file',
        help='输入的CSV文件路径（36x36矩阵）'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='输出文件路径（如果不指定则打印到屏幕）'
    )
    
    parser.add_argument(
        '-f', '--format',
        choices=['c', 'python', 'hex', 'raw'],
        default='c',
        help='输出格式：c(C语言), python, hex(纯16进制), raw(原始16进制) (默认: c)'
    )
    
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=1e-10,
        help='非零元素判断阈值 (默认: 1e-10)'
    )
    
    parser.add_argument(
        '-r', '--reverse',
        action='store_true',
        help='反转位顺序（LSB对应第0列）'
    )
    
    parser.add_argument(
        '--no-binary',
        action='store_true',
        help='不显示二进制表示'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='简洁模式，只输出bitmask'
    )
    
    parser.add_argument(
        '-v', '--version',
        action='version',
        version='CSV转Bitmask工具 v1.0'
    )
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.csv_file):
        print(f"错误: 文件 '{args.csv_file}' 不存在")
        sys.exit(1)
    
    # 读取CSV文件
    matrix = read_csv_to_matrix(args.csv_file)
    
    # 生成bitmask
    bitmask_array, hex_strings, binary_strings = generate_bitmask(
        matrix, 
        threshold=args.threshold,
        reverse_bits=args.reverse
    )
    
    # 打印bitmask
    if args.output:
        # 保存到文件
        save_bitmask_to_file(hex_strings, args.output, args.format)
        
        if not args.quiet:
            # 同时在屏幕上显示（简洁版）
            print(f"\n已生成 {len(hex_strings)} 个bitmask值")
            print(f"格式: {args.format}")
            print(f"输出到: {args.output}")
            
            # 显示前几个值作为示例
            if len(hex_strings) > 0:
                print(f"\n示例（前3个值）:")
                for i in range(min(3, len(hex_strings))):
                    print(f"  [{i}] {hex_strings[i]}")
    else:
        # 打印到屏幕
        show_binary = not args.no_binary
        show_details = not args.quiet
        
        print_bitmask(
            bitmask_array, 
            hex_strings, 
            binary_strings, 
            matrix,
            show_binary=show_binary,
            show_details=show_details,
            format=args.format
        )
    
    # 显示完成信息
    if not args.quiet:
        print(f"\n✓ 转换完成!")
        if args.reverse:
            print(f"  位顺序: LSB对应第0列")
        else:
            print(f"  位顺序: MSB对应第0列（默认）")
        print(f"  阈值: {args.threshold}")
        print(f"  总行数: {len(hex_strings)}")

if __name__ == "__main__":
    main()