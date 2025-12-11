import numpy as np
import random
from scipy.sparse import coo_matrix
import os
import csv

def generate_36x36_mtx_matrix(density=0.3, symmetric=False, output_file='matrix_36x36.mtx'):
    """
    生成36x36大小的MTX格式矩阵
    
    参数:
    density: 矩阵的非零元素密度 (0.0到1.0之间)
    symmetric: 是否生成对称矩阵
    output_file: 输出文件名
    """
    
    n = 36  # 矩阵大小
    nnz = int(n * n * density)  # 非零元素数量
    
    # 生成随机行索引、列索引和值
    if symmetric:
        # 对于对称矩阵，只需要生成上三角部分
        rows = []
        cols = []
        for i in range(n):
            for j in range(i, n):
                if random.random() < density:
                    rows.append(i)
                    cols.append(j)
        
        # 确保有足够的非零元素
        while len(rows) < max(nnz, n):  # 至少保证每个对角线上有一个元素
            i = random.randint(0, n-1)
            j = random.randint(i, n-1)
            if (i, j) not in zip(rows, cols):
                rows.append(i)
                cols.append(j)
        
        nnz = len(rows)
        data = [random.uniform(0, 10) for _ in range(nnz)]
        
        # 创建对称矩阵（复制上三角到下三角）
        rows_sym = rows + [cols[i] for i in range(nnz) if rows[i] != cols[i]]
        cols_sym = cols + [rows[i] for i in range(nnz) if rows[i] != cols[i]]
        data_sym = data + [data[i] for i in range(nnz) if rows[i] != cols[i]]
        
        rows, cols, data = rows_sym, cols_sym, data_sym
        nnz = len(data)
    else:
        # 生成非零元素位置（避免重复）
        positions = set()
        rows = []
        cols = []
        
        # 确保对角线至少有1个非零元素
        for i in range(n):
            positions.add((i, i))
            rows.append(i)
            cols.append(i)
        
        # 添加其他非零元素
        while len(positions) < nnz:
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if (i, j) not in positions:
                positions.add((i, j))
                rows.append(i)
                cols.append(j)
        
        nnz = len(positions)
        data = [random.uniform(0, 10) for _ in range(nnz)]
    
    # 创建稀疏矩阵
    matrix = coo_matrix((data, (rows, cols)), shape=(n, n))
    
    # 写入MTX文件
    with open(output_file, 'w') as f:
        # MTX文件头
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write("% Generated 36x36 matrix\n")
        f.write(f"{n} {n} {nnz}\n")
        
        # 写入数据（行和列从1开始索引）
        for i in range(nnz):
            f.write(f"{rows[i]+1} {cols[i]+1} {data[i]:.6f}\n")
    
    print(f"MTX矩阵已生成并保存到: {output_file}")
    print(f"矩阵大小: {n} x {n}")
    print(f"非零元素数量: {nnz}")
    print(f"非零元素密度: {nnz/(n*n):.3f}")
    
    # 创建稠密矩阵用于CSV输出
    matrix_dense = np.zeros((n, n))
    for r, c, v in zip(rows, cols, data):
        matrix_dense[r, c] = v
    
    return matrix, output_file, matrix_dense

def generate_specific_pattern_mtx(pattern_type='diagonal', output_file='pattern_matrix.mtx'):
    """
    生成特定模式的36x36矩阵
    
    参数:
    pattern_type: 矩阵模式类型
        - 'diagonal': 对角矩阵
        - 'tridiagonal': 三对角矩阵
        - 'banded': 带状矩阵
        - 'random': 随机稀疏矩阵
    output_file: 输出文件名
    """
    
    n = 36
    rows = []
    cols = []
    data = []
    
    if pattern_type == 'diagonal':
        # 对角矩阵
        for i in range(n):
            rows.append(i)
            cols.append(i)
            data.append(random.uniform(1, 10))
        nnz = n
    
    elif pattern_type == 'tridiagonal':
        # 三对角矩阵
        for i in range(n):
            # 主对角线
            rows.append(i)
            cols.append(i)
            data.append(random.uniform(5, 10))
            
            # 上次对角线
            if i > 0:
                rows.append(i)
                cols.append(i-1)
                data.append(random.uniform(1, 3))
            
            # 下次对角线
            if i < n-1:
                rows.append(i)
                cols.append(i+1)
                data.append(random.uniform(1, 3))
        nnz = len(data)
    
    elif pattern_type == 'banded':
        # 带状矩阵（带宽为5）
        bandwidth = 5
        for i in range(n):
            for j in range(max(0, i-bandwidth), min(n, i+bandwidth+1)):
                rows.append(i)
                cols.append(j)
                if i == j:
                    data.append(random.uniform(5, 10))
                else:
                    data.append(random.uniform(0.1, 2))
        nnz = len(data)
    
    elif pattern_type == 'random':
        # 随机稀疏矩阵
        return generate_36x36_mtx_matrix(density=0.2, output_file=output_file)
    
    else:
        raise ValueError("未知的模式类型")
    
    # 创建稠密矩阵用于CSV输出
    matrix_dense = np.zeros((n, n))
    for r, c, v in zip(rows, cols, data):
        matrix_dense[r, c] = v
    
    # 写入MTX文件
    with open(output_file, 'w') as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(f"% {pattern_type.capitalize()} pattern matrix\n")
        f.write(f"{n} {n} {nnz}\n")
        
        for i in range(nnz):
            f.write(f"{rows[i]+1} {cols[i]+1} {data[i]:.6f}\n")
    
    print(f"{pattern_type}模式矩阵已保存到: {output_file}")
    
    matrix = coo_matrix((data, (rows, cols)), shape=(n, n))
    return matrix, output_file, matrix_dense

def save_matrix_as_csv(matrix_dense, csv_file, precision=6, include_headers=True):
    """
    将稠密矩阵保存为CSV格式，每个值放在对应位置
    
    参数:
    matrix_dense: 稠密矩阵 (numpy array)
    csv_file: 输出的CSV文件路径
    precision: 浮点数精度
    include_headers: 是否包含行列标题
    """
    
    n = matrix_dense.shape[0]
    
    print(f"正在保存CSV文件: {csv_file}")
    print(f"矩阵大小: {n} x {n}")
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        if include_headers:
            # 第一行：列标题
            header = [''] + [f'C{i+1}' for i in range(n)]
            writer.writerow(header)
        
        # 写入每一行数据
        for i in range(n):
            if include_headers:
                # 每行开头添加行标题
                row_data = [f'R{i+1}'] + [f'{val:.{precision}f}' for val in matrix_dense[i]]
            else:
                # 不添加标题，直接写数据
                row_data = [f'{val:.{precision}f}' for val in matrix_dense[i]]
            writer.writerow(row_data)
    
    print(f"CSV文件已保存: {csv_file}")
    print(f"文件大小: {os.path.getsize(csv_file)} 字节")
    
    return csv_file

def mtx_to_dense_csv(mtx_file, csv_file=None, precision=6, include_headers=True):
    """
    将MTX文件转换为稠密格式的CSV
    
    参数:
    mtx_file: 输入的MTX文件路径
    csv_file: 输出的CSV文件路径（如果为None，则自动生成）
    precision: 浮点数精度
    include_headers: 是否包含行列标题
    """
    
    if csv_file is None:
        base_name = os.path.splitext(mtx_file)[0]
        csv_file = f"{base_name}_dense.csv"
    
    # 读取MTX文件
    with open(mtx_file, 'r') as f:
        lines = f.readlines()
    
    # 解析MTX文件
    data_lines = [line.strip() for line in lines if not line.startswith('%')]
    
    if not data_lines:
        raise ValueError("MTX文件格式错误：没有数据")
    
    # 读取矩阵维度
    nrows, ncols, nnz = map(int, data_lines[0].split())
    
    # 初始化稠密矩阵
    matrix_dense = np.zeros((nrows, ncols))
    
    # 填充非零元素
    for line in data_lines[1:]:
        if line:  # 跳过空行
            parts = line.split()
            if len(parts) >= 3:
                r = int(parts[0]) - 1  # MTX是1-based索引
                c = int(parts[1]) - 1
                v = float(parts[2])
                matrix_dense[r, c] = v
    
    print(f"正在转换MTX文件: {mtx_file}")
    print(f"矩阵大小: {nrows} x {ncols}")
    print(f"非零元素: {nnz}")
    
    # 保存为CSV
    return save_matrix_as_csv(matrix_dense, csv_file, precision, include_headers)

def display_matrix_preview(matrix_dense, rows=5, cols=5):
    """
    显示矩阵的预览（前几行和前几列）
    
    参数:
    matrix_dense: 稠密矩阵
    rows: 显示的行数
    cols: 显示的列数
    """
    
    n = matrix_dense.shape[0]
    rows = min(rows, n)
    cols = min(cols, n)
    
    print(f"\n矩阵预览 (前{rows}行, 前{cols}列):")
    print("-" * 80)
    
    # 列标题
    col_headers = [f"{'C'+str(i+1):>10}" for i in range(cols)]
    print(f"{'':>12}" + "".join(col_headers))
    
    # 行数据和标题
    for i in range(rows):
        row_header = f"R{i+1:>10}"
        row_values = [f"{matrix_dense[i, j]:10.4f}" for j in range(cols)]
        print(f"{row_header}: " + " ".join(row_values))
    
    # 显示矩阵属性
    print(f"\n矩阵属性:")
    print(f"  大小: {n} × {n}")
    print(f"  非零元素数量: {np.count_nonzero(matrix_dense)}")
    print(f"  矩阵密度: {np.count_nonzero(matrix_dense)/(n*n):.3%}")
    print(f"  最大元素: {matrix_dense.max():.4f}")
    print(f"  最小元素: {matrix_dense.min():.4f}")
    print(f"  平均元素值: {matrix_dense[matrix_dense != 0].mean():.4f}")
    
def generate_and_save_matrix(density=0.2, symmetric=False, pattern=None, 
                           precision=6, include_headers=True, preview=True):
    """
    完整流程：生成矩阵并保存为MTX和CSV格式
    
    参数:
    density: 矩阵密度
    symmetric: 是否对称
    pattern: 矩阵模式（None表示随机）
    precision: 浮点数精度
    include_headers: CSV是否包含标题
    preview: 是否显示预览
    """
    
    if pattern:
        # 生成特定模式矩阵
        if pattern == 'random':
            matrix, mtx_file, matrix_dense = generate_36x36_mtx_matrix(
                density=density, symmetric=symmetric,
                output_file=f'{pattern}_36x36.mtx'
            )
        else:
            matrix, mtx_file, matrix_dense = generate_specific_pattern_mtx(
                pattern_type=pattern,
                output_file=f'{pattern}_36x36.mtx'
            )
    else:
        # 生成随机矩阵
        matrix, mtx_file, matrix_dense = generate_36x36_mtx_matrix(
            density=density, symmetric=symmetric,
            output_file=f'random_{density}_36x36.mtx'
        )
    
    # 保存为CSV
    csv_file = mtx_file.replace('.mtx', '.csv')
    save_matrix_as_csv(matrix_dense, csv_file, precision, include_headers)
    
    # 显示预览
    if preview:
        display_matrix_preview(matrix_dense)
    
    return mtx_file, csv_file, matrix_dense

def batch_generate_matrices(configs):
    """
    批量生成多种矩阵配置
    
    参数:
    configs: 配置列表，每个配置是字典
    """
    
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n{'='*60}")
        print(f"生成矩阵 {i}/{len(configs)}")
        print(f"配置: {config}")
        print('='*60)
        
        try:
            mtx_file, csv_file, matrix = generate_and_save_matrix(**config)
            results.append({
                'config': config,
                'mtx_file': mtx_file,
                'csv_file': csv_file,
                'matrix': matrix
            })
            print(f"✓ 完成: {mtx_file}")
        except Exception as e:
            print(f"✗ 失败: {e}")
    
    return results

# 使用示例
if __name__ == "__main__":
    print("="*70)
    print("生成36x36矩阵并保存为MTX和CSV格式")
    print("CSV格式：每个值放在对应的行和列位置")
    print("="*70)
    
    # 示例1: 生成随机矩阵
    print("\n示例1: 生成随机稀疏矩阵")
    print("-"*40)
    
    mtx_file1, csv_file1, matrix1 = generate_and_save_matrix(
        density=0.15,
        symmetric=False,
        pattern=None,
        precision=4,
        include_headers=True,
        preview=True
    )
    
    # 示例2: 生成对称矩阵
    print("\n\n示例2: 生成对称矩阵")
    print("-"*40)
    
    mtx_file2, csv_file2, matrix2 = generate_and_save_matrix(
        density=0.1,
        symmetric=True,
        pattern=None,
        precision=4,
        include_headers=False,  # 不包含标题
        preview=True
    )
    
    # 示例3: 生成特定模式矩阵
    print("\n\n示例3: 生成三对角矩阵")
    print("-"*40)
    
    mtx_file3, csv_file3, matrix3 = generate_specific_pattern_mtx(
        pattern_type='tridiagonal',
        output_file='tridiagonal_36x36.mtx'
    )
    save_matrix_as_csv(matrix3, 'tridiagonal_36x36.csv', precision=4, include_headers=True)
    display_matrix_preview(matrix3)
    
    # 示例4: 批量生成多种矩阵
    print("\n\n示例4: 批量生成多种矩阵配置")
    print("-"*40)
    
    batch_configs = [
        {'density': 0.05, 'symmetric': False, 'pattern': None, 'include_headers': True},
        {'density': 0.1, 'symmetric': True, 'pattern': None, 'include_headers': False},
        {'pattern': 'diagonal', 'include_headers': True},
        {'pattern': 'banded', 'include_headers': True},
    ]
    
    results = batch_generate_matrices(batch_configs)
    
    # 显示生成的CSV文件内容示例
    print("\n\nCSV文件格式示例:")
    print("="*70)
    
    # 读取第一个CSV文件的前几行显示
    if os.path.exists(csv_file1):
        print(f"\n文件: {csv_file1}")
        print("-"*40)
        with open(csv_file1, 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i < 6:  # 只显示前6行
                    # 限制每行显示的长度
                    row_display = [cell[:8] + "..." if len(cell) > 8 else cell for cell in row[:8]]
                    print(f"行 {i+1}: {', '.join(row_display)}")
                else:
                    print("... (只显示前6行)")
                    break
    
    # 显示所有生成的文件
    print("\n\n生成的文件列表:")
    print("="*70)
    
    all_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if (file.endswith('.mtx') or file.endswith('.csv')) and '36x36' in file:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                all_files.append((file, file_size, file_path))
    
    # 按类型分组
    mtx_files = [(f, s, p) for f, s, p in all_files if f.endswith('.mtx')]
    csv_files = [(f, s, p) for f, s, p in all_files if f.endswith('.csv')]
    
    print(f"\nMTX文件 ({len(mtx_files)}个):")
    for file, size, path in sorted(mtx_files):
        print(f"  {file:<35} {size:>10,} 字节")
    
    print(f"\nCSV文件 ({len(csv_files)}个):")
    for file, size, path in sorted(csv_files):
        print(f"  {file:<35} {size:>10,} 字节")
    
    print("\n" + "="*70)
    print("使用说明:")
    print("1. CSV文件可以直接用Excel、Numbers等电子表格软件打开")
    print("2. 包含标题的CSV文件会在第一列显示行号，第一行显示列号")
    print("3. 不包含标题的CSV文件是纯数据，适合程序读取")
    print("4. 使用display_matrix_preview()函数可以预览矩阵内容")
    print("5. 使用mtx_to_dense_csv()函数可以将现有MTX文件转换为CSV")
    print("="*70)