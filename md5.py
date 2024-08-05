import hashlib  

def calculate_md5(file_path):  
    # 创建一个新的MD5 hash对象  
    md5_hash = hashlib.md5()  
    
    # 打开文件，以二进制模式读取  
    with open(file_path, "rb") as f:  
        # 分块读取文件，防止文件过大导致内存不足  
        for chunk in iter(lambda: f.read(4096), b""):  
            md5_hash.update(chunk)  
    
    # 返回MD5值，转换为十六进制格式  
    return md5_hash.hexdigest()  

# 文件路径  
file_path = "/home/zlt/Documents/SGraFormer-master/checkpoint/epoch_50.pth"  
md5_value = calculate_md5(file_path)  
print(f"MD5: {md5_value}")