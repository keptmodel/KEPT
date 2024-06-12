import os


def save_string_to_file(strings, folder_path, file_name):
    # 获取当前Python脚本的路径
    current_path = os.path.dirname(os.path.realpath(__file__))
    # 使用当前路径和文件夹路径创建完整的文件夹路径
    full_folder_path = os.path.join(current_path, folder_path)

    # 检查文件夹路径是否存在，如果不存在，创建它
    os.makedirs(full_folder_path, exist_ok=True)

    # 使用文件夹路径和文件名创建完整的文件路径
    file_path = os.path.join(full_folder_path, file_name)

    # 打开这个文件，并将字符串列表写入到这个文件中
    with open(file_path, 'w') as f:
        if isinstance(strings, list):
            for string in strings:
                f.write(string)
        else:
            f.write(strings)

def read_string_from_file(folder_path, file_name):
    # 获取当前Python脚本的路径
    current_path = os.path.dirname(os.path.realpath(__file__))
    # 使用当前路径和文件夹路径创建完整的文件夹路径
    full_folder_path = os.path.join(current_path, folder_path)
    file_path = os.path.join(full_folder_path, file_name)

    with open(file_path, 'r') as file:
        data = file.read()
    return data
def get_all_absdirpath_in_folder(folder_path):
    # 获取当前Python脚本的路径
    current_path = os.path.dirname(os.path.realpath(__file__))
    # 使用当前路径和文件夹路径创建完整的文件夹路径
    full_folder_path = os.path.join(current_path, folder_path)

    # 获取文件夹中的所有文件夹
    dirs = os.listdir(full_folder_path)
    dirs = [folder_path+'/'+dir for dir in dirs if os.path.isdir(os.path.join(full_folder_path, dir))]
    return dirs
def get_resource_path(path):
    current_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(current_path, path)
def read_all_files_in_folder(folder_path):
    # 获取当前Python脚本的路径
    current_path = os.path.dirname(os.path.realpath(__file__))
    # 使用当前路径和文件夹路径创建完整的文件夹路径
    full_folder_path = os.path.join(current_path, folder_path)

    # 获取文件夹中的所有文件
    files = os.listdir(full_folder_path)

    # 创建一个空列表来存储结果
    result = []

    # 遍历每个文件
    for file_name in files:
        # 创建完整的文件路径
        file_path = os.path.join(full_folder_path, file_name)
        # 打开并读取文件
        with open(file_path, 'r') as file:
            data = file.read()
        # 将文件名和文件内容添加到结果列表中
        result.append((file_name, data))

    return result