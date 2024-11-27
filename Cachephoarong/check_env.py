# Tạo file check_dependencies.py
import os
from importlib.metadata import distributions
import re

def get_imports(path):
    imports = set()
    # Đọc tất cả file .py trong project
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.py'):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Tìm tất cả câu lệnh import
                    imports.update(re.findall(r'^import\s+(\w+)', content, re.MULTILINE))
                    imports.update(re.findall(r'^from\s+(\w+)', content, re.MULTILINE))
    return imports

def get_installed_packages():
    return {dist.metadata['Name'].lower() for dist in distributions()}

if __name__ == "__main__":
    project_path = '.'  # Đường dẫn tới thư mục project
    used_imports = get_imports(project_path)
    installed_packages = get_installed_packages()
    
    print("Các thư viện đang được sử dụng:")
    print(used_imports)
    print("\nCác thư viện đã cài đặt:")
    print(installed_packages)
    
    print("\nCác thư viện có thể uninstall:")
    unused_packages = installed_packages - used_imports
    print(unused_packages)