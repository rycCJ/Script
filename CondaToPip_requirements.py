# convert_conda_to_pip.py

def convert_conda_to_pip(conda_file, pip_file):
    with open(conda_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    pip_lines = []
    for line in lines:
        line = line.strip()
        # 跳过空行和注释
        if not line or line.startswith("#"):
            continue
        # conda 格式: 包=版本=构建号
        parts = line.split("=")
        if len(parts) >= 2:
            pkg = parts[0]
            ver = parts[1]
            pip_lines.append(f"{pkg}=={ver}")
        else:
            pip_lines.append(line)

    with open(pip_file, "w", encoding="utf-8") as f:
        f.write("\n".join(pip_lines))

    print(f"转换完成 ✅ 已保存到 {pip_file}")


if __name__ == "__main__":
    convert_conda_to_pip("requirements1.txt", "requirements_pip.txt")
