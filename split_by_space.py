# split_by_space.py

def split_by_space(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 把空格替换成换行
    new_content = content.replace(" ", "\n")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"✅ 处理完成，结果保存在 {output_file}")


if __name__ == "__main__":
    split_by_space("requirements1.txt", "requirements_split.txt")
