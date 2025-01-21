import pandas as pd
from collections import Counter
import jieba.posseg as pseg  # 用于分词和 POS 标注
import re  # 用于处理正则表达式

# 自定义停用词列表（可以扩展）
CUSTOM_STOPWORDS = {"图片", "视频", "链接", "原图", "全文", "网页链接"}


# 加载数据：读取 .txt 文件并解析为 DataFrame 格式
def load_data(file_paths):
    data = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        # 使用制表符或空格分割文本和标签
                        if '\t' in line:
                            text, label = line.rsplit('\t', 1)
                        else:
                            text, label = line.rsplit(' ', 1)
                        data.append({'text': text.strip(), 'label': int(label)})
                    except ValueError:
                        print(f"Skipping invalid line: {line}")
    return pd.DataFrame(data)


# 清理文本：移除 "#" 和 "@" 后的内容，以及指定无意义词语和特定模式
def clean_text(text):
    if not isinstance(text, str):
        return ""

    # 移除 "#" 和 "@" 后的内容
    text = re.sub(r"#\S+", "", text)  # 移除以 "#" 开头的单词（如 "#话题"）
    text = re.sub(r"@\S+", "", text)  # 移除以 "@" 开头的单词（如 "@用户"）

    # 移除类似 "xxx的微博视频" 的模式，其中 xxx 是任意字符
    text = re.sub(r"\S+的微博视频", "", text)

    return text


# 分组统计指定类别的频次，并获取高频单字
def compute_pos_statistics(df, group_column, text_column):
    pos_statistics = {}

    # 定义目标 POS 标签映射表
    target_pos_tags = {
        'Noun': ['n', 'nr', 'ns', 'nt', 'nz'],  # 名词及其子类，如人名、地名等
        'Verb': ['v', 'vd', 'vn'],  # 动词及其子类，如动名词等
        'Pronouns': ['r'],  # 代词，如“我”、“你”
        'Adjectives': ['a', 'ad', 'an'],  # 形容词及其子类，如副形容词等
        'Adverbs': ['d'],  # 副词，如“很”、“非常”
        'Prepositions': ['p'],  # 介词，如“在”、“从”
        'Conjunctions': ['c']  # 连词，如“和”、“但是”
    }

    for group in df[group_column].unique():
        group_data = df[df[group_column] == group][text_column]

        pos_counts = {pos: Counter() for pos in target_pos_tags}  # 初始化计数字典

        for text in group_data:
            if isinstance(text, str):  # 确保是字符串类型
                cleaned_text = clean_text(text)  # 清理文本

                words_with_pos = pseg.cut(cleaned_text)  # 使用 jieba.posseg 对清理后的文本进行分词和 POS 标注

                for word, tag in words_with_pos:
                    if word not in CUSTOM_STOPWORDS and word.strip():  # 排除停用词和空白字符
                        for pos_category, pos_tags in target_pos_tags.items():
                            if tag in pos_tags:  # 如果当前标注属于目标 POS 类别，则计数
                                pos_counts[pos_category][word] += 1

        total_count = sum(sum(counter.values()) for counter in pos_counts.values())

        pos_statistics[group] = {
            'pos_counts': pos_counts,
            'total_count': total_count,
            'percentage': {
                pos_category: {word: (count / total_count * 100) for word, count in counter.items()}
                for pos_category, counter in pos_counts.items()
            }
        }

    return pos_statistics


# 保存统计结果到 CSV 文件，每组每个类别生成一个文件
def save_to_csv_by_group_and_category(pos_statistics, output_dir="output"):
    import os

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for group, stats in pos_statistics.items():
        group_name = "Suicide_Risk" if group == 0 else "No_Suicide_Risk"

        for pos_category, counter in stats['pos_counts'].items():
            all_data = []

            for word, freq in counter.items():
                percentage = stats['percentage'][pos_category][word]
                all_data.append({
                    "Word": word,
                    "Frequency": freq,
                    "Percentage (%)": f"{percentage:.4f}"
                })

            df_output = pd.DataFrame(all_data)

            category_file_name = f"{group_name}_{pos_category}_Statistics.csv"
            output_file_path = os.path.join(output_dir, category_file_name)

            df_output.to_csv(output_file_path, index=False, encoding='utf-8-sig')

            print(f"Saved results to {output_file_path}")


# 主函数：加载数据并计算结果，并生成 CSV 输出
def main():
    # 文件路径
    file_paths = ['/kaggle/input/cipintongji/dev.txt', '/kaggle/input/cipintongji/train.txt']  # 替换为实际文件路径

    # 加载数据
    df = load_data(file_paths)

    print("\nLoaded Data:")
    print(df.head())

    # 计算词性统计信息
    pos_statistics = compute_pos_statistics(df, group_column='label', text_column='text')

    print("\nSaving POS Statistics to CSV files...")

    save_to_csv_by_group_and_category(pos_statistics)


if __name__ == '__main__':
    main()