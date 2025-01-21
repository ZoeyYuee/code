
import pandas as pd
from collections import Counter
import jieba.posseg as pseg
import re
from prettytable import PrettyTable


CUSTOM_STOPWORDS = {"图片", "视频", "链接", "原图", "全文", "网页链接"}


PRONOUN_CATEGORIES = {
    'First Person Singular': {"我", "自己", "俺", "咱"},
    'First Person Plural': {"我们", "咱们"},
    'Second Person': {"你", "你们", "您", "您们"},
    'Third Person': {"他", "她", "他们", "她们", "爸爸", "妈妈", "哥哥", "姐姐", "叔叔", "大伯"}
}



def load_data(file_paths):
    data = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:

                        if '\t' in line:
                            text, label = line.rsplit('\t', 1)
                        else:
                            text, label = line.rsplit(' ', 1)
                        data.append({'text': text.strip(), 'label': int(label)})
                    except ValueError:
                        print(f"Skipping invalid line: {line}")
    return pd.DataFrame(data)



def clean_text(text):
    if not isinstance(text, str):
        return ""


    text = re.sub(r"#\S+", "", text)
    text = re.sub(r"@\S+", "", text)


    text = re.sub(r"\S+的微博视频", "", text)

    return text



def compute_pronoun_statistics(df, group_column, text_column):
    pronoun_statistics = {}

    for group in df[group_column].unique():
        group_data = df[df[group_column] == group][text_column]

        pronoun_counts = {category: Counter() for category in PRONOUN_CATEGORIES}

        for text in group_data:
            if isinstance(text, str):
                cleaned_text = clean_text(text)

                words_with_pos = pseg.cut(cleaned_text)

                for word, tag in words_with_pos:
                    if word not in CUSTOM_STOPWORDS and word.strip() and tag == 'r':
                        for category, pronouns in PRONOUN_CATEGORIES.items():
                            if word in pronouns:
                                pronoun_counts[category][word] += 1

        total_count = sum(sum(counter.values()) for counter in pronoun_counts.values())

        pronoun_statistics[group] = {
            'pronoun_counts': pronoun_counts,
            'total_count': total_count,
            'percentage': {
                category: {word: (count / total_count * 100) for word, count in counter.items()}
                for category, counter in pronoun_counts.items()
            }
        }

    return pronoun_statistics



def generate_table(pronoun_statistics):
    table = PrettyTable()

    table.field_names = ["Pronoun Category", "Group", "Frequency", "Percentage (%)"]

    for group, stats in pronoun_statistics.items():
        total_count = stats['total_count']

        for category, counter in stats['pronoun_counts'].items():
            category_total = sum(counter.values())
            table.add_row([category,
                           "Suicide Risk" if group == 0 else "No Suicide Risk",
                           category_total,
                           f"{category_total / total_count * 100:.2f}"])

    return table



def get_top_n_pronouns(pronoun_statistics, top_n=20):
    top_words_by_group = {}

    for group, stats in pronoun_statistics.items():
        top_words_by_group[group] = {}

        for category, counter in stats['pronoun_counts'].items():
            top_words_by_group[group][category] = [
                (word, count, stats['percentage'][category][word])
                for word, count in counter.most_common(top_n)
            ]

    return top_words_by_group



def main():
    file_paths = ['/kaggle/input/cipintongji/dev.txt', '/kaggle/input/cipintongji/train.txt']  # 替换为实际文件路径


    df = load_data(file_paths)

    print("\nLoaded Data:")
    print(df.head())


    pronoun_statistics = compute_pronoun_statistics(df, group_column='label', text_column='text')

    print("\nPronoun Statistics Table:")

    result_table = generate_table(pronoun_statistics)

    print(result_table)

    print("\nTop Pronouns by Group:")

    top_pronouns_by_group = get_top_n_pronouns(pronoun_statistics)

    for group, words_info in top_pronouns_by_group.items():
        print(f"\nGroup {'Suicide Risk' if group == 0 else 'No Suicide Risk'}:")

        for category, words_list in words_info.items():
            print(f"\nTop {category}:")
            for word_info in words_list:
                print(f"Word: {word_info[0]}, Count: {word_info[1]}, Percentage: {word_info[2]:.2f}%")


if __name__ == '__main__':
    main()