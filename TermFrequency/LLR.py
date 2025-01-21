# LLR分数
import pandas as pd
from collections import Counter
import jieba.posseg as pseg
import re
from math import log


CUSTOM_STOPWORDS = {"图片", "视频", "链接", "原图", "全文", "网页链接"}


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



def compute_pos_statistics(df, group_column, text_column):
    pos_statistics = {}

    target_pos_tags = {
        'Noun': ['n', 'nr', 'ns', 'nt', 'nz'],
        'Verb': ['v', 'vd', 'vn'],  #
        'Pronouns': ['r'],
        'Adjectives': ['a', 'ad', 'an'],  #
        'Adverbs': ['d'],
        'Prepositions': ['p'],
        'Conjunctions': ['c']
    }

    total_texts_per_group = df[group_column].value_counts().to_dict()

    for group in df[group_column].unique():
        group_data = df[df[group_column] == group][text_column]

        pos_counts = {pos: Counter() for pos in target_pos_tags}

        for text in group_data:
            if isinstance(text, str):  #
                cleaned_text = clean_text(text)

                words_with_pos = pseg.cut(cleaned_text)

                for word, tag in words_with_pos:
                    if word not in CUSTOM_STOPWORDS and word.strip():
                        for pos_category, pos_tags in target_pos_tags.items():
                            if tag in pos_tags:
                                pos_counts[pos_category][word] += 1

        pos_statistics[group] = {
            'pos_counts': pos_counts,
            'total_texts': total_texts_per_group[group],
        }

    return pos_statistics



def compute_llr(pos_statistics):
    llr_results_group_0 = {}
    llr_results_group_1 = {}

    groups = list(pos_statistics.keys())

    if len(groups) != 2:
        raise ValueError("This function expects exactly two groups to compute LLR.")

    group_0_stats, group_1_stats = pos_statistics[groups[0]], pos_statistics[groups[1]]

    for pos_category in ['Noun', 'Verb']:
        counter_0 = group_0_stats['pos_counts'][pos_category]
        counter_1 = group_1_stats['pos_counts'][pos_category]

        total_texts_0 = group_0_stats['total_texts']
        total_texts_1 = group_1_stats['total_texts']

        all_words_group_0 = set(counter_0.keys())
        all_words_group_1 = set(counter_1.keys())

        llr_scores_group_0 = []
        llr_scores_group_1 = []

        for word in all_words_group_0:
            N11 = counter_0[word]
            N01 = counter_1[word]
            N10 = total_texts_0 - N11
            N00 = total_texts_1 - N01

            if N11 > 0 and N01 > 0 and N10 > 0 and N00 > 0:
                llr_score_group_0 = log((N11 * N00) / (N10 * N01))
                llr_scores_group_0.append((word, llr_score_group_0, N11))

        for word in all_words_group_1:
            N11 = counter_1[word]
            N01 = counter_0[word]
            N10 = total_texts_1 - N11
            N00 = total_texts_0 - N01

            if N11 > 0 and N01 > 0 and N10 > 0 and N00 > 0:
                llr_score_group_1 = log((N11 * N00) / (N10 * N01))
                llr_scores_group_1.append((word, llr_score_group_1, N11))

        llr_results_group_0[pos_category] = sorted(llr_scores_group_0, key=lambda x: x[1], reverse=True)[:100]
        llr_results_group_1[pos_category] = sorted(llr_scores_group_1, key=lambda x: x[1], reverse=True)[:100]

    return llr_results_group_0, llr_results_group_1



def main():
    file_paths = ['/kaggle/input/cipintongji/dev.txt', '/kaggle/input/cipintongji/train.txt']  # 替换为实际文件路径

    df = load_data(file_paths)

    print("\nLoaded Data:")
    print(df.head())

    pos_statistics = compute_pos_statistics(df, group_column='label', text_column='text')

    print("\nComputing LLR Scores...")

    llr_results_group_0, llr_results_group_1 = compute_llr(pos_statistics)

    print("\nTop Words by LLR (Group: Suicide Risk):")

    for pos_category, words_info in llr_results_group_0.items():
        print(f"\nTop {pos_category}:")
        for word, score, freq in words_info:
            print(f"Word: {word}, LLR Score: {score:.4f}, Frequency: {freq}")

    print("\nTop Words by LLR (Group: No Suicide Risk):")

    for pos_category, words_info in llr_results_group_1.items():
        print(f"\nTop {pos_category}:")
        for word, score, freq in words_info:
            print(f"Word: {word}, LLR Score: {score:.4f}, Frequency: {freq}")


if __name__ == '__main__':
    main()