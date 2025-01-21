# IG
import pandas as pd
import re
from math import log2


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

    text = re.sub(r"#\S+", "", text)  # 移除以 "#" 开头的单词（如 "#话题"）
    text = re.sub(r"@\S+", "", text)  # 移除以 "@" 开头的单词（如 "@用户"）
    return text



def load_words(word_file):
    words = []
    with open(word_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(r'Word:\s*(\S+),', line)  # 提取 "Word:" 后面的部分
            if match:
                words.append(match.group(1))
    return words



def entropy(probabilities):
    return -sum(p * log2(p) for p in probabilities if p > 0)



def compute_information_gain(word, group_0_texts, group_1_texts, total_texts):

    N11 = sum(1 for text in group_0_texts if word in text)
    N01 = sum(1 for text in group_1_texts if word in text)


    N10 = len(group_0_texts) - N11
    N00 = len(group_1_texts) - N01


    total_group_0 = len(group_0_texts)
    total_group_1 = len(group_1_texts)


    P_group_0 = total_group_0 / total_texts
    P_group_1 = total_group_1 / total_texts


    H_T = entropy([P_group_0, P_group_1])


    P_w = (N11 + N01) / total_texts
    P_not_w = (N10 + N00) / total_texts


    H_T_given_w = 0
    H_T_given_not_w = 0

    if P_w > 0:
        P_T_given_w = [N11 / (N11 + N01), N01 / (N11 + N01)]
        H_T_given_w += P_w * entropy(P_T_given_w)

    if P_not_w > 0:
        P_T_given_not_w = [N10 / (N10 + N00), N00 / (N10 + N00)]
        H_T_given_not_w += P_not_w * entropy(P_T_given_not_w)

    IG_value = H_T - (H_T_given_w + H_T_given_not_w)

    return IG_value, N11, N01



def main():

    train_file_path = '/kaggle/input/shyan3-4/train.txt'
    dev_file_path = '/kaggle/input/shyan3-4/dev.txt'
    word_file_path = '/kaggle/input/high-freq-word/1-LLR-V.txt'


    df = load_data([train_file_path, dev_file_path])


    df['text'] = df['text'].apply(clean_text)


    group_0_texts = df[df['label'] == 0]['text'].tolist()
    group_1_texts = df[df['label'] == 1]['text'].tolist()

    total_texts = len(df)  #


    words = load_words(word_file_path)


    results = []

    print("\nCalculating Information Gain for each word...\n")

    for word in words:
        ig_value, count_group_0, count_group_1 = compute_information_gain(
            word, group_0_texts, group_1_texts, total_texts)


        total_count = count_group_0 + count_group_1
        if total_count > 0:
            percentage_group_0 = (count_group_0 / total_count) * 100
            percentage_group_1 = (count_group_1 / total_count) * 100
        else:
            percentage_group_0 = percentage_group_1 = 0

        results.append({
            'word': word,
            'information_gain': ig_value,
            'percentage_suicide': f"{percentage_group_0:.2f}%",
            'percentage_non_suicide': f"{percentage_group_1:.2f}%"
        })

        print(
            f"Word: {word}, IG: {ig_value:.4f}, Suicide: {percentage_group_0:.2f}%, Non-Suicide: {percentage_group_1:.2f}%")


    results_df = pd.DataFrame(results)
    results_df.to_csv('word_information_gain4.csv', index=False, encoding='utf-8')

    print("\nResults saved to 'word_information_gain3.csv'.")


if __name__ == '__main__':
    main()
