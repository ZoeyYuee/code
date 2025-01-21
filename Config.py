# 文件路径
TRAIN_PATH = 'data/train.txt'
TEST_PATH = 'data/dev.txt'
BERT_MODEL = 'bert-base-chinese/'
MODEL_DIR = 'models/'


BERT_PAD_ID = 0
MAX_LEN = 200       # 最大长度，超过截取，缺少填充
BATH_SIZE = 128
EPOCH = 5
LR = 0.0001         # 学习率
CLASS_NUM = 2      # 分类数，二分类
EMBEDDING = 768    # 子向量维度，BERT 中默认


OUTPUT_SIZE = 2    # BiLSTM 参数
N_LAYERS = 2       # BiLSTM 的层数
HIDDEN_DIM = 128   # LSTM 中隐层的维度，768/2
DROP_PROB = 0.2    # dropout 参数

