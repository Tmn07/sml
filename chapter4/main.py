# coding=utf-8

import re


# rule=re.compile(r'[^a-z]')
# result = rule.sub(' ',string)

class preprocess():
    stop_word_list = []
    word_list = dict()
    rule = re.compile(r'[^a-zA-Z]')

    def __init__(self):
        self.read_stop_word()

    def read_stop_word(self):
        """
        读取停用词
        """
        with open('stop_word.txt') as f:
            while 1:
                line = f.readline().strip()
                if line:
                    self.stop_word_list.append(line)
                else:
                    break

    def yy(self, etype):
        """
        一个类型的转化，传入ham返回0，传入spam返回1
        :param etype: 类型
        :return: 类型id
        """
        if etype == 'ham':
            return 0
        else:
            return 1

    def textprocess(self, line):
        """
        文本预处理
        :param line:一行邮件文本
        :return:处理后的单词列表
        """
        data = self.rule.sub(' ', line).strip().lower().split()
        # print (data)
        # data = line.split()
        result = []
        for word in data:
            if word in self.stop_word_list:
                continue
            else:
                result.append(word)
        return result

    def add_word(self, word, etype):
        if word not in self.word_list:
            # 贝叶斯估计 +r
            self.word_list[word] = [1, 1]
        self.word_list[word][etype] += 1

    def add_word_list(self, wl, etype):
        for i in xrange(len(wl)):
            self.add_word(wl[i], etype)


class bys(preprocess):

    def __init__(self):
        preprocess.__init__(self)

    def main(self):
        self.read_train_data()
        self.read_test_data()

    def read_data(self,filename):
        """
        读取数据文件
        :param filename: 文件名
        :return: (x,y)数据和分类
        """
        x = []
        y = []
        with open(filename) as f:
            while 1:
                line = f.readline()
                if line:
                    tmp = line.split(',')
                    etype = self.yy(tmp[0])
                    wordlist = self.textprocess(tmp[1])
                    y.append(etype)
                    x.append(wordlist)
                else:
                    break
        return (x,y)

    def read_train_data(self):
        train_data = self.read_data('data1.txt')
        x = train_data[0]
        y = train_data[1]
        # 贝叶斯估计 +r
        self.N = len(y)
        self.ham_num = y.count(0)
        self.spam_num = y.count(1)
        for i in xrange(self.N):
            self.add_word_list(x[i],y[i])


    def read_test_data(self):
        """
        读取测试数据集
        """
        test_data = self.read_data('data2.txt')
        x = test_data[0]
        real_y = test_data[1]
        y = []
        for i in xrange(len(x)):
            # 贝叶斯估计
            p0 = (1.0*self.ham_num+1)/(self.N+2)
            p1 = (1.0*self.spam_num+1) / (self.N+2)

            for word in x[i]:
                p0 *= self.xianyan(word,0)
                p1 *= self.xianyan(word,1)

            y.append(0 if p0>p1 else 1)

        print (self.compare(y,real_y))

    def xianyan(self,word,etype):
        """
        用贝叶斯公式计算后验概率，需要获取先验的条件概率
        :param word: 句子中的某个词
        :param etype: 这个句子的类型
        :return: 先验
        """
        if word in self.word_list:
            return self.word_list[word][etype]
            # if self.word_list[word][etype] == 0:
            #     return 1
            # else:
            #     return self.word_list[word][etype]
        else:
            # 若存在在训练集里没有出现过的单词，返回1（相当于对各分类判断无影响
            print ('111')
            return 1

    def compare(self,y,real_y):
        """
        对比预测和真实结果
        :param y: 预测的分类结果
        :param real_y: 真实的分类结果
        :return: 准确率
        """
        n = len(y)
        s = 0
        for i in xrange(n-1):
            if y[i] == real_y[i]:
                s += 1
        return  1.0*s/n


if __name__ == '__main__':
    c1 = bys()
    c1.main()
