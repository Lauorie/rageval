import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Plots:
    def __init__(self, file_path):
        self.df = pd.read_excel(file_path)
        self.score_cloumns = self.score_cloumns()

    def score_cloumns(self):
        """
        获取评估分数列名。
        """
        return [column for column in self.df.columns if 'score' in column]
    
    def analyze_scores(self):
        """
        对评估分数进行描述性统计分析。
        """
        print(self.df.describe())

    def visualize_distributions(self):
        """
        可视化评估分数的分布。
        """
        plt.figure(figsize=(10, 6))
        for column in self.score_cloumns:
            sns.histplot(self.df[column], kde=True, label=column)
        plt.legend()
        plt.show()

    def visualize_boxplots(self):
        """
        使用箱形图比较不同评估类型的分数分布。
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df[self.score_cloumns])
        plt.show()

    def visualize_scatterplot(self, x, y):
        """
        可视化两种评估类型分数之间的关系。
        """
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=self.df, x=x, y=y)
        plt.show()

    def calculate_correlation(self):
        """
        计算指定评估类型分数之间的相关系数。
        """
        print(self.df[self.score_cloumns].corr())

    def run(self):
        """
        运行分析流程。
        """
        self.analyze_scores()
        self.visualize_distributions()
        self.visualize_boxplots()
        self.visualize_scatterplot(self.score_cloumns[-2], self.score_cloumns[-1])
        self.calculate_correlation()
    
if __name__ == '__main__':
    file_path = '/root/app/PLC问题测试_processed.xlsx'
    plots = Plots(file_path)
    plots.run()
    


