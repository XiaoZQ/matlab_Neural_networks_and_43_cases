import numpy as np


class BPNuNeuralNetwork:
    def __init__(self, indata, outdata, hide_layer, activation_fun,
                 training_fun=None, learn_fun=None, performance_fun=lambda x,y:x-y,
                 input_processing_fun=None, output_processing_fun=None):
        self.indata = indata
        self.outdata = outdata
        self.hide_layer = hide_layer
        self.AF = activation_fun  # 传递函数
        self.TF = training_fun    # 训练函数
        self.LF = learn_fun   # 学习函数
        self.PF = performance_fun  # 性能函数，即误差
        self.IPF = input_processing_fun   # 输入处理函数
        self.OPT = output_processing_fun  # 输出处理函数
        self.DDF = None   # 验证数据划分函数
        self.w = None
        self.b = None

    def train(self, epochs=100, learn_ratio=0.1, goal=0.00004):
        # np.random.seed(10)
        innum = self.indata.shape[1]
        midnum = self.hide_layer
        outnum = self.outdata.shape[1]
        w = {}
        b = {}
        # w[0] = np.random.rand(innum, midnum[0])
        if isinstance(midnum, int):
            w[0] = (np.random.random((innum, midnum))*2)-1
            b[0] = (np.random.random((1, midnum))*2)-1
            w[1] = (np.random.random((midnum, outnum))*2)-1
            b[1] = (np.random.random((1, outnum))*2)-1
            layernum = 1
        else:
            w[0] = (np.random.random((innum, midnum[0]))*2)-1
            b[0] = (np.random.random((1, midnum[0]))*2)-1
            layernum = len(midnum)
            for i in range(1,len(midnum)):
                w[i] = (np.random.random((midnum[i-1], midnum[i]))*2)-1
                b[i] = (np.random.random((1, midnum[i]))*2)-1
            w[layernum] = (np.random.random((midnum[-1], outnum))*2)-1
            b[layernum] = (np.random.random((1, outnum))*2)-1
        Err = np.zeros(epochs)

        for ii in range(epochs):

            for i in range(len(self.indata)):
                dw = {}
                db = {}
                Hin = {}

                Hout = {}

                Hin[0] = self.indata[i:i+1, :]
                Hout[0] = self.AF(np.dot(Hin[0], w[0])+b[0]).function()
                for L in range(1, layernum):
                    Hin[L] = Hout[L-1]
                    Hout[L] = self.AF(np.dot(Hout[L-1], w[L])+b[L]).function()

                yn = np.dot(Hout[layernum-1], w[layernum])+b[layernum]
                e = self.PF(self.outdata[i,:], yn)
                Err[ii] += np.abs(e).sum()
                dw[layernum] = np.dot(np.transpose(Hout[layernum-1]), e)
                db[layernum] = e
                # print(yn)
                xita = {layernum:e}
                for L in range(layernum):
                    Lt = layernum-L-1
                    FI = self.AF(np.dot(Hin[Lt], w[Lt])+b[Lt]).derivative()
                    # FI = Hout[Lt]*(1-Hout[Lt])
                    # print(FI)
                    xita[Lt] = np.dot(xita[Lt+1], np.transpose(w[Lt+1]))*FI
                    dw[Lt] = np.dot(np.transpose(Hin[Lt]), xita[Lt])
                    db[Lt] = xita[Lt]

                for L in range(layernum+1):
                    w[L] += learn_ratio*dw[L]
                    b[L] += learn_ratio*db[L]

        self.w = w
        self.b = b

    def predict(self, data):
        temp = data
        if isinstance(self.hide_layer,int):
            layernum=1
        else:
            layernum=len(self.hide_layer)

        for L in range(layernum+1):
            temp = self.AF(np.dot(temp,self.w[L])+self.b[L]).function()
        return temp

