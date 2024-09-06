
## token序列常规编码方式
RNN (如LSTM)
$$  
\boldsymbol{y}_t =f(\boldsymbol{y}_{t-1},\boldsymbol{x}_t)  
$$

CNN
$$
\boldsymbol{y}_t = f(\boldsymbol{x}_{t-1},\boldsymbol{x}_t,\boldsymbol{x}_{t+1})
$$

Attention:
$$
\boldsymbol{y}_t = f(\boldsymbol{x}_t,\boldsymbol{A},\boldsymbol{B})
$$
其中, A和B是另外的词语序列 (矩阵)