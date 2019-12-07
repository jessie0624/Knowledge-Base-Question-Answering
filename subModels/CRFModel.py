from typing import List, Optional
import torch
import torch.nn as nn

'''
CRF
从例子说起——词性标注问题
啥是词性标注问题？

非常简单的，就是给一个句子中的每个单词注明词性。比如这句话：“Bob drank coffee at Starbucks”，注明每个单词的词性后是这样的：“Bob (名词) drank(动词) coffee(名词) at(介词) Starbucks(名词)”。

下面，就用条件随机场来解决这个问题。

以上面的话为例，有5个单词，我们将：(名词，动词，名词，介词，名词)作为一个标注序列，称为l，可选的标注序列有很多种，比如l还可以是这样：（名词，动词，动词，介词，名词），我们要在这么多的可选标注序列中，挑选出一个最靠谱的作为我们对这句话的标注。

怎么判断一个标注序列靠谱不靠谱呢？

就我们上面展示的两个标注序列来说，第二个显然不如第一个靠谱，因为它把第二、第三个单词都标注成了动词，动词后面接动词，这在一个句子中通常是说不通的。
假如我们给每一个标注序列打分，打分越高代表这个标注序列越靠谱，我们至少可以说，凡是标注中出现了动词后面还是动词的标注序列，要给它负分！！

上面所说的动词后面还是动词就是一个特征函数，我们可以定义一个特征函数集合，用这个特征函数集合来为一个标注序列打分，并据此选出最靠谱的标注序列。也就是说，每一个特征函数都可以用来为一个标注序列评分，把集合中所有特征函数对同一个标注序列的评分综合起来，就是这个标注序列最终的评分值。

定义CRF中的特征函数 ()
现在，我们正式地定义一下什么是CRF中的特征函数，所谓特征函数，就是这样的函数，它接受四个参数：
1. 句子s（就是我们要标注词性的句子）
2. i，用来表示句子s中第i个单词
3. l_i，表示要评分的标注序列给第i个单词标注的词性
4. l_i-1，表示要评分的标注序列给第i-1个单词标注的词性

它的输出值是0或者1,0表示要评分的标注序列不符合这个特征，1表示要评分的标注序列符合这个特征。
Note:这里，我们的特征函数仅仅依靠当前单词的标签和它前面的单词的标签对标注序列进行评判，这样建立的CRF也叫作线性链CRF，这是CRF中的一种简单情况。为简单起见，本文中我们仅考虑线性链CRF。

从特征函数到概率
定义好一组特征函数后，我们要给每个特征函数f_j赋予一个权重λ_j。
现在，只要有一个句子s，有一个标注序列l，我们就可以利用前面定义的特征函数集来对l评分。 
其中i 是句子s中第i个单词. j是第j个特征函数.
score(l|s) = \sum_{j=1}^m \sum_{i=1}^n λ_j f_j(s, i, l_i, l_{i-1})

上式中有两个求和，外面的求和用来求每一个特征函数f_j评分值的和，里面的求和用来求句子中每个位置的单词的的特征值的和.

对这个分数进行指数化和标准化，我们就可以得到标注序列l的概率值p(l|s)，如下所示：
句子s 的标注序列为l的概率:
p(l|s) = exp[score(l|s)] / \sum exp[score(l|s)]

几个特征函数的例子
前面我们已经举过特征函数的例子，下面我们再看几个具体的例子，帮助增强大家的感性认识。
f1(s,i,li,l{i-1}) = 1  
当l_i是“副词”并且第i个单词以“ly”结尾时，我们就让f1 = 1，其他情况f1为0。
不难想到，f1特征函数的权重λ1应当是正的。而且λ1越大，表示我们越倾向于采用那些把以“ly”结尾的单词标注为“副词”的标注序列.

f2(s,i,li,l{i-1}) = 1 
如果i=1，l_i=动词，并且句子s是以“？”结尾时，f2=1，其他情况f2=0。
同样，λ2应当是正的，并且λ2越大，表示我们越倾向于采用那些把问句的第一个单词标注为“动词”的标注序列。

f3(s,i,li,l{i-1}) = 1 
当l_i-1是介词，l_i是名词时，f3 = 1，其他情况f3=0。λ3也应当是正的，并且λ3越大，说明我们越认为介词后面应当跟一个名词。

f4(s,i,li,l{i-1}) = 1 
如果l_i和l_i-1都是介词，那么f4等于1，其他情况f4=0。
这里，我们应当可以想到λ4是负的，并且λ4的绝对值越大，表示我们越不认可介词后面还是介词的标注序列。

好了，一个条件随机场就这样建立起来了，让我们总结一下：
为了建一个条件随机场，我们首先要定义一个特征函数集，每个特征函数都以整个句子s，当前位置i，位置i和i-1的标签为输入。
然后为每一个特征函数赋予一个权重，然后针对每一个标注序列l，对所有的特征函数加权求和，必要的话，可以把求和的值转化为一个概率值。

CRF与逻辑回归的比较
事实上，条件随机场是逻辑回归的序列化版本。逻辑回归是用于分类的对数线性模型，条件随机场是用于序列化标注的对数线性模型。

CRF与HMM的比较
对于词性标注问题，HMM模型也可以解决。HMM的思路是用生成办法，就是说，在已知要标注的句子s的情况下，去判断生成标注序列l的概率，如下所示：
p(l,s) = p(l1)\prod_i p(li|l{i-1})p(wi|li)
这里：
p(l_i|l_i-1)是转移概率，比如，l_i-1是介词，l_i是名词，此时的p表示介词后面的词是名词的概率。
p(w_i|l_i)表示发射概率（emission probability），比如l_i是名词，w_i是单词“ball”，此时的p表示在是名词的状态下，是单词“ball”的概率。


那么，HMM和CRF怎么比较呢？
答案是：CRF比HMM要强大的多，它可以解决所有HMM能够解决的问题，并且还可以解决许多HMM解决不了的问题。
事实上，我们可以对上面的HMM模型取对数，就变成下面这样：
log p(l,s) = logp(l0) + \sum_i log p(li|li-1) + \sum_i log p(wi|li)



CRF可以定义数量更多，种类更丰富的特征函数。HMM模型具有天然具有局部性，
就是说，在HMM模型中，当前的单词只依赖于当前的标签，当前的标签只依赖于前一个标签。
这样的局部性限制了HMM只能定义相应类型的特征函数，我们在上面也看到了。
但是CRF却可以着眼于整个句子s定义更具有全局性的特征函数.


'''

class CRF(nn.Module):
    def __init__(self, num_tags: int=2, batch_first:bool=True) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags:{num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        ## start 到其他tag(不包含end)的得分
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        ## 其他tag (不含start) 到end的得分
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        '''
        从_compute_normalizer中 next_score= broadcast_score + self.transitions + broadcast_mession
        可以看出 transitons[i][j]表示从第j个tag 到第i个tag的分数
        更正:transitions[i][j] 表示第i个tag 到第j个tag的分数
        '''
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self):
        init_range = 0.1
        nn.init.uniform_(self.start_transitions, -init_range, init_range)
        nn.init.uniform_(self.end_transitions, -init_range, init_range)
        nn.init.uniform_(self.transitions, -init_range, init_range)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'
    
    def forward(self, emissions: torch.Tensor,
                tags: torch.Tensor = None,
                mask: Optional[torch.ByteTensor] = None,
                reduction: str='mean') -> torch.Tensor:
        self.__validate(emissions, tags=tags, mask=mask)
        reduction = reduction.lower()
        if reduction not in ('none', 'sum','mean', 'token_mean'):
            raise ValueError(f'invalid reduction {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, type=torch.unit8)
        if self.batch_first:
            # emissions: seq_len, batch_size, tag_num
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)
        
        #shape: (batch_size, )
        # numerator 分子
        numerator = self.__compute_score(emissions=emissions,tags=tags, mask=mask)
        # denominator 分母
        denominator = self.__compute_normalizer(emissions=emissions, mask=mask)
        llh = denominator - numerator

        if reduction == 'none':
            return llh
        elif reduction == 'sum':
            return llh.sum()
        elif reduction == 'mean':
            return llh.mean()
        assert reduction == 'token_mean'
        return llh.sum() / mask.float().sum()

    def decode(self, emissions:torch.Tensor,
               mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:
        self.__validate(emissions=emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.unit8)
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)
        return self.__validate_decode(emissions, mask)
    
    def __validate(self, emissions:torch.Tensor,
                   tags:Optional[torch.LongTensor] = None,
                   mask:Optional[torch.ByteTensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError(f'emssions must have dimesion of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimesion of emission is {self.num_tags}, got {emssions.size(2)}')
        if tags is not None:
            if emissions.shape[:2] != mask.shape:
                    raise ValueError(
                    'the first two dimensions of emissions and mask must match,'
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}'
                )
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:,0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def __compute_score(self, emissions: torch.Tensor,
                        tags: torch.LongTensor,
                        mask: torch.ByteTensor) -> torch.Tensor:
        # batch second
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()

        seq_length, batch_size = tags.shape
        mask = mask.float()

        ## self.start_transitions  start 到其他tag(不含end)的部分
        score = self.start_transitions[tags[0]]

        #emssions.shape (seq_len, batch_size, tag_nums)
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            score += self.transitions[tags[i-1], tags[i]] * mask[i]
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]
        ##这里是为了获取每个样本最后一个词的tag.
        # shape: batch_size,
        seq_ends = mask.long().sum(dim=0) - 1
        ##每个样本最后一个词的tag
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape:(batch_size) 每个样本到最后一个词的得分加上之前的score
        score += self.end_transitions[last_tags]
        return score
    
    def __compute_normalizer(self, emissions:torch.Tensor,
                            mask: torch.ByteTensor) -> torch.Tensor:
        # emission(seq_len, batch_size, num_tags)
        # mask: (seq_len, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length = emissions.size(0)
        # shape: batch_size, num_tag
        # self.start_transitions, start 到其他tagd的得分,不含end
        # start_transitions.shape tag_nums, emission[0].shape (batch_size, tag_size)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            #shape: batch_size, num_tag, 1
            broadcast_score = score.unsqueeze(dim=2)

            #shape: (batch_size,1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
        # shape (batch_size, num_tags)
        score += self.end_transitions
        return torch.logsumexp(score, dim=1)
    
    def __viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor) -> List[List[int]]:
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length , batch_size = mask.shape
        # self.start_transitions  start 到其他tag(不包含end)的得分
        score = self.start_transitions + emissions[0]
        history = []
        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)
        score += self.end_transitions
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []
        
        for idx in range(batch_size):
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())
            best_tags.reverse()
            best_tags_list.append(best_tags)
        return best_tags_list