from learning2rank.learning2rank.rank import RankNet
from learning2rank.learning2rank.utils import NNfuncs
import numpy as np


X = np.random.rand(200, 152)
y = np.random.randint(0, 10, size=(200,))

RankNetModel = RankNet.RankNet()
RankNetModel.fit(X, y)
RankNetModel.predict(X)

# RankNetModel.loading("RankNet.model", X)
# #print RankNetModel.predictTargets(X, 100)
# scores = np.squeeze(RankNetModel.predict(X), axis=1)
# labels = []
# for i, s in enumerate(scores):
#     labels.append({
#         "index": i,
#         "score": s
#     })
# sortedLabels = sorted(labels, key=lambda k: k['score'])
# print [label['index'] for label in sortedLabels]
# print [label['score'] for label in sortedLabels]
