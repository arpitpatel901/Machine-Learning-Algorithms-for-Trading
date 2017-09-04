
import numpy as np
import BagLearner as bl
import LinRegLearner as lr

class InsaneLearner(object):
    def __init__(self,verbose):
        self.verbose=verbose
        self.learner = bl.BagLearner(learner=bl.BagLearner, kwargs={"learner": lr.LinRegLearner, "bags": 20, "kwargs": {"verbose": False}, "boost": False, "verbose": False}, bags=20,
                               boost=False, verbose=False)
    def author(self):
        return 'apatel380'
    def addEvidence(self,X,Y):
        self.learner.addEvidence(X,Y)

    def query(self, points): #query the Y values by scnanning the TREE and then average the Y's
        return self.learner.query(points)


#Hi Arpit, sorry for the delay.
#Basically you call add evidence and query only once and without a loop.
# Make sure your kwargs are nested(it's not). So you ll have a BL , inside that.. Your learner is of
#  type BL and  20 in number specified by bags. In the first kwargs, set the parameters for the second
#  level bag learners, i.e they must be have 20 bags of linreg learners each. In the inner kwargs which is
#   the second kwargs .. They will have parameters for linreg, i.e just verbose parameter.
