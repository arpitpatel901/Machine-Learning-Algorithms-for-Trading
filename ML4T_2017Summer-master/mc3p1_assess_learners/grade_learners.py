"""MC3-P1: Assess learners - grading script.

Usage:
- Switch to a student feedback directory first (will write "points.txt" and "comments.txt" in pwd).
- Run this script with both ml4t/ and student solution in PYTHONPATH, e.g.:
    PYTHONPATH=ml4t:MC3-P1/jdoe7 python ml4t/mc3_p1_grading/grade_learners.py
"""

import pytest
from grading.grading import grader, GradeResult, time_limit, run_with_timeout, IncorrectOutput
import util

import os
import sys
import traceback as tb

import numpy as np
import pandas as pd
from collections import namedtuple

import math

import time

import random

# Grading parameters
# rmse_margins = dict(KNNLearner=1.10, BagLearner=1.10)  # 1.XX = +XX% margin of RMS error
# points_per_test_case = dict(KNNLearner=3.0, BagLearner=2.0)  # points per test case for each group
# seconds_per_test_case = 10  # execution time limit
seconds_per_test_case = 6

# More grading parameters (picked up by module-level grading fixtures)
max_points = 60.0  # 4.0 * 10 + 2.0 * 10
html_pre_block = True  # surround comments with HTML <pre> tag (for T-Square comments field)

# Test cases
LearningTestCase = namedtuple('LearningTestCase', ['description', 'group', 'datafile', 'seed', 'outputs'])
learning_test_cases = [
    LearningTestCase(
        description="Test Case 01: Single Tree",
        group='RTLearner',
        datafile='Istanbul.csv',
        seed=1481090001,
        outputs=dict(
            insample_corr_min=0.95,
            outsample_corr_min=0.15,
            insample_corr_max=0.95
            )
        ),
    LearningTestCase(
        description="Test Case 01: Bagging",
        group='BagLearner',
        datafile='Istanbul.csv',
        seed=1481090001,
        outputs=None
        ),
    LearningTestCase(
        description="Test Case 02: Single Tree",
        group='RTLearner',
        datafile='Istanbul.csv',
        seed=1481090002,
        outputs=dict(
            insample_corr_min=0.95,
            outsample_corr_min=0.15,
            insample_corr_max=0.95
            )
        ),
    LearningTestCase(
        description="Test Case 02: Bagging",
        group='BagLearner',
        datafile='Istanbul.csv',
        seed=1481090002,
        outputs=None
        ),
    LearningTestCase(
        description="Test Case 03: Single Tree",
        group='RTLearner',
        datafile='Istanbul.csv',
        seed=1481090003,
        outputs=dict(
            insample_corr_min=0.95,
            outsample_corr_min=0.15,
            insample_corr_max=0.95
            )
        ),
    LearningTestCase(
        description="Test Case 03: Bagging",
        group='BagLearner',
        datafile='Istanbul.csv',
        seed=1481090003,
        outputs=None
        ),
    LearningTestCase(
        description="Test Case 04: Single Tree",
        group='RTLearner',
        datafile='Istanbul.csv',
        seed=1481090004,
        outputs=dict(
            insample_corr_min=0.95,
            outsample_corr_min=0.15,
            insample_corr_max=0.95
            )
        ),
    LearningTestCase(
        description="Test Case 04: Bagging",
        group='BagLearner',
        datafile='Istanbul.csv',
        seed=1481090004,
        outputs=None
        ),
    LearningTestCase(
        description="Test Case 05: Single Tree",
        group='RTLearner',
        datafile='Istanbul.csv',
        seed=1481090005,
        outputs=dict(
            insample_corr_min=0.95,
            outsample_corr_min=0.15,
            insample_corr_max=0.95
            )
        ),
    LearningTestCase(
        description="Test Case 05: Bagging",
        group='BagLearner',
        datafile='Istanbul.csv',
        seed=1481090005,
        outputs=None
        ),
    LearningTestCase(
        description="Test Case 06: Single Tree",
        group='RTLearner',
        datafile='Istanbul.csv',
        seed=1481090006,
        outputs=dict(
            insample_corr_min=0.95,
            outsample_corr_min=0.15,
            insample_corr_max=0.95
            )
        ),
    LearningTestCase(
        description="Test Case 06: Bagging",
        group='BagLearner',
        datafile='Istanbul.csv',
        seed=1481090006,
        outputs=None
        ),
     LearningTestCase(
         description="Test Case 07: Single Tree",
         group='RTLearner',
         datafile='Istanbul.csv',
         seed=1481090007,
         outputs=dict(
             insample_corr_min=0.95,
             outsample_corr_min=0.15,
             insample_corr_max=0.95
             )
         ),
     LearningTestCase(
         description="Test Case 07: Bagging",
         group='BagLearner',
         datafile='Istanbul.csv',
         seed=1481090007,
         outputs=None
         ),
     LearningTestCase(
         description="Test Case 08: Single Tree",
         group='RTLearner',
         datafile='Istanbul.csv',
         seed=1481090008,
         outputs=dict(
             insample_corr_min=0.95,
             outsample_corr_min=0.15,
             insample_corr_max=0.95
             )
         ),
     LearningTestCase(
         description="Test Case 08: Bagging",
         group='BagLearner',
         datafile='Istanbul.csv',
         seed=1481090008,
         outputs=None
         ),
]


# Test functon(s)
@pytest.mark.parametrize("description,group,datafile,seed,outputs", learning_test_cases)
def test_learners(description, group, datafile, seed, outputs, grader):
    """Test ML models returns correct predictions.

    Requires test description, test case group, inputs, expected outputs, and a grader fixture.
    """

    points_earned = 0.0  # initialize points for this test case
    try:
        learner_class = None
        kwargs = {'verbose':False}

        # (BPH) Copied from grade_strategy_qlearning.py
        #Set fixed seed for repetability
        np.random.seed(seed)
        random.seed(seed)
        # These lines will be uncommented in the batch grader to
        # prevent accidentally fixing the seed within student
        # code
        # tmp_numpy_seed = np.random.seed
        # tmp_random_seed = random.seed
        # np.random.seed = fake_seed
        # random.seed = fake_rseed

        # Try to import KNNLearner (only once)
        # if not 'KNNLearner' in globals():
        #     from KNNLearner import KNNLearner
        if not 'RTLearner' in globals():
            from RTLearner import RTLearner
        if group is 'BagLearner' and (not 'BagLearner' in globals()):
            from BagLearner import BagLearner

        # Tweak kwargs
        # kwargs.update(inputs.get('kwargs', {}))

        # Read separate training and testing data files
        # with open(inputs['train_file']) as f:
        # data_partitions=list()
        testX,testY,trainX,trainY = None,None, None,None
        permutation = None
        author = None
        with util.get_learner_data_file(datafile) as f:
            alldata = np.genfromtxt(f,delimiter=',')
            # Skip the date column and header row if we're working on Istanbul data
            if datafile == 'Istanbul.csv':
                alldata = alldata[1:,1:]
            datasize = alldata.shape[0]
            cutoff = int(datasize*0.6)
            permutation = np.random.permutation(alldata.shape[0])
            col_permutation = np.random.permutation(alldata.shape[1]-1)
            train_data = alldata[permutation[:cutoff],:]
            # trainX = train_data[:,:-1]
            trainX = train_data[:,col_permutation]
            trainY = train_data[:,-1]
            test_data = alldata[permutation[cutoff:],:]
            # testX = test_data[:,:-1]
            testX = test_data[:,col_permutation]
            testY = test_data[:,-1]

        if group is "RTLearner":
            corr_in, corr_out, corr_in_50 = None,None,None
            def oneleaf():
                learner = RTLearner(leaf_size=1,verbose=False)
                learner.addEvidence(trainX,trainY)
                insample = learner.query(trainX)
                outsample = learner.query(testX)
                return insample, outsample, learner.author()
            def fiftyleaves():
                learner = RTLearner(leaf_size=50,verbose=False)
                learner.addEvidence(trainX,trainY)
                return learner.query(trainX)

            predY_in, predY_out, author = run_with_timeout(oneleaf,seconds_per_test_case,(),{})
            predY_in_50 = run_with_timeout(fiftyleaves,seconds_per_test_case,(),{})
            corr_in = np.corrcoef(predY_in,y=trainY)[0,1]
            corr_out = np.corrcoef(predY_out,y=testY)[0,1]
            corr_in_50 = np.corrcoef(predY_in_50,y=trainY)[0,1]
            incorrect = False

            msgs = []
            if corr_in < outputs['insample_corr_min']:
                incorrect = True
                msgs.append("    In-sample with leaf_size=1 correlation less than allowed: got {} expected {}".format(corr_in,outputs['insample_corr_min']))
            else:
                points_earned += 1.5
            if corr_out < outputs['outsample_corr_min']:
                incorrect = True
                msgs.append("    Out-of-sample correlation less than allowed: got {} expected {}".format(corr_out,outputs['outsample_corr_min']))
            else:
                points_earned += 1.5
            if corr_in_50 > outputs['insample_corr_max']:
                incorrect = True
                msgs.append("    In-sample correlation with leaf_size=50 greater than allowed: got {} expected {}".format(corr_in_50,outputs['insample_corr_max']))
            else:
                points_earned += 1.0
            # Check author string
            if (author is None) or (author =='tb34'):
                incorrect = True
                msgs.append("    Invalid author: {}".format(author))
                points_earned += -1.0

        elif group is "BagLearner":
            corr1, corr20 = None,None
            def onebag():
                learner1 = BagLearner(learner=RTLearner,kwargs={"leaf_size":1},bags=1,boost=False,verbose=False)
                learner1.addEvidence(trainX,trainY)
                return learner1.query(testX),learner1.author()
            def twentybags():
                learner20 = BagLearner(learner=RTLearner,kwargs={"leaf_size":1},bags=20,boost=False,verbose=False)
                learner20.addEvidence(trainX,trainY)
                return learner20.query(testX)
            predY1,author = run_with_timeout(onebag,seconds_per_test_case,pos_args=(),keyword_args={})
            predY20 = run_with_timeout(twentybags,seconds_per_test_case,(),{})

            corr1 = np.corrcoef(predY1,testY)[0,1]
            corr20 = np.corrcoef(predY20,testY)[0,1]
            incorrect = False
            msgs = []
            if corr20 <= corr1:
                incorrect = True
                msgs.append("    Out-of-sample correlation for 20 bags is not greater than for 1 bag. 20 bags:{}, 1 bag:{}".format(corr20,corr1))
            else:
                points_earned += 2.0
            # Check author string
            if (author is None) or (author=='tb34'):
                incorrect = True
                msgs.append("    Invalid author: {}".format(author))
                points_earned += -1.0

        if incorrect:
            inputs_str = "    data file: {}\n" \
                         "    permutation: {}".format(datafile, permutation)
            raise IncorrectOutput, "Test failed on one or more output criteria.\n  Inputs:\n{}\n  Failures:\n{}".format(inputs_str, "\n".join(msgs))
    except Exception as e:
        # Test result: failed
        msg = "Description: {} (group: {})\n".format(description, group)
        
        # Generate a filtered stacktrace, only showing erroneous lines in student file(s)
        tb_list = tb.extract_tb(sys.exc_info()[2])
        for i in xrange(len(tb_list)):
            row = tb_list[i]
            tb_list[i] = (os.path.basename(row[0]), row[1], row[2], row[3])  # show only filename instead of long absolute path
        tb_list = [row for row in tb_list if (row[0] == 'RTLearner.py') or (row[0] == 'BagLearner.py')]
        if tb_list:
            msg += "Traceback:\n"
            msg += ''.join(tb.format_list(tb_list))  # contains newlines
        msg += "{}: {}".format(e.__class__.__name__, e.message)

        # Report failure result to grader, with stacktrace
        grader.add_result(GradeResult(outcome='failed', points=points_earned, msg=msg))
        raise
    else:
        # Test result: passed (no exceptions)
        grader.add_result(GradeResult(outcome='passed', points=points_earned, msg=None))

def fake_seed(*args):
    pass
def fake_rseed(*args):
    pass

if __name__ == "__main__":
    pytest.main(["-s", __file__])
