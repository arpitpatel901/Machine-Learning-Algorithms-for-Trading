"""MC3-H1: Best4{LR,RT} - grading script.

Usage:
- Switch to a student feedback directory first (will write "points.txt" and "comments.txt" in pwd).
- Run this script with both ml4t/ and student solution in PYTHONPATH, e.g.:
    PYTHONPATH=ml4t:MC3-P1/jdoe7 python ml4t/mc3_p1_grading/grade_learners.py
"""

import pytest
from grading.grading import grader, GradeResult, time_limit, run_with_timeout, IncorrectOutput
# Make sure your RTLearner is in your PYTHONPATH before running this script
# The following line will be commented out in the batch grader, and replaced with our
# reference solution.
from RTLearner import RTLearner

import os
import sys
import traceback as tb

import numpy as np
import pandas as pd
from collections import namedtuple

import math

import time

# Grading parameters
# rmse_margins = dict(KNNLearner=1.10, BagLearner=1.10)  # 1.XX = +XX% margin of RMS error
# points_per_test_case = dict(KNNLearner=3.0, BagLearner=2.0)  # points per test case for each group
# seconds_per_test_case = 10  # execution time limit
seconds_per_test_case = 5

# More grading parameters (picked up by module-level grading fixtures)
max_points = 100.0  # 3.0 * 10 + 2.0 * 10
html_pre_block = True  # surround comments with HTML <pre> tag (for T-Square comments field)

# Test cases
Best4TestCase = namedtuple('Best4TestCase', ['description', 'group','max_tests','needed_wins','row_limits','col_limits','seed'])
best4_test_cases = [
    Best4TestCase(
        description="Test Case 1: Best4LinReg",
        group="best4lr",
        max_tests=15,
        needed_wins=10,
        row_limits=(10,1000),
        col_limits=(2,1000),
        seed=1489683274
        ),
    Best4TestCase(
        description="Test Case 2: Best4RT",
        group="best4rt",
        max_tests=15,
        needed_wins=10,
        row_limits=(10,1000),
        col_limits=(2,1000),
        seed=1489683274
        )
]

# Test functon(s)
@pytest.mark.parametrize("description,group,max_tests,needed_wins,row_limits,col_limits,seed", best4_test_cases)
def test_learners(description, group, max_tests, needed_wins, row_limits, col_limits, seed, grader):
    """Test data generation methods beat given learner.

    Requires test description, test case group, and a grader fixture.
    """

    points_earned = 0.0  # initialize points for this test case
    incorrect = True
    try:
        # Try to import KNNLearner (only once)
        # if not 'KNNLearner' in globals():
        #     from KNNLearner import KNNLearner
        dataX, dataY = None,None
        same_dataX, same_dataY = None,None
        diff_dataX, diff_dataY = None,None
        betterLearner, worseLearner = None, None
        if group=="best4rt":
            from gen_data import best4RT
            dataX, dataY = run_with_timeout(best4RT,seconds_per_test_case,(),{'seed':seed})
            same_dataX,same_dataY = run_with_timeout(best4RT,seconds_per_test_case,(),{'seed':seed})
            diff_dataX,diff_dataY = run_with_timeout(best4RT,seconds_per_test_case,(),{'seed':seed+1})
            betterLearner = RTLearner
            worseLearner = LinRegLearner
        else:
            from gen_data import best4LinReg
            dataX, dataY = run_with_timeout(best4LinReg,seconds_per_test_case,(),{'seed':seed})
            same_dataX, same_dataY = run_with_timeout(best4LinReg,seconds_per_test_case,(),{'seed':seed})
            diff_dataX, diff_dataY = run_with_timeout(best4LinReg,seconds_per_test_case,(),{'seed':seed+1})
            betterLearner = LinRegLearner
            worseLearner = RTLearner

        num_samples = dataX.shape[0]
        cutoff = int(num_samples*0.6)
        worse_better_err = []
        for run in range(max_tests):
            permutation = np.random.permutation(num_samples)
            train_X,train_Y = dataX[permutation[:cutoff]], dataY[permutation[:cutoff]]
            test_X,test_Y = dataX[permutation[cutoff:]], dataY[permutation[cutoff:]]
            better = betterLearner()
            worse = worseLearner()
            better.addEvidence(train_X,train_Y)
            worse.addEvidence(train_X,train_Y)
            better_pred = better.query(test_X)
            worse_pred = worse.query(test_X)
            better_err = np.linalg.norm(test_Y-better_pred)
            worse_err = np.linalg.norm(test_Y-worse_pred)
            worse_better_err.append( (worse_err,better_err) )
        worse_better_err.sort(lambda a,b: int((b[0]-b[1])-(a[0]-a[1])))
        better_wins_count = 0
        for worse_err,better_err in worse_better_err:
            if better_err < 0.9*worse_err:
                better_wins_count = better_wins_count+1
                points_earned += 5.0
            if better_wins_count >= needed_wins:
                break
        incorrect = False
        msgs = []
        if (dataX.shape[0] < row_limits[0]) or (dataX.shape[0]>row_limits[1]):
            incorrect = True
            msgs.append("    Invalid number of rows. Should be between {}, found {}".format(row_limits,dataX.shape[0]))
            points_earned = max(0,points_earned-20)
        if (dataX.shape[1] < col_limits[0]) or (dataX.shape[1]>col_limits[1]):
            incorrect = True
            msgs.append("    Invalid number of columns. Should be between {}, found {}".format(col_limits,dataX.shape[1]))
            points_earned = max(0,points_earned-20)
        if better_wins_count < needed_wins:
            incorrect = True
            msgs.append("    Better learner did not exceed worse learner. Expected {}, found {}".format(needed_wins,better_wins_count))
        if not(np.array_equal(same_dataY,dataY)) or not(np.array_equal(same_dataX,dataX)):
            incorrect = True
            msgs.append("    Did not produce the same data with the same seed.\n"+\
                        "      First dataX:\n{}\n".format(dataX)+\
                        "      Second dataX:\n{}\n".format(same_dataX)+\
                        "      First dataY:\n{}\n".format(dataY)+\
                        "      Second dataY:\n{}\n".format(same_dataY))
            points_earned = max(0,points_earned-20)
        if np.array_equal(diff_dataY,dataY) and np.array_equal(diff_dataX,dataX):
            incorrect = True
            msgs.append("    Did not produce different data with different seeds.\n"+\
                        "      First dataX:\n{}\n".format(dataX)+\
                        "      Second dataX:\n{}\n".format(diff_dataX)+\
                        "      First dataY:\n{}\n".format(dataY)+\
                        "      Second dataY:\n{}\n".format(diff_dataY))
            points_earned = max(0,points_earned-20)            
        if incorrect:
            inputs_str = "    Residuals: {}".format(worse_better_err)
            raise IncorrectOutput, "Test failed on one or more output criteria.\n  Inputs:\n{}\n  Failures:\n{}".format(inputs_str, "\n".join(msgs))
        else:
            avg_ratio = 0.0
            worse_better_err.sort(lambda a,b: int(np.sign((b[0]-b[1])-(a[0]-a[1]))))
            for we,be in worse_better_err[:10]:
                avg_ratio += (float(we) - float(be))
            avg_ratio = avg_ratio/10.0
            if group=="best4rt":
                grader.add_performance(np.array([avg_ratio,0]))
            else:
                grader.add_performance(np.array([0,avg_ratio]))
    except Exception as e:
        # Test result: failed
        msg = "Description: {} (group: {})\n".format(description, group)
        
        # Generate a filtered stacktrace, only showing erroneous lines in student file(s)
        tb_list = tb.extract_tb(sys.exc_info()[2])
        for i in xrange(len(tb_list)):
            row = tb_list[i]
            tb_list[i] = (os.path.basename(row[0]), row[1], row[2], row[3])  # show only filename instead of long absolute path
        tb_list = [row for row in tb_list if (row[0] == 'gen_data.py')]
        if tb_list:
            msg += "Traceback:\n"
            msg += ''.join(tb.format_list(tb_list))  # contains newlines
        elif 'grading_traceback' in dir(e):
            msg += "Traceback:\n"
            msg += ''.join(tb.format_list(e.grading_traceback))
        msg += "{}: {}".format(e.__class__.__name__, e.message)

        # Report failure result to grader, with stacktrace
        grader.add_result(GradeResult(outcome='failed', points=points_earned, msg=msg))
        raise
    else:
        # Test result: passed (no exceptions)
        grader.add_result(GradeResult(outcome='passed', points=points_earned, msg=None))

class LinRegLearner(object):

    def __init__(self, verbose = False):
        pass # move along, these aren't the drones you're looking for

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """

        # slap on 1s column so linear regression finds a constant term
        newdataX = np.ones([dataX.shape[0],dataX.shape[1]+1])
        newdataX[:,0:dataX.shape[1]]=dataX

        # build and save the model
        self.model_coefs, residuals, rank, s = np.linalg.lstsq(newdataX, dataY)
        
    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        return (self.model_coefs[:-1] * points).sum(axis = 1) + self.model_coefs[-1]

if __name__ == "__main__":
    pytest.main(["-s", __file__])
