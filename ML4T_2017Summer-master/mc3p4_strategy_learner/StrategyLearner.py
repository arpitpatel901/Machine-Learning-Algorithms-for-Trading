"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""

import datetime as dt
import numpy as np
import pandas as pd
import util as ut
import random
import QLearner as ql

class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False):
        self.verbose = verbose
        NumBins=5
        NumStates=(NumBins+1)*(NumBins+1)*(NumBins+1)*(NumBins+1)*3
        self.learner = ql.QLearner(num_states = NumStates,num_actions =5,alpha = 0.2,gamma = 0.9,rar = 0.98,radr = 0.999,dyna = 0,verbose = False)

    def Environment(self,BB,HoldingState,action,Price,YesterdayPrice):
        FixedTransactionCost=9.95 #Commission
        FixedTransactionCost=0
        ##No Market Impact Considered
        ActionMapVal={0:-400,1:-200,2:0,3:200,4:400}
        HoldingStateMapHolding={0:-200,1:0,2:200}
        StateMapActionMap={-400:0,-200:1,0:2,200:3,400:4}
        StateMapHoldingState={-200:0,0:1,200:2}
        ##Ideally
        NewStateVal=ActionMapVal[action]+HoldingStateMapHolding[HoldingState]
        change=ActionMapVal[action]
        #Limit 1
        if NewStateVal>200:
            NewStateVal = 200
            change=200-HoldingStateMapHolding[HoldingState]
        if NewStateVal<-200:
            NewStateVal = -200
            change= -200-HoldingStateMapHolding[HoldingState]
        #Limit 2
        NewBB=BB-(change*Price)
        if NewBB<0: #Invalid trade , dont do anything
            NewStateVal=HoldingStateMapHolding[HoldingState]
            change=0
            NewBB=BB
        BB=NewBB
        ##Assign Rewards
        if (change!=0):
            Reward=((Price/YesterdayPrice)-1)*change - FixedTransactionCost ## TODO Change this to minus
        else: Reward= 0 ## if you dont do anything I will take money from you
        #Return the values
        HoldingState=StateMapHoldingState[NewStateVal]
        ChangeInPortVal=-change*Price
        result=np.array([HoldingState,Reward,StateMapActionMap[change],BB])
        return result

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000):
        sv=sv
        ActionMapVal={0:-400,1:-200,2:0,3:200,4:400}
        HoldingStateMapHolding={0:-200,1:0,2:200}
        StateMapActionMap={-400:0,-200:1,0:2,200:3,400:4}
        StateMapHoldingState={-200:0,0:1,200:2}
        # Read in adjusted closing prices for given symbols, date range
        syms=[symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices_all = prices_all.fillna(method='ffill')
        prices_all = prices_all.fillna(method='bfill')
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        #if self.verbose: print prices
        # Pre-process the data and make a table of Indicator values for each date: Your learner wont be know any future dates
        PriceValues=prices.values
        ## Simple Moving Average
        SMA_n1=5  ##short window
        SMA_n2=20 #long term window(use for Ballinger)
        SMA_n3=12 #for MACD: short term
        SMA_n4=26 #MACD: long term
        SMA1=pd.rolling_mean(prices,window=SMA_n1,min_periods=1)
        SMA2=pd.rolling_mean(prices,window=SMA_n2,min_periods=1)
        SMA3=pd.rolling_mean(prices,window=SMA_n3,min_periods=1)
        SMA4=pd.rolling_mean(prices,window=SMA_n4,min_periods=1)
        #SMA1=prices.rolling(window=SMA_n1,center=False).mean()
        #SMA2=prices.rolling(window=SMA_n2,center=False).mean()
        #if self.verbose: print SMA1
        ## Standard Deviation
        StdDev=pd.rolling_std(prices,window=SMA_n2,min_periods=1)
        # Now we can turn the SMA into an SMA ratio, which is more useful.
        sma_ratio = prices / SMA_n1
        # Calculate Relative Strength Index (14 day) for the entire date range for all symbols.
        lookback=5
        rs = prices.copy()
        rsi = prices.copy()
        # Calculate daily_rets for the entire period (and all symbols).
        daily_rets = prices.copy()
        daily_rets.values[1:, :] = prices.values[1:, :] - prices.values[:-1, :]
        daily_rets.values[0, :] = np.nan
        # Split daily_rets into a same-indexed DataFrame of only up days and only down days,
        # and accumulate the total-up-days-return and total-down-days-return for every day.
        up_rets = daily_rets[daily_rets >= 0].fillna(0).cumsum()
        down_rets = -1 * daily_rets[daily_rets < 0].fillna(0).cumsum()
        # Apply the sliding lookback window to produce for each day, the cumulative return
        # of all up days within the window, and separately for all down days within the window.
        up_gain = prices.copy()
        up_gain.ix[:, :] = 0
        up_gain.values[lookback:, :] = up_rets.values[lookback:, :] - up_rets.values[:-lookback, :]
        down_loss = prices.copy()
        down_loss.ix[:, :] = 0
        down_loss.values[lookback:, :] = down_rets.values[lookback:, :] - down_rets.values[:-lookback, :]
        # Now we can calculate the RS and RSI all at once.
        rs = (up_gain / lookback) / (down_loss / lookback)
        rsi = 100 - (100 / (1 + rs))
        rsi.ix[:lookback, :] = np.nan
        # An infinite value here indicates the down_loss for a period was zero (no down days), in which
        # case the RSI should be 100 (its maximum value).
        rsi[rsi == np.inf] = 100
        RSI1=rsi
        ###INDICATORS
        windowBuffer=SMA_n1
        Momentum= np.squeeze(np.asarray(PriceValues[:]))/np.squeeze(np.array(np.transpose(np.append(PriceValues[0:SMA_n1],PriceValues[:-SMA_n1])))) ## can change SMA_n1 to 6
        BallingerBand=np.squeeze(np.asarray(PriceValues-(SMA2.values-(2*StdDev.values)))/(4*StdDev.values)) #Out of 1
        MACD=np.squeeze(np.asarray(SMA3.values-SMA4.values))
        RSI=RSI1
        RSIV=np.squeeze(np.asarray(RSI.values))
        #MACD=np.squeeze(np.asarray(sma_ratio.values))
        #RSIV=np.squeeze(np.asarray(sma_ratio.values))
        Momentum=Momentum[windowBuffer:]
        BallingerBand=BallingerBand[windowBuffer:]
        MACD=MACD[windowBuffer:]
        RSIV=RSIV[windowBuffer:]

        #BIN THE data
        NumBins=5
        MomentumBin=np.histogram(Momentum,bins=NumBins)[1]
        BallingerBandBin=np.histogram(BallingerBand,bins=NumBins,range=[0,1])[1]
        MACDBin=np.histogram(MACD,bins=NumBins)[1]
        RSIBin=np.histogram(RSIV,bins=NumBins,range=[0,100])[1]
        MomentumInd=np.digitize(Momentum,MomentumBin) #STATE 1
        BallingerInd=np.digitize(BallingerBand,BallingerBandBin) #STATE 2
        MACDInd=np.digitize(MACD,MACDBin) #STATE 3
        RSIInd =np.digitize(RSIV,RSIBin) #STATE 4
        ##IMPLIMENT BETTER BINNING FOR BETTER ACCURACY
        # LData=len(BallingerBand) #iterations = length of data
        # bins=np.zeros((4,NumBins)) ## 4 indicatrs and 10 bins
        # MomentumS=Momentum[np.argsort(Momentum)]
        # BallingerBandS = BallingerBand[np.argsort(BallingerBand)]
        # MACDS = MACD[np.argsort(MACD)]
        # RSIS = RSIV[np.argsort(RSIV)]
        # RSIS=RSIV[np.argsort(RSIV)]
        # bins[0,:]=MomentumS[np.rint(np.linspace(0,LData-1,num=NumBins)).astype(int)]
        # bins[1,:]=BallingerBandS[np.rint(np.linspace(0,LData-1,num=NumBins)).astype(int)]
        # bins[2,:]=MACDS[np.rint(np.linspace(0,LData-1,num=NumBins)).astype(int)]
        # bins[3,:]=RSIS[np.rint(np.linspace(0,LData-1,num=NumBins)).astype(int)]
        # MomentumInd=np.digitize(Momentum,bins[0,:]) #STATE 1
        # BallingerInd=np.digitize(BallingerBand,bins[1,:]) #STATE 2
        # MACDInd=np.digitize(MACD,bins[2,:]) #STATE 3
        # RSIInd =np.digitize(RSIV,bins[3,:]) #STATE 4
        #print(MomentumInd)
        #STATE 5:
        #Position/HoldingState : -2,0,2 * 100 shares
        #Stateval : 0,1,2
        #Actions : -4,-2,0,2,4 *100 shares
        #ActionVal : 0,1,2,3,4
        # add your code to do learning here
        ##Instantiate the learner
        # NumStates=(NumBins+1)*(NumBins+1)*(NumBins+1)*(NumBins+1)*3
        # self.learner = ql.QLearner(num_states = NumStates,num_actions =5,alpha = 0.2,gamma = 0.9,rar = 0.98,radr = 0.999,dyna = 0,verbose = False)
        HoldingState=1#Initial State [You own 0 shares, State 5 = 5] and
        InitialState=MomentumInd[0]*BallingerInd[0]*MACDInd[0]*RSIInd[0]*HoldingState
        a=self.learner.querysetstate(InitialState)
        InitialReward=0
        r=InitialReward
        flag=1
        CumReturnLastEpoch=0
        StopThreshold=0.05
        PriceValues=np.squeeze(np.asarray(PriceValues))
        Epochs=0
        #For each epoch
        while flag:
            BB=sv
            #For the each epoch
            HoldingState=1#Initial State [You own 0 shares, State 5 = 5] and
            Epochs=Epochs+1
            for dt in range(len(Momentum)):
                #s=MomentumInd[dt]*BallingerInd[dt]*MACDInd[dt]*RSIInd[dt]*(HoldingState+1) - 1 # MY POSITION:Today morning based on todays price and my last days holding(BUY 200)
                PriceToday=PriceValues[dt+windowBuffer]
                OldHolding=HoldingState
                PriceYesterday=PriceValues[dt+windowBuffer-1]
                result=self.Environment(BB,HoldingState,a,PriceToday,PriceYesterday) # I decided during my day to take this action
                HoldingState=result[0] # This would bring my holding tomorrow to this day ,and Satan said it was a bad idea(reward)
                r=result[1] #Satan gave me this reward
                BB=result[3]
                s=MomentumInd[dt]*BallingerInd[dt]*MACDInd[dt]*RSIInd[dt]*(HoldingState+1) - 1 # MY POSITION:Today morning based on todays price and my last days holding(BUY 200)
                a = self.learner.query(s,r) # What I learnt by doing this action at this state
                if (dt==(len(Momentum)-1)):
                    a=StateMapActionMap[-HoldingStateMapHolding[OldHolding]]
                    BB=BB-(a*PriceToday)
                    # print('Last Action')
                    # print(a)
            CumRetThisEpoch=(BB/sv)-1
            if Epochs>=1000:
                flag=0
            if (abs(CumRetThisEpoch-CumReturnLastEpoch) < StopThreshold) and (CumRetThisEpoch>0): #Stopping criteria
                flag=0 #Stop looping
            CumReturnLastEpoch=CumRetThisEpoch
        if self.verbose: print volume

    def author(self):
        return 'apatel380'

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):
        ActionMapVal={0:-400,1:-200,2:0,3:200,4:400}
        HoldingStateMapHolding={0:-200,1:0,2:200}
        StateMapActionMap={-400:0,-200:1,0:2,200:3,400:4}
        StateMapHoldingState={-200:0,0:1,200:2}
        # Read in adjusted closing prices for given symbols, date range
        syms=[symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices_all = prices_all.fillna(method='ffill')
        prices_all = prices_all.fillna(method='bfill')
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        #if self.verbose: print prices
        # Pre-process the data and make a table of Indicator values for each date: Your learner wont be know any future dates
        PriceValues=prices.values
        ## Simple Moving Average
        SMA_n1=5  ##short window
        SMA_n2=20 #long term window(use for Ballinger)
        SMA_n3=12 #for MACD: short term
        SMA_n4=26 #MACD: long term
        SMA1=pd.rolling_mean(prices,window=SMA_n1,min_periods=1)
        SMA2=pd.rolling_mean(prices,window=SMA_n2,min_periods=1)
        SMA3=pd.rolling_mean(prices,window=SMA_n3,min_periods=1)
        SMA4=pd.rolling_mean(prices,window=SMA_n4,min_periods=1)
        #SMA1=prices.rolling(window=SMA_n1,center=False).mean()
        #SMA2=prices.rolling(window=SMA_n2,center=False).mean()
        #if self.verbose: print SMA1
        ## Standard Deviation
        StdDev=pd.rolling_std(prices,window=SMA_n2,min_periods=1)
        # Now we can turn the SMA into an SMA ratio, which is more useful.
        sma_ratio = prices / SMA_n1
        # Calculate Relative Strength Index (14 day) for the entire date range for all symbols.
        lookback=5
        rs = prices.copy()
        rsi = prices.copy()
        # Calculate daily_rets for the entire period (and all symbols).
        daily_rets = prices.copy()
        daily_rets.values[1:, :] = prices.values[1:, :] - prices.values[:-1, :]
        daily_rets.values[0, :] = np.nan
        # Split daily_rets into a same-indexed DataFrame of only up days and only down days,
        # and accumulate the total-up-days-return and total-down-days-return for every day.
        up_rets = daily_rets[daily_rets >= 0].fillna(0).cumsum()
        down_rets = -1 * daily_rets[daily_rets < 0].fillna(0).cumsum()
        # Apply the sliding lookback window to produce for each day, the cumulative return
        # of all up days within the window, and separately for all down days within the window.
        up_gain = prices.copy()
        up_gain.ix[:, :] = 0
        up_gain.values[lookback:, :] = up_rets.values[lookback:, :] - up_rets.values[:-lookback, :]
        down_loss = prices.copy()
        down_loss.ix[:, :] = 0
        down_loss.values[lookback:, :] = down_rets.values[lookback:, :] - down_rets.values[:-lookback, :]
        # Now we can calculate the RS and RSI all at once.
        rs = (up_gain / lookback) / (down_loss / lookback)
        rsi = 100 - (100 / (1 + rs))
        rsi.ix[:lookback, :] = np.nan
        # An infinite value here indicates the down_loss for a period was zero (no down days), in which
        # case the RSI should be 100 (its maximum value).
        rsi[rsi == np.inf] = 100
        RSI1=rsi
        ###INDICATORS
        windowBuffer=SMA_n1
        Momentum= np.squeeze(np.asarray(PriceValues[:]))/np.squeeze(np.array(np.transpose(np.append(PriceValues[0:SMA_n1],PriceValues[:-SMA_n1])))) ## can change SMA_n1 to 6
        BallingerBand=np.squeeze(np.asarray(PriceValues-(SMA2.values-(2*StdDev.values)))/(4*StdDev.values)) #Out of 1
        MACD=np.squeeze(np.asarray(SMA3.values-SMA4.values))
        RSI=RSI1
        RSIV=np.squeeze(np.asarray(RSI.values))
        #MACD=np.squeeze(np.asarray(sma_ratio.values))
        Momentum=Momentum[windowBuffer:]
        BallingerBand=BallingerBand[windowBuffer:]
        MACD=MACD[windowBuffer:]
        RSIV=RSIV[windowBuffer:]

        #BIN THE data
        NumBins=5
        MomentumBin=np.histogram(Momentum,bins=NumBins)[1]
        BallingerBandBin=np.histogram(BallingerBand,bins=NumBins,range=[0,1])[1]
        MACDBin=np.histogram(MACD,bins=NumBins)[1]
        RSIBin=np.histogram(RSIV,bins=NumBins,range=[0,100])[1]
        MomentumInd=np.digitize(Momentum,MomentumBin) #STATE 1
        BallingerInd=np.digitize(BallingerBand,BallingerBandBin) #STATE 2
        MACDInd=np.digitize(MACD,MACDBin) #STATE 3
        RSIInd =np.digitize(RSIV,RSIBin) #STATE 4
        ##IMPLIMENT BETTER BINNING FOR BETTER ACCURACY
        # LData=len(BallingerBand) #iterations = length of data
        # bins=np.zeros((4,NumBins)) ## 4 indicatrs and 10 bins
        # MomentumS=Momentum[np.argsort(Momentum)]
        # BallingerBandS = BallingerBand[np.argsort(BallingerBand)]
        # MACDS = MACD[np.argsort(MACD)]
        # RSIS = RSIV[np.argsort(RSIV)]
        # RSIS=RSIV[np.argsort(RSIV)]
        # bins[0,:]=MomentumS[np.rint(np.linspace(0,LData-1,num=NumBins)).astype(int)]
        # bins[1,:]=BallingerBandS[np.rint(np.linspace(0,LData-1,num=NumBins)).astype(int)]
        # bins[2,:]=MACDS[np.rint(np.linspace(0,LData-1,num=NumBins)).astype(int)]
        # bins[3,:]=RSIS[np.rint(np.linspace(0,LData-1,num=NumBins)).astype(int)]
        # MomentumInd=np.digitize(Momentum,bins[0,:]) #STATE 1
        # BallingerInd=np.digitize(BallingerBand,bins[1,:]) #STATE 2
        # MACDInd=np.digitize(MACD,bins[2,:]) #STATE 3
        # RSIInd =np.digitize(RSIV,bins[3,:]) #STATE 4
        #print(MomentumInd)
        #STATE 5:
        #Position/HoldingState : -2,0,2 * 100 shares
        #Stateval : 0,1,2
        #Actions : -4,-2,0,2,4 *100 shares
        #ActionVal : 0,1,2,3,4
        # add your code to do learning here
        ##Instantiate the learner
        # NumStates=(NumBins+1)*(NumBins+1)*(NumBins+1)*(NumBins+1)*3
        # self.learner = ql.QLearner(num_states = NumStates,num_actions =5,alpha = 0.2,gamma = 0.9,rar = 0.98,radr = 0.999,dyna = 0,verbose = False)
        trades = prices_all[[symbol,]]  # only portfolio symbols
        trades_SPY = prices_all['SPY']  # only SPY, for comparison later
        trades.values[:,:] = 0 # set them all to nothing
        HoldingState=1 #you initially own 0 shares
        ActionMapVal={0:-400,1:-200,2:0,3:200,4:400}
        BB=sv
        TotalTrades=0
        # actionsTaken=np.zeros((1,len(PriceValues)))
        # states=np.zeros((1,len(PriceValues)))
        date=0
        # HoldingStates=np.zeros((1,len(PriceValues)))
        for date in range(len(PriceValues)):
            if date < windowBuffer: #you havent reached the buffer window size
                trades.values[date,:] = 0
                # actionsTaken[0,date-1]=0
                # states[0,date]=88
                # HoldingStates[0,date-1]=88

            else:
                s=MomentumInd[date-windowBuffer]*BallingerInd[date-windowBuffer]*MACDInd[date-windowBuffer]*RSIInd[date-windowBuffer]*(HoldingState+1)-1
                # HoldingStates[0,date]=HoldingState
                # states[0,date]=s
                a=self.learner.querysetstate(s)
                # actionsTaken[0,date]=a
                PriceToday=PriceValues[date]
                PriceYesterday=PriceValues[date-1]
                OldHolding=HoldingState
                result=self.Environment(BB,HoldingState,a,PriceToday,PriceYesterday) # I decided during my day to take this action
                HoldingState=result[0] # This would bring my holding tomorrow to this day ,and Satan said it was a bad idea(reward)
                a=result[2]
                BB=result[3]
                if (date==(len(PriceValues)-1)):
                    a=StateMapActionMap[-HoldingStateMapHolding[OldHolding]]
                trades.values[date,:]=ActionMapVal[a]
                TotalTrades=TotalTrades+ActionMapVal[a]
        # trades.values[3,:] = 200 # add a BUY at the 4th date
        # trades.values[5,:] = -200 # add a SELL at the 6th date
        # trades.values[6,:] = 200 # add a SELL at the 7th date
        # trades.values[8,:] = -400 # add a BUY at the 9th date

        # print('HoldingStates')
        # print(HoldingStates)
        # print('Learner Repeated query')
        # print(self.learner.querysetstate(1500))
        # print('states')
        # print(states)
        # print('actionsTaken')
        # print(actionsTaken)
        # print('trades')
        # print(trades)
        if self.verbose: print type(trades) # it better be a DataFrame!
        if self.verbose: print trades
        if self.verbose: print prices_all
        return trades

if __name__=="__main__":
    print "One does not simply think up a strategy"
