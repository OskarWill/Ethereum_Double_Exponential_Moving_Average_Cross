
import requests
from datetime import datetime
import time

import math
import random
import ast
import hmac
import functools
import operator
import statistics
import sys
import os

import numpy as np

import bybit

import http.client 

from datetime import datetime

from pprint import pprint
import gspread
from oauth2client.service_account import ServiceAccountCredentials

from googleapiclient import discovery

import socket
#import smtpLib, ssl 
import email.mime.multipart 
#import MIMEMultipart
import imaplib
import email
from email.header import decode_header
import traceback 
from imaplib import IMAP4_SSL

import pandas as pd

import ta as ta


def EMA_signal(closes):
  ser = pd.Series(closes, copy=False)
#  print(ser)

#  df =  dropna(ser)

  technicals_trend = ta.trend.EMAIndicator(ser)
 # trend = ta.trend()
#  print(trend)
  
  EMA21 = ta.trend.ema_indicator(close=ser, window=21, fillna=True)
#  print(EMA21.iloc[-1])

  EMA9 = ta.trend.ema_indicator(close=ser, window=9, fillna=True)
#  print(EMA9.iloc[-1])

  return EMA9.iloc[-1],EMA21.iloc[-1]


  


class HistoricalPrice(object):

    def __init__(self, host, symbol, interval, timestamp, limit, client):

        self.host = host
        self.symbol = symbol
        self.interval = interval
        self.timestamp = timestamp - (48 * 3600)
        self.limit = limit 
        self.client = client

        self.url = '{}/v2/public/kline/list'.format(self.host)


    def api_historical_response(self):
      #  r = self.client.LinearKline.LinearKline_get(symbol="BTCUSDT", interval=self.interval, limit=None, **{'from':int(self.timestamp)}).result() #IS A TYPE TUPLE FOR SOME REASON
        r = self.client.Kline.Kline_get(symbol=self.symbol, interval=self.interval, **{'from':self.timestamp}).result()
    
        for entries in r:
            self.results = entries['result']
#            print(entries)
            return self.results

    def volume(self):

        volumes = []

        for result in self.results:
            volumes.append(result['volume'])

            
        return volumes
        

    def price_close(self):

        closes = []


        for result in self.results:
            closes.append(float(result['close']))

        return closes

    def price_open(self):
        opens = [] 

        for result in self.results:
            opens.append(float(result['open']))

        return opens

    def price_high(self):
        highs = []

        for result in self.results:
            highs.append(float(result['high']))

        return highs

    def price_low(self):
        lows = []

        for result in self.results:
            lows.append(float(result['low']))

        return lows

    def candles(self):
        candlez = []

        for result in self.results:
            candlez.append((float(result['open']),float(result['close']),float(result['high']),float(result['low']))) 

        return candlez


class LivePrice(object):

    def __init__(self, host, param_str, symbol, interval, timestamp):
        self.host = host
         
        self.symbol = symbol
        
        self.url = "{}/v2/public/tickers".format(self.host)




    def price_response(self):
        r = requests.get(self.url)                  #TODO: Insert URL for response
        response = r.text
        response_dict = ast.literal_eval(response)

        return (response_dict)

    def price_dict(self):
        self.response_dict = self.price_response()
        dict_result = list(self.response_dict["result"])

   #     return dict_result

        for result in dict_result:
            if result['symbol'] == self.symbol:
                price = result['last_price']


        return float(price)



class timeStamp(object):

    def __init__(self, client):
        self.client = client
        
    def api_time_request(self):
        r = self.client.Common.Common_get().result()[0]
        time = float(r['time_now'])

 #       print('API TIME: ' + str(r))
        return int(time)

    



def get_signature(api_secret,params):
    '''Encryption Signature'''

    _val = '&'.join([str(k)+"="+str(v) for k, v in sorted(params.items()) if (k != 'sign') and (v is not None)])
    # print(_val)
    return str(hmac.new(bytes(api_secret, "utf-8"), bytes(_val, "utf-8"), digestmod="sha256").hexdigest())

class ExecuteOrder(object):

    def __init__(self,client,symbol,side,size,price,take_profit,stop_loss):

        self.client = client
        self.symbol = symbol
        self.side = side
 #       print(int(price))
  #      print(float(size))
        self.size = size #int(round(int(price) * float(size),0))
        self.price = int(round(price,0))
   #     print(self.size)

        self.take_profit = int(round(take_profit,0))

   #     print(int(self.take_profit))
    #    print(int(self.size))
     #   print(int(self.price))
        self.stop_loss = stop_loss

            
 
        

    def order(self):

     #   client_order = client.Order.Order_newV2(side=self.side,symbol=self.symbol,order_type="Limit",qty=int(self.size),price=int(self.price),time_in_force="ImmediateOrCancel",
     #                                           take_profit=int(self.take_profit),stop_loss=self.stop_loss,order_link_id=None).result()

        client_order = client.Order.Order_newV2(side=self.side,symbol=self.symbol,order_type="Limit",qty=int(self.size),price=int(self.price),time_in_force="FillOrKill", stop_loss=self.stop_loss,order_link_id=None).result()

      #  print(client.Order.Order_new(side=self.side,symbol=self.symbol,order_type="Limit",qty=self.size,price=self.price,time_in_force="PostOnly", take_profit=(self.take_profit),stop_loss=self.stop_loss).result())

  #      client_order = client.LinearOrder.LinearOrder_new(side=self.side,symbol=self.symbol,order_type="Market",qty=self.size,price=self.price,time_in_force="FillOrKill",reduce_only=False,take_profit=int(self.take_profit),stop_loss=self.stop_loss,close_on_trigger=False).result()
  #      client_order = client.LinearConditional.LinearConditional_new(stop_px=self.stop_px, side=self.side,symbol=self.symbol,order_type="Limit",qty=self.size,base_price=self.base_price, price=self.price,time_in_force="PostOnly",reduce_only=False,take_profit=int(self.take_profit),stop_loss=self.stop_loss,close_on_trigger=False).result())
  #      print(client_order)


        return (client_order)
     #   for entries in client_order:
     #       results = entries['result']
     #       order_id = results['order_id']
     #       return order_id

      #  print(type(client_order))
     #   result = client_order['result']
     #   print("ORDER RESULT: " + str(result))
     #   order_id = result['order_id']
        
        
      #  return client_order

   
 

        


class Position(object):

    def __init__(self,host,param_str,symbol):

        self.client = client
        self.host = host
        self.params = param_str
        self.symbol = symbol

        self.url = '{}/v2/private/position/list?{}'.format(self.host,self.params)
        

    def wrapper_position(self):
        previous = []
        
        try:
            r = self.client.Positions.Positions_myPosition().result()
            for entries in r:
                results = entries['result']
                for result in results:
                    if result['symbol'] == self.symbol:
                        return float(result['position_value'])
        except Exception as e:
            print("Main program position error: " + str(e))
                
         
        


    def HTTP_connect_position(self):
        '''NOT IN USE'''        
        print("position host: " + str(self.host))
        print("position params: " + str(self.params))
        r = requests.get(self.url)
        response = r.text
        try:
           response_dict = ast.literal_eval(response)
           dict_result = response_dict['result']
           for result in dict_result.values():
               if result == self.symbol:
                  position_value = dict_result['position_value']
 
           if int(position_value) > 0:
                return True
           else:
                return False
 
        except Exception:
            server_time = int(response[143:156])
            recv_window = int(response[170:174])

            x = server_time - recv_window
            y = server_time + 1000

            print("Timestamp must be greater than this: " + str(server_time - recv_window))
            print("Timestamp must be less than this: " + str(server_time + 1000))

            midpoint = int((y+x)/2)

            print("MIDPOINT: " + str(midpoint))
            

       #     if server_time - recv_window <= timestamp < server_time + 1000:
        #            return timestamp

            
            
            return response

class Wallet(object):

    def __init__(self,client,host,param_str,symbol):

        self.client = client
        self.host = host
        self.params = param_str
        self.symbol = symbol

        self.url = '{}/v2/private/wallet/balance?{}'.format(self.host,self.params)

 

    def HTTP_connect_wallet(self):
        '''NOT IN USE'''
        r = requests.get(self.url)                   
        response = r.text
        try:
           response_dict = ast.literal_eval(response)
           dict_result = response_dict['result']
           for result in dict_result.keys():
                if result == self.symbol[0:3]:
                    balance = dict_result[result]['available_balance']
                
                
     #      print(response) 
           return balance
        except Exception:
            return response

    def wrapper_wallet(self):
        new_wallet = client.Wallet.Wallet_getBalance(coin="ETH").result()

        try:
            for new in new_wallet:
                result = new['result']
                ETH = result['ETH']
                available_balance = ETH['available_balance']
                return float(available_balance)
        except Exception:
            self.wrapper_wallet()
        
    #    return "done" 
        
         
 

def LB(SMA,closes):
    '''Lower Bollinger Band'''
    return SMA - (statistics.stdev(closes)*2)

def UB(SMA,closes):
    '''Upper Bollinger Band'''
    return SMA + (statistics.stdev(closes) * 2)


def SMA(closes):
    '''20 Day Simple Moving Average Calculation'''

    return sum(closes) / len(closes)

def EMA(closes, exponential_averages):
    '''Exponential moving averages over a period of time'''

    multiplier = (2/(len(closes) + 1))

    previous_day = SMA(closes)

    if len(exponential_averages) == 0:
        previous_day = SMA(closes)
    else:
        previous_day = exponential_averages[-1]

    EMA_calculation = ((closes[-1] - previous_day) * multiplier) + previous_day

    if len(exponential_averages) < 1 or exponential_averages[-1] != EMA_calculation:
        exponential_averages.append(EMA_calculation)

   # print((closes[-1] - previous_day) * multiplier + previous_day)
   # print((1 - multiplier) * previous_day + multiplier * closes[-1] )
   # print(closes[-1] * multiplier + previous_day * (1 - multiplier))

  #  print(str(len(closes)) + " EMA: " + str(EMA_calculation))

    return EMA_calculation

   # return (1 - multiplier) * previous_day + multiplier * closes[-1] 

    #return closes[-1] * multiplier + previous_day * (1 - multiplier)



def short_entry_candle(candles,downwards_candle):

    if candles[1] < candles[0]:     #bearish candle
        #print('short entry candle reached')
        if len(downwards_candle) == 0 or candles[3] != downwards_candle[-1]:
            downwards_candle.append(candles[3])
        return True
    else:
        return False 


def long_entry_candle(candles,upwards_candle):

    if candles[1] > candles[0]:       # bullish candle
        #print('long entry candle reached')
        if len(upwards_candle) == 0 or candles[2] != upwards_candle[-1]:
            upwards_candle.append(candles[2])
        return True
        
    else:
        return False



def MACD(closes, MACD_crosses, EMA12, EMA26):

    
    MAC_D = EMA(closes[-9:-1],EMA12) - EMA(closes[-22:-1],EMA26) # 11 and 25

  #  print(len(closes[-12:-1]))
  #  print(len(closes[-26:-1]))

  #  print("EMA26: " + str(EMA26))
  #  print("EMA12: " + str(EMA12))

    if MAC_D not in MACD_crosses:
        MACD_crosses.append(MAC_D)
        MACD_crosses = MACD_crosses[-3:-1]
      #  print(MACD_crosses)
      #  print("EMA 12: " + str(EMA12[-1]))
      #  print("EMA 26: " + str(EMA26[-1]))

def live_api_time():
    
    api_time = client.Common.Common_getTime().result()[0]
    for time in api_time:
        if time == "time_now":
            api_time = float(api_time["time_now"])+ 160800 #-(1*6000)
            return(api_time)


def time_period(client):

    #May 1 - August 1st Consolidation 
  

    today = datetime.today()   
    
    #january 10 - February 10 

    start_date = datetime(2019, 4, 23) #YY / MM / DD           Check file to see if theres an existing datetime and entry for latest EMA
  
    print(start_date)
    start =  int(start_date.timestamp())   #+ 170200 - 9390)# * 1000)) #date in millisecond timestamp
    print(start)


    end_date = datetime(2021, 10, 22)

    time_stamp = timeStamp(client)
  #  api_time = time_stamp.api_time_request() + 137000

    end = int(end_date.timestamp())# + 137000)

    print(end)
#    end.append(api_time)

    return start, end
 
    

def entry(sheet, api_time, row, price, side):


#  sheet.get(3)

  row.append(row[-1] + 1) 


  date = datetime.fromtimestamp(api_time)

  #Get corresponding cell

#  cell = sheet.cell(row[-1],1).value #row, column
  sheet.update_cell(row[-1],1, "Side: " + str(side))
  sheet.update_cell(row[-1],2, "Price: " + str(price))
#  sheet.update_cell(row[-1],8, "MACD: " + str(MACD_crosses[-1]))
 
  sheet.update_cell(row[-1],14, "Datetime: " + str(datetime.now()))


#  sheet.update_cell(row[-1],13, "Balance: " + str(balance))


def stoploss(sheet,start,row,stop_loss,first_reduction,second_reduction, third_reduction, fourth_reduction):

#  date = datetime.fromtimestamp(start)

  if first_reduction == False:
      sheet.update_cell(row[-1],8, str(stop_loss))
  #    sheet.update_cell(row[-1],13, str(balance))
 
  elif first_reduction == True and second_reduction == False:
      sheet.update_cell(row[-1],9, str(stop_loss))
  #    sheet.update_cell(row[-1],13, str(balance))
 

  elif second_reduction == True and third_reduction == False: 
      sheet.update_cell(row[-1],10, str(stop_loss))
   #   sheet.update_cell(row[-1],13, str(balance))
 

  elif third_reduction == True and fourth_reduction == False:
      sheet.update_cell(row[-1],11, str(stop_loss))
   #   sheet.update_cell(row[-1],13, str(balance))
 

  elif fourth_reduction == True:
      sheet.update_cell(row[-1],12, str(stop_loss))
    #  sheet.update_cell(row[-1],13, str(balance)) 
 

def takeprofit(sheet, start, row, take_profit, first_reduction, second_reduction, third_reduction, fourth_reduction):

 # date = datetime.fromtimestamp(start)   

  if first_reduction == False:
      sheet.update_cell(row[-1],3, str(take_profit))
  #    sheet.update_cell(row[-1],13, str(balance))
  elif first_reduction == True and second_reduction == False:
      sheet.update_cell(row[-1],4, str(take_profit))
 #     sheet.update_cell(row[-1],13, str(balance))
  elif second_reduction == True and third_reduction == False:
      sheet.update_cell(row[-1],5, str(take_profit))
 #     sheet.update_cell(row[-1],13, str(balance))     
  elif third_reduction == True and fourth_reduction == False:
      sheet.update_cell(row[-1],6, str(take_profit))
  #    sheet.update_cell(row[-1],13, str(balance)) 

  elif fourth_reduction == True:
    sheet.update_cell(row[-1],7, str(take_profit))
  #  sheet.update_cell(row[-1],13, str(balance)) 


def taker_order(price,quantity,balance):

    return ((quantity/price) * 1.075)

def maker_order(price,quantity,balance):

    return ((quantity/price) * 1.025)


def stop_timer(early_close, host, symbol, interval, api_time, limit, client):

  while True:
    time_stamp = timeStamp(client)
    api_time = time_stamp.api_time_request() - 6000
    historical_price = HistoricalPrice(host, symbol, interval, api_time, limit, client)
    api_historical_response = historical_price.api_historical_response()
    closes = historical_price.price_close()
    if early_close != closes[-2]:
      break 




def trade(host, param_str, symbol, interval, timestamp, params, limit, client, api_time, api_key, signature, sheet, api_secret):
    """The Actual Strategy """

    #TODO:
    #Fix Break Even entries
    #Bring 
    #Test?

#    print(sheet.get('A25'))

    #5.9% TP
    #2% SL
    #30 Min Time Frame 

 #   print(sheet.get('A1'))




    #Backtesting objects
    row = [2]
    sent_requests = 0
 
    minute = 60 * 30
    start, end = time_period(client)
    sent_requests = 0


 

    #Strategy objects


    
    
    break_even = []

    
    short_cross = False
    long_cross = False

    first_reduction = False
    second_reduction = False
    third_reduction = False
    fourth_reduction = False

    long_position = 0
    short_position = 0

    historical_price = HistoricalPrice(host, symbol, interval, api_time, limit, client)
    api_historical_response = historical_price.api_historical_response()
    closes = historical_price.price_close()
     

    EMA9, EMA21 = EMA_signal(closes)

    print("EMA9: " + str(EMA9))
    print("EMA21: " + str(EMA21))






    while True:       
      try:
        if datetime.now().minute == 00 or datetime.now().minute == 30:
          time_stamp = timeStamp(client)
          api_time = time_stamp.api_time_request() - 6000
          historical_price = HistoricalPrice(host, symbol, interval, api_time, limit, client)
          api_historical_response = historical_price.api_historical_response()
          closes = historical_price.price_close()
          candles = historical_price.candles()
          MA55 = SMA(closes[-56:-1])
      #    print("latest close: " + str(closes[-1]))
          EMA9, EMA21 = EMA_signal(closes)
      #    print("9 Exponential Moving Average " + str(EMA9))
      #    print("21 Exponential Moving Average " + str(EMA21))
      #    print("55 Moving average " + str(MA55))


          

          if long_position == 0 and short_position == 0:
              if closes[-1] > EMA9 and EMA9 > EMA21 and EMA21 > MA55: 
                  break_even[:] = []
                  first_reduction = False
                  second_reduction = False
                  third_reduction = False
                  fourth_reduction = False
                  side = "Buy"

                  
            #      reduction_2 = price + (distance * 5)          # At 5R Move stop loss to 4R 

           #       final_trailing_stop = price + (distance * 4)


                  
              #    wallet = Wallet(client,host,param_str,symbol)
              #    balance = wallet.wrapper_wallet()

               #   while balance == None:
              #        balance = wallet.wrapper_wallet()

               #   live_price = LivePrice(host, param_str, symbol, interval, timestamp)
               #   current_price = live_price.price_dict()

                  stop_loss = round(closes[-1] - (closes[-1] * 0.02),0)

                #  print("stop loss: " + str(stop_loss))


                  take_profit_1 = round(closes[-1] * 1.029,0)
                  take_profit_2 = round(closes[-1] * 1.059,0)
                  take_profit_3 = round(closes[-1] * 1.079,0)
                  take_profit_4 = round(closes[-1] * 1.099,0)
                  break_even.append(closes[-1])


                #  second_stop = round(stop_loss - (distance/2),0)

             #     size = (float(balance) * current_price) * 0.95
                  size = 15
                  balance = 15

             #     print("balance: " + str(balance))
               #   print("distance: " + str(distance))

                #  size = float((0.01 * balance) / (distance)) * price
                #  size = int(balance) - 1
               #   print("order size: " + str(size))


                  
                #  order = ExecuteOrder(client,symbol,side,size,current_price,take_profit_4,stop_loss)
                #  execute = order.order()
               #   print("LONG ORDER: " + str(execute))                  


                  
                #  positions = Position(host,param_str,symbol)

                  print("Side: " + str(side))

                  print("Take profit 1: " + str(take_profit_1))
                  print("Take profit 2: " + str(take_profit_2))

                  print("Take profit 3: " + str(take_profit_3))

                  print("Take profit 4: " + str(take_profit_4))
                  print("Stop loss: " + str(stop_loss))
                  print("Break even: " + str(closes[-1]))
                  print("9 Exponential Moving Average " + str(EMA9))
                  print("21 Exponential Moving Average " + str(EMA21))
                  print("55 Moving average " + str(MA55))
                 

                    

                  entry(sheet, api_time, row, closes[-1], side)
                  sent_requests += 3



                      
                  long_position = 1
              



   


              if closes[-1] < EMA9 and EMA9 < EMA21 and EMA21 < MA55: 
                  break_even[:] = []
                  first_reduction = False
                  second_reduction = False
                  third_reduction = False
                  fourth_reduction = False 
                  side = "Sell"

                  
          #        reduction_2 = price - (distance * 5)          # At 5R Move stop loss to 4R
                  

         #         final_trailing_stop = price - (distance * 4)          #Price moved to after price hits 5R


                  

              #    wallet = Wallet(client,host,param_str,symbol)
              #    balance = wallet.wrapper_wallet()

              #    while balance == None:
              #        balance = wallet.wrapper_wallet()
                      
             #     live_price = LivePrice(host, param_str, symbol, interval, timestamp)
            #      current_price = live_price.price_dict()
                  stop_loss = round(closes[-1] * 1.02,0)

                #  print("stop loss: " + str(stop_loss))

               #   distance = stop_loss - current_price

         #         second_stop = stop_loss + (distance/2)

                  take_profit_1 = round(closes[-1] - (closes[-1] * 0.029),0)
                  take_profit_2 = round(closes[-1] - (closes[-1] * 0.059),0)
                  take_profit_3 = round(closes[-1] - (closes[-1] * 0.079),0)
                  take_profit_4 = round(closes[-1] - (closes[-1] * 0.099),0)

                  print("Side: " + str(side))

                  print("Take profit 1: " + str(take_profit_1))
                  print("Take profit 2: " + str(take_profit_2))

                  print("Take profit 3: " + str(take_profit_3))

                  print("Take profit 4: " + str(take_profit_4))
                  print("Stop loss: " + str(stop_loss))
                  print("Break even: " + str(closes[-1]))
                  print("9 Exponential Moving Average " + str(EMA9))
                  print("21 Exponential Moving Average " + str(EMA21))
                  print("55 Moving average " + str(MA55))




                  break_even.append(closes[-1])
                  entry(sheet, start, row, closes[-1], side)
                  sent_requests += 3
                  short_position = 1


            

                                    
          if long_position > 0 or short_position > 0:   
                if side == "Buy":
                    if long_position > 0 and first_reduction == False and candles[-1][3] <= stop_loss:
                      stoploss(sheet,start,row,stop_loss,first_reduction,second_reduction, third_reduction, fourth_reduction)
                      sent_requests += 1
                      long_position = 0
                      while True:
                        if  EMA9 > MA55 and closes[-1] < EMA9:
                            print("Position re-entry confirmed.")
                            break 
                        elif EMA9 < MA55 and closes[-1] > EMA9:
                            print("Position re-entry confirmed.")
                            break
                        else:
                            time_stamp = timeStamp(client)
                            api_time = time_stamp.api_time_request() - 6000
                            historical_price = HistoricalPrice(host, symbol, interval, api_time, limit, client)
                            closes = historical_price.price_close()
                            EMA9, EMA21 = EMA_signal(closes)
                            MA55 = SMA(closes[-56:-1])
                 

                    if long_position > 0 and candles[-1][2] >= take_profit_1:
                        if first_reduction == False:
                            sent_requests += 1
                            takeprofit(sheet, start, row, take_profit_1, first_reduction, second_reduction, third_reduction, fourth_reduction)
                        #    print(client.Positions.Positions_tradingStop(symbol=symbol,take_profit="0", stop_loss=str(break_even[-1]), trailing_stop="0", new_trailing_active="0").result())
                      #      stop_moves = ChrisVBTC.trailing_stop(side, take_profit_1, take_profit_2, break_even[-1], symbol)

                     #       stop_move_1 = stop_moves.stop_move_1()

                            first_reduction = True
                        #    position = 0

                    if long_position > 0 and candles[-1][2] >= take_profit_2:
                        if second_reduction == False:
                          takeprofit(sheet, start, row, take_profit_2, first_reduction, second_reduction, third_reduction, fourth_reduction)
                          sent_requests += 1

                  #          limit_stops(break_even[-1],reduction,client,size,symbol,side)
                         #   print(client.Conditional.Conditional_new(order_type="Limit",side="Sell",symbol=symbol,qty=int(size),price=reduction,base_price=reduction + 1,stop_px=reduction - 1,time_in_force="GoodTillCancel", order_link_id=None,close_on_trigger=True).result())
                       #     print(client.Positions.Positions_tradingStop(symbol=symbol,take_profit="0", stop_loss=str(take_profit_1), trailing_stop="0", new_trailing_active="0").result())
                      #      stop_moves = ChrisVBTC.trailing_stop(side, take_profit_1, take_profit_2, break_even[-1], symbol)

                     #       stop_move_2 = stop_moves.stop_move_2()
                          second_reduction = True

                    if long_position > 0 and candles[-1][2] >= take_profit_3:
                        if third_reduction == False:
                          takeprofit(sheet, start, row, take_profit_3, first_reduction, second_reduction, third_reduction, fourth_reduction)
                          sent_requests += 1
                      #    print(client.Positions.Positions_tradingStop(symbol=symbol,take_profit="0", stop_loss=str(take_profit_2), trailing_stop="0", new_trailing_active="0").result())
                      #    stop_moves = ChrisVBTC.trailing_stop(side, take_profit_1, take_profit_2, break_even[-1], symbol)

                    #      stop_move_3 = stop_moves.stop_move_3()
                          third_reduction = True

                    if long_position > 0 and candles[-1][2] >= take_profit_4: 
                      if fourth_reduction == False:
                        takeprofit(sheet, start, row, take_profit_4, first_reduction, second_reduction, third_reduction, fourth_reduction)
                        sent_requests += 1
                      #  print(client.Positions.Positions_tradingStop(symbol=symbol,take_profit="0", stop_loss=str(take_profit_3), trailing_stop="0", new_trailing_active="0").result())
                        fourth_reduction = True
                        long_position = 0
                        while True:
                          if  EMA9 > MA55 and closes[-1] < EMA9:
                              print("Position re-entry confirmed.")
                              break 
                          elif EMA9 < MA55 and closes[-1] > EMA9:
                              print("Position re-entry confirmed.")
                              break
                          else:
                              time_stamp = timeStamp(client)
                              api_time = time_stamp.api_time_request() - 6000
                              historical_price = HistoricalPrice(host, symbol, interval, api_time, limit, client)
                              closes = historical_price.price_close()
                              EMA9, EMA21 = EMA_signal(closes)
                              MA55 = SMA(closes[-56:-1])


                    if long_position > 0 and first_reduction == True and second_reduction == False and  candles[-1][3] <= break_even[-1]:
                      stoploss(sheet,start,row,break_even[-1],first_reduction,second_reduction, third_reduction, fourth_reduction)
                      sent_requests += 1
                      long_position = 0 
                      while True:
                        if  EMA9 > MA55 and closes[-1] < EMA9:
                            print("Position re-entry confirmed.")
                            break 
                        elif EMA9 < MA55 and closes[-1] > EMA9:
                            print("Position re-entry confirmed.")
                            break
                        else:
                            time_stamp = timeStamp(client)
                            api_time = time_stamp.api_time_request() - 6000
                            historical_price = HistoricalPrice(host, symbol, interval, api_time, limit, client)
                            closes = historical_price.price_close()
                            EMA9, EMA21 = EMA_signal(closes)
                            MA55 = SMA(closes[-56:-1])


                    if long_position > 0 and second_reduction == True and third_reduction == False and candles[-1][3] <= take_profit_1: 
                      stoploss(sheet,start,row,take_profit_1,first_reduction,second_reduction, third_reduction, fourth_reduction)
                      sent_requests += 1
                      long_position = 0
                      while True:
                        if  EMA9 > MA55 and closes[-1] < EMA9:
                            print("Position re-entry confirmed.")
                            break 
                        elif EMA9 < MA55 and closes[-1] > EMA9:
                            print("Position re-entry confirmed.")
                            break
                        else:
                            time_stamp = timeStamp(client)
                            api_time = time_stamp.api_time_request() - 6000
                            historical_price = HistoricalPrice(host, symbol, interval, api_time, limit, client)
                            closes = historical_price.price_close()
                            EMA9, EMA21 = EMA_signal(closes)
                            MA55 = SMA(closes[-56:-1])


                    if long_position > 0 and third_reduction == True and fourth_reduction == False and candles[-1][3] <= take_profit_2:
                      stoploss(sheet,start,row,take_profit_2,first_reduction,second_reduction, third_reduction, fourth_reduction)
                      sent_requests += 1
                      long_position = 0
                      while True:
                        if  EMA9 > MA55 and closes[-1] < EMA9:
                            print("Position re-entry confirmed.")
                            break 
                        elif EMA9 < MA55 and closes[-1] > EMA9:
                            print("Position re-entry confirmed.")
                            break
                        else:
                            time_stamp = timeStamp(client)
                            api_time = time_stamp.api_time_request() - 6000
                            historical_price = HistoricalPrice(host, symbol, interval, api_time, limit, client)
                            closes = historical_price.price_close()
                            EMA9, EMA21 = EMA_signal(closes)
                            MA55 = SMA(closes[-56:-1])
            

                if side == "Sell":

                    if short_position > 0 and first_reduction == False and candles[-1][2] >= stop_loss:
                      stoploss(sheet,start,row,stop_loss,first_reduction,second_reduction, third_reduction, fourth_reduction)
                      sent_requests += 1
                      short_position = 0
                      while True:
                        if  EMA9 > MA55 and closes[-1] < EMA9:
                            print("Position re-entry confirmed.")
                            break 
                        elif EMA9 < MA55 and closes[-1] > EMA9:
                            print("Position re-entry confirmed.")
                            break
                        else:
                            time_stamp = timeStamp(client)
                            api_time = time_stamp.api_time_request() - 6000
                            historical_price = HistoricalPrice(host, symbol, interval, api_time, limit, client)
                            closes = historical_price.price_close()
                            EMA9, EMA21 = EMA_signal(closes)
                            MA55 = SMA(closes[-56:-1])

                    

                    if short_position > 0 and candles[-1][3] <= take_profit_1:
                        if first_reduction == False:
                            takeprofit(sheet, start, row, take_profit_1, first_reduction, second_reduction, third_reduction, fourth_reduction)
                            sent_requests += 1
                   #         limit_stops(break_even[-1],stop_loss,client,size,symbol,side)
                    #        print(client.Conditional.Conditional_new(order_type="Limit",side="Buy",symbol=symbol,qty=int(size),price=break_even[-1],base_price=break_even[-1] - 1,stop_px=break_even[-1] + 1,time_in_force="GoodTillCancel", order_link_id=None,close_on_trigger=True).result())
                   #         print(client.Positions.Positions_tradingStop(symbol=symbol,take_profit="0", stop_loss=str(break_even[-1]), trailing_stop="0", new_trailing_active="0").result())
                       #     stop_moves = ChrisVBTC.trailing_stop(side, take_profit_1, take_profit_2, break_even[-1], symbol)

                       #     stop_move_3 = stop_moves.stop_move_3()
                            first_reduction = True
                      #      position = 0

                    if short_position > 0 and candles[-1][3] <= take_profit_2:
                        if second_reduction == False:
                            takeprofit(sheet, start, row, take_profit_2, first_reduction, second_reduction, third_reduction, fourth_reduction)
                            sent_requests += 1

                   #         limit_stops(reduction,break_even[-1],client,size,symbol,side)
                     #       print(client.Conditional.Conditional_new(order_type="Limit",side="Buy",symbol=symbol,qty=int(size),price=reduction,base_price=reduction - 1,stop_px=reduction + 1,time_in_force="GoodTillCancel", order_link_id=None,close_on_trigger=True).result())
                        #    stop_moves = ChrisVBTC.trailing_stop(side, take_profit_1, take_profit_2, break_even[-1], symbol)

                        #    stop_move_2 = stop_moves.stop_move_2()
                   #         print(client.Positions.Positions_tradingStop(symbol=symbol,take_profit="0", stop_loss=str(take_profit_1), trailing_stop="0", new_trailing_active="0").result())
                            second_reduction = True

                    if short_position > 0 and candles[-1][3] <= take_profit_3:
                        if third_reduction == False:
                          takeprofit(sheet, start, row, take_profit_3, first_reduction, second_reduction, third_reduction, fourth_reduction)
                          sent_requests += 1
                     #     stop_moves = ChrisVBTC.trailing_stop(side, take_profit_1, take_profit_2, break_even[-1], symbol)

                    #      stop_move_3 = stop_moves.stop_move_3()
                    #      print(client.Positions.Positions_tradingStop(symbol=symbol,take_profit="0", stop_loss=str(take_profit_2), trailing_stop="0", new_trailing_active="0").result())
                          third_reduction = True

                    if short_position > 0 and candles[-1][3] <= take_profit_4: 
                      if fourth_reduction == False:
                        takeprofit(sheet, start, row, take_profit_4, first_reduction, second_reduction, third_reduction, fourth_reduction)
                        sent_requests += 1
                      #  print(client.Positions.Positions_tradingStop(symbol=symbol,take_profit="0", stop_loss=str(take_profit_3), trailing_stop="0", new_trailing_active="0").result())
                        fourth_reduction = True
                        short_position = 0 
                        while True:
                          if  EMA9 > MA55 and closes[-1] < EMA9:
                            print("Position re-entry confirmed.")
                            break 
                          elif EMA9 < MA55 and closes[-1] > EMA9:
                            print("Position re-entry confirmed.")
                            break
                          else:
                            time_stamp = timeStamp(client)
                            api_time = time_stamp.api_time_request() - 6000
                            historical_price = HistoricalPrice(host, symbol, interval, api_time, limit, client)
                            closes = historical_price.price_close()
                            EMA9, EMA21 = EMA_signal(closes)
                            MA55 = SMA(closes[-56:-1])

                    if short_position > 0 and first_reduction == True and second_reduction == False and candles[-1][2] >= break_even[-1]:
                      stoploss(sheet,start,row,break_even[-1],first_reduction,second_reduction, third_reduction, fourth_reduction)
                      sent_requests += 1
                      short_position = 0
                      while True:
                        if  EMA9 > MA55 and closes[-1] < EMA9:
                            print("Position re-entry confirmed.")
                            break 
                        elif EMA9 < MA55 and closes[-1] > EMA9:
                            print("Position re-entry confirmed.")
                            break
                        else:
                            time_stamp = timeStamp(client)
                            api_time = time_stamp.api_time_request() - 6000
                            historical_price = HistoricalPrice(host, symbol, interval, api_time, limit, client)
                            closes = historical_price.price_close()
                            EMA9, EMA21 = EMA_signal(closes)
                            MA55 = SMA(closes[-56:-1]) 

                    if short_position > 0 and second_reduction == True and third_reduction == False and candles[-1][2] >= take_profit_1: 
                      stoploss(sheet,start,row,take_profit_1,first_reduction,second_reduction, third_reduction, fourth_reduction)
                      sent_requests += 1
                      short_position = 0
                      while True:
                        if  EMA9 > MA55 and closes[-1] < EMA9:
                            print("Position re-entry confirmed.")
                            break 
                        elif EMA9 < MA55 and closes[-1] > EMA9:
                            print("Position re-entry confirmed.")
                            break
                        else:
                            time_stamp = timeStamp(client)
                            api_time = time_stamp.api_time_request() - 6000
                            historical_price = HistoricalPrice(host, symbol, interval, api_time, limit, client)
                            closes = historical_price.price_close()
                            EMA9, EMA21 = EMA_signal(closes)
                            MA55 = SMA(closes[-56:-1])  

                    if short_position > 0 and third_reduction == True and fourth_reduction == False and candles[-1][2] >= take_profit_2:
                      stoploss(sheet,start,row,take_profit_2,first_reduction,second_reduction, third_reduction, fourth_reduction)
                      sent_requests += 1
                      short_position = 0
                      while True:
                        if  EMA9 > MA55 and closes[-1] < EMA9:
                            print("Position re-entry confirmed.")
                            break 
                        elif EMA9 < MA55 and closes[-1] > EMA9:
                            print("Position re-entry confirmed.")
                            break
                        else:
                            time_stamp = timeStamp(client)
                            api_time = time_stamp.api_time_request() - 6000
                            historical_price = HistoricalPrice(host, symbol, interval, api_time, limit, client)
                            closes = historical_price.price_close()
                            EMA9, EMA21 = EMA_signal(closes)
                            MA55 = SMA(closes[-56:-1]) 

                    

      except Exception as e:
            print(e)
            time.sleep(20)
            client = bybit.bybit(test=False, api_key=api_key, api_secret=api_secret)   
            continue







   # closes = historical_price.price_close()
   # print("CLOSES: " + str(closes))
   # print(len(closes))
   # time.sleep(60)

   # zero_cross = MACD(closes, MACD_crosses)
   # print(MACD_crosses)
    
#    print(len(closes))
#    volume = historical_price.volume()
#    volume_sma = SMA(volume[-20:-1])
#    print(type(volume[-1]))

#    low = historical_price.price_low()
#    high = historical_price.price_high()
#    print("Low: " + str(low))
#    print("High: " + str(high))


#    simple_moving_average = SMA(closes[-20:-1])
#    upper_band = UB(simple_moving_average,closes[-21:-1])
#    lower_band = LB(simple_moving_average,closes[-21:-1])

 #   print("Upper Band: " + str(upper_band))
 #   print("Lower Band: " + str(lower_band))

 #   print("Latest Close: " + str(closes[-1]))

  #  positions = Position(host,param_str,symbol)
  #  position = positions.wrapper_position()
  #  live_price = LivePrice(host, param_str, symbol, interval, timestamp)
   # print("POSITION: " + str(position))
    

 #   wallet = Wallet(client,host,param_str,symbol)
 #   size = wallet.wrapper_wallet()
 #   print("Balance: " + str(size))# * live_price.price_dict()))

  #  live_price = LivePrice(host, param_str, symbol, interval, timestamp)
  #  price = live_price.price_dict()
  #  print(price)


  #Record entry, stop loss, take profit, and date/time in google sheets
  #














if __name__ == "__main__":

  #GOOGLE SHEETS 
  scope = ['https://www.googleapis.com/auth/drive','https://www.googleapis.com/auth/drive.file','https://www.googleapis.com/auth/spreadsheets'] #authorization For Drive


  credentials = ServiceAccountCredentials.from_json_keyfile_name("Bybit_Bitcoin_1_Minute_Data-517ea6748c2b.json",scope)     #JSON FILE WITH Credentials

  client = gspread.authorize(credentials)   

  sheet = client.open("Double EMA Single SMA").worksheet('ETH October - November Forward Test')#SHEET

#  data = sheet.get_all_records() 

  # TODO: Change code below to process the `response` dict:
# pprint(data)
  timestamp =  int(time.time()*1000) + 4000000 - 310000 #+5000
  timestamp = int(time.time()*1000) + 2500
  api_domain = {"live": "MSSbffFG1IWPkA1SZq", "test": "b1PEl6WF2IldIg4nGb"} #Ontario IP: "JOY4FE04n78T30XJ0r"}
  secret = {"live": "wT32i3s7DXz7bv0gPILdh0ESQpTxoTVpZ8za", "test": "ka1afUc2iAPe4FR0KP7KtqHKYfWLm8216lQm"} #"h7UObGL1FcTYtezWuNH9qolfY32uSAVaShlC"}
  url_domain = {"live": "https://api.bybit.com", "test": "https://api-testnet.bybit.com"}

  domain = "live"
  api_key = api_domain[domain]
  host = url_domain[domain]
  api_secret = secret[domain]
  client = bybit.bybit(test=False, api_key=api_key, api_secret=api_secret)   
  limit = '5'
  symbols = ["BTCUSD","ETHUSD","EOSUSD","XPRUSD","BTCUSDT"]
  symbol = "ETHUSD"
  leverage = "1"
  interval = "30"          #timeframe
  time_stamp = timeStamp(client)
  api_time = time_stamp.api_time_request() - 6000   #3 Min = + 137000       15 min = - 6000
  print("API TIME: " + str(api_time)) 
  print("TIMESTAMP: " + str(timestamp))
  client.Positions.Positions_saveLeverage(symbol=symbol, leverage="1").result()
  params = {}
  params['api_key'] = api_domain[domain]
  params['leverage'] = leverage
  params['symbol'] = symbol
  params['timestamp'] = timestamp
  signature = get_signature(api_secret,params)
  param_str = "api_key={}&leverage={}&symbol={}&timestamp={}&sign={}".format(api_key, leverage, symbol, timestamp, signature)  # Parameter required for HTTP requests
  trade(host, param_str, symbol, interval, timestamp, params, limit, client, api_time, api_key, signature, sheet, api_secret)




#Idea: If we can set 3 alarms that send an email when price closes above EMA9, EMA21, MA55
#If those emails contain those correct prices we can create a strategy with the TO-CHART EMAs 



#Necessary Alerts:
# Alert when EMA 9 is greater or less than EMA 21
# Alert when EMA 21 is greater or less than MA 55



#Todays Meeting:
#Any changes to the TP or SL placement
#Look at Chris' backtesting sheet and calculate the weighting of each take profit
#Turn Off BTC signals
#Go Through Cybersecurity sheet
#Ask for websites to model ours after





