######################################################################################
# realtimetest
# リアルタイムでのテストを行う。oandapyModuleを用いて行う。
# 過去5000足に関して情報を取得、リッジ回帰による学習を行い価格予測をして
# 自動売買を行う。
# デバッグをオフにすることで実際の売買を行える。
######################################################################################
import oandapyV20
from oandapyV20.endpoints.instruments import InstrumentsCandles
import pandas as pd
import numpy as np
from datetime import datetime
import json
import time
import oandapyModule
from sklearn.linear_model import Ridge
 
def realtimetestmain(chart_ins,chart_sec,term): 
	#ログ用フラグ
	flag = {
		"buy_signal":0,
		"sell_signal":0,
		"order":{
			"exist" : False,
			"side" : "",
			"price" : 0,
			"count" : 0
		},
		"position":{
			"exist" : False,
			"side" : "",
			"price": 0,
			"count": 0
		},
		"records":{
			"buy-count": 0,
			"buy-winning" : 0,
			"buy-return":[],
			"buy-profit": [],
			"buy-holding-periods":[],

			"sell-count": 0,
			"sell-winning" : 0,
			"sell-return":[],
			"sell-profit":[],
			"sell-holding-periods":[],

			"slippage":[],
			"log":[]
		}
	}

	#価格チャートを取得 price1に価格情報を代入
	#現在から5000足取得する
	price1 = oandapyModule.getPrice(chart_ins,chart_sec,5000)
	#リッジ回帰の定義、学習
	ridge = Ridge(alpha=0.01)
	oandapyModule.learnRidge2(ridge,price1)
	
	last_data = []
	i = 0
	X = oandapyModule.predictRidge(ridge,price1,0,5000)
	price = []
	#for i,x in enumerate(price1):
	#	price.append({ "close_time" : x["close_time"] ,
	#				"open_price" :     x["open_price"] ,
	#				"close_price" :    x["close_price"] ,
	#				"low_price" :      x["low_price"] ,
	#				"high_price" :     x["high_price"] ,
	#				"pred_price" :     X[i]})
	print("--------------------------")
	print("テスト期間：")
	print("開始時点 : " + str(price[0]["close_time"]))
	print("終了時点 : " + str(price[-1]["close_time"]))
	print(str(len(price)) + "件のローソク足データで検証")
	print("--------------------------")

	#i=0
	#price1 = oandapyModule.getPrice(chart_ins,"M1",5000)
	#ridge = Ridge(alpha=0.1)
	#oandapyModule.learnRidge2(ridge,price1)
	#price = []

	last_data = price
	#price1から価格情報に加えて価格予測値を代入して成形する
	for i,x in enumerate(price1):
		price.append({ "close_time" : x["close_time"] ,
					"open_price" :     x["open_price"] ,
					"close_price" :    x["close_price"] ,
					"low_price" :      x["low_price"] ,
					"high_price" :     x["high_price"] ,
					"pred_price" :     X[i]})
	print(i)
	#メインループ
	#過去5000足の情報に現在価格を付け加えていく
	#60秒ごとに1分足の情報を加える処理にと価格予測を行う
	while i < 10000:
		try:
			price1 = price1 + oandapyModule.getPriceNow(chart_ins, "M1")
			#価格予測を行う
			X = oandapyModule.predictRidge(ridge,price1,i-term,i)
			#price配列に最新価格と予想価格を入れる
			price.append({ "close_time" :  price1[-1]["close_time"] ,
						"open_price" :     price1[-1]["open_price"] ,
						"close_price" :    price1[-1]["close_price"] ,
						"low_price" :      price1[-1]["low_price"] ,
						"high_price" :     price1[-1]["high_price"] ,
						"pred_price" :     X[-1]})

			data = price[-1]
			flag = oandapyModule.logPrice(data,flag)

			#エントリークローズ確認をする
			#未約定の注文がないかチェック
			if flag["order"]["exist"]: 
				flag = oandapyModule.checkOrder( flag )
			#ポジションがある場合、クローズ条件を満たすか確認する
			elif flag["position"]["exist"]:
				flag = oandapyModule.closePosition( data,last_data,flag,chart_ins ) #ポジションがあれば決済条件を満たしていないかチェック
			#ポジションがない場合エントリー条件を満たすか確認する
			else:
				flag = oandapyModule.entrySignal( data,last_data,flag,chart_ins ) #ポジションが無い場合はエントリー条件を満たしているかチェック
			print(price[-1])
			print("ポジション{}".format(flag["position"]["side"]))
			print("価格{}".format(flag["position"]["price"]))
	 
			last_data.append( data )
			i += 1
			oandapyModule.printLog(flag)
			#5分に一回backtestの状況をprintする
			if i % 5 == 0:
				oandapyModule.backtest(flag)
			time.sleep(60)
		except:
			time.sleep(60)
 


#メイン処理
#パラメータを設定する。
chart_ins = "USD_HUF"
chart_sec = "M1"         # 1時間足
term = 20                #過去ｎ日の設定
try:
	realtimetestmain(chart_ins,chart_sec,term)
except:
	#例外発生時は5秒停止する
	print("error 時間足{0}、通貨ペア{1}".format(chart_ins,chart_sec))
	time.sleep(5)