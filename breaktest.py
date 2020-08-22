######################################################################################
# breaktest
# 総合テストを行う。oandapyModuleを用いて行う。
# oandaで扱う全ての通貨ペアとDay以下の時間足全てに関してテストする。
# テスト期間は過去5000足、もしくは指定期間から現在までで行う。
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
 
 
def breaktestmain(chart_ins,chart_sec,term):
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
	#実験用 現在から5000足取得する
	#price1 = oandapyModule.getPrice(chart_ins,chart_sec,5000)
	#検証用 期間を指定して価格を取得する。
	price1 = oandapyModule.getPriceHist(chart_ins,chart_sec,"2020-01-01T00:00:00.000000Z")
	#リッジ回帰の定義、学習
	ridge = Ridge(alpha=1)
	oandapyModule.learnRidge2(ridge,price1)
	X = oandapyModule.predictRidge(ridge,price1,0,len(price1))

	#last_dataには最新の価格情報が入る。
	last_data = []
	i = 0
	price = []
	#price1から価格情報に加えて価格予測値を代入して成形する
	for i,x in enumerate(price1):
		price.append({ "close_time" : x["close_time"] ,
					"open_price" :     x["open_price"] ,
					"close_price" :    x["close_price"] ,
					"low_price" :      x["low_price"] ,
					"high_price" :     x["high_price"] ,
					"pred_price" :     X[i]})
	print(price[0])
	print("--------------------------")
	print("テスト期間：")
	print("開始時点 : " + str(price[0]["close_time"]))
	print("終了時点 : " + str(price[-1]["close_time"]))
	print(str(len(price)) + "件のローソク足データで検証")
	print("--------------------------")
	i=0
	#このループでは全期間において実際の価格の動きをシミュレーションして
	#エントリー、クローズを行う。
	#その結果得られる損益のシミュレーションを行う。
	while i < len(price):
	
		#ブレイクアウト判定に使う最低限ｎ期間分のローソク足をセット
		#ブレイクアウトには移動平均値を取るための23期間は最低でも必要
		if len(last_data) < term:
			last_data.append(price[i])
			#ログ出力
			flag = oandapyModule.logPrice(price[i],flag)
			i += 1
			continue

		data = price[i]
		#ログ出力
		flag = oandapyModule.logPrice(data,flag)

		#未約定の注文がないかチェック
		if flag["order"]["exist"]: 
			flag = oandapyModule.checkOrder( flag )
		#ポジションがあれば決済条件を満たしていないかチェック
		elif flag["position"]["exist"]:
			flag = oandapyModule.closePosition( data,last_data,flag ,chart_ins) 
		#ポジションが無い場合はエントリー条件を満たしているかチェック
		else:
			flag = oandapyModule.entrySignal( data,last_data,flag, chart_ins ) 

		#最新情報を入力
		last_data.append( data )
		i += 1
	
	#ループ完了後、テスト結果をテキストで出力
	oandapyModule.backtest(flag)
	return flag

#メイン処理
#パラメータを設定する。
#購入ロット数、スリッページ、移動平均期間、デバッグパラメータ、ブレイクアウト変数
oandapyModule.setTestNum(1)
#通貨ペア一覧
chart_ins = ["USD_JPY","EUR_JPY","AUD_JPY","GBP_JPY","NZD_JPY","CAD_JPY","CHF_JPY","ZAR_JPY","EUR_USD","GBP_USD","NZD_USD","AUD_USD","USD_CHF","EUR_CHF","GBP_CHF","EUR_GBP","AUD_NZD","AUD_CAD","AUD_CHF","CAD_CHF","EUR_AUD","EUR_CAD","EUR_DKK","EUR_NOK","EUR_NZD","EUR_SEK","GBP_AUD","GBP_CAD","GBP_NZD","NZD_CAD","NZD_CHF","USD_CAD","USD_DKK","USD_NOK","USD_SEK","AUD_HKD","AUD_SGD","CAD_HKD","CAD_SGD","CHF_HKD","CHF_ZAR","EUR_CZK","EUR_HKD","EUR_HUF","EUR_PLN","EUR_SGD","EUR_TRY","EUR_ZAR","GBP_HKD","GBP_PLN","GBP_SGD","GBP_ZAR","HKD_JPY","NZD_HKD","NZD_SGD","SGD_CHF","SGD_HKD","SGD_JPY","TRY_JPY","USD_CNH","USD_CZK","USD_HKD","USD_HUF","USD_INR","USD_MXN","USD_PLN","USD_SAR","USD_SGD","USD_THB","USD_TRY","USD_ZAR"]
#時間足一覧
chart_sec = ["M1","M5","M15","M30","H1","H2","H4"]
#chart_ins = ["USD_JPY"]
#chart_sec = ["M1"]

#ログ保存用ファイルオープン
f =  open("./breaktest_result_{0}.txt".format(datetime.now().strftime("%Y-%m-%d-%H-%M")),'wt',encoding='utf-8')
print("通貨ペア,時間足,買い利益,売り利益,買い勝率,売り勝率",file=f)
#通貨ペアループ
for i in range(len(chart_ins)):
	#時間足ループ
	for j in range(len(chart_sec)):
		try:
			flag = breaktestmain(chart_ins[i],chart_sec[j],30)
		except:
			print("error 時間足{0}、通貨ペア{1}".format(chart_ins[i],chart_sec[j]))
		try:
			print("{0},{1},{2},{3},{4},{5}".format(chart_ins[i],chart_sec[j],np.sum(flag["records"]["buy-profit"]),np.sum(flag["records"]["sell-profit"]),round(flag["records"]["buy-winning"] / flag["records"]["buy-count"] * 100,1),round(flag["records"]["sell-winning"] / flag["records"]["sell-count"] * 100,1)), file=f)
		except:
			print("{0},{1},0,0,0,0".format(chart_ins[i],chart_sec[j]),file=f)
