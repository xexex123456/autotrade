######################################################################################
# oandapyModule
# oandapyV20 APIを使って価格予想、バックテスト、エントリー、クローズ
# などを総合的に行うライブラリ
# エントリークローズを同時に行っているのはbreakout関数で、このライブラリで
# もっとも重要である。
# flag変数は全ての買い、売りの情報といままでのログを記録する。
# flagは連想配列で作られている。
# 開発はVSCodeで行っているためpyCharmではWarningが出るかもしれません。
# TODO:リッジ回帰の学習精度が低いため、入力値を変更する必要がある
# TODO:機能ごとにファイル分割をする
######################################################################################
# backtest			: バックテストの結果をprintする
# breakout			: ブレイクを判定する。ドンチャン法
# breakout2			: ブレイクを判定する。価格予想を用いたドンチャン法
# breakout3			: ブレイクを判定する。ポリンジャーバンドを利用。
# checkOrder		: オーダーが通っているか判定する。
# closePosition	 	: ポジションを決済する
# entrySignal		: エントリー注文を出す
# getMovingAverage	: 移動平均を計算する
# getPrice			: 現在から過去5000足に関して価格情報を取得する
# getPriceFromFile	: jsonファイルから価格情報を取得する
# getPriceHist		: 指定した過去の地点から現在までの価格情報を取得する
# getPriceNow		: 指定した時間足の最新の価格情報を取得する
# getStandardValue	: 標準偏差を取得する
# learnRidge		: リッジ回帰で学習を行う。過去5足の終値を用いる
# learnRidge2		: リッジ回帰で学習を行う。過去5足の始値と終値を用いる
# logPrice			: 時間と始値と終値をログ情報として記載する
# ordersDecide		: 注文を決済する。
# predictRidge		: リッジ回帰で学習した内容から価格の予想を行う
# printLog			: ログ情報の一番最後をプリントする
# printPrice		: 時間と始値と終値をprintする
# records			: 各トレードのパフォーマンスを記録する
# setTestNum		: テスト用のパラメータをセットする。
# tradeCloseDecide	: ポジションをクローズする
######################################################################################
import oandapyV20
from oandapyV20.endpoints.instruments import InstrumentsCandles
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
import pandas as pd
import numpy as np
from datetime import datetime
import json
import time
from datetime import datetime, timedelta
 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
 
#APIkeyを読み込む
path = "./APIKEY.ini"
f = open(path)
l_strip = [s.strip() for s in f.readlines()]
accountID = l_strip[0]
access_token = l_strip[1]
 
 
api = oandapyV20.API(access_token = access_token, environment = "practice")
lot=10000
slippage=0.003
term=30
debug=1
break_num=0.02

 
#テスト用のパラメータをセットする。
def setTestNum(_x1=10000,_x2=0.003,_x3=30,_x4=1,_x5=0.02):
	global lot
	global slippage
	global term
	global debug
	global break_num

	lot = _x1
	slippage = _x2
	term = _x3
	debug = _x4
	break_num = _x5

	print("ロット数：{0}、スリッページ：{1}、移動平均期間：{2}、デバッグ：{3}、breakout変数：{4}".format(lot,slippage,term,debug,break_num))
	return
#指定した過去の地点から現在までの価格情報を取得する 
#base_startから現在の期間における値段リストを作成する
def getPriceHist(chart_ins,chart_sec,base_start):
	price = []

	#日付を1日毎ずらしてリストを作成
	#base_start = "2020-01-01T00:00:00.000000Z"
	#base_end = base_start + timedelta(days=1)
	base_now = datetime.now()
	base_now = base_now + timedelta(days=-0) 
	base_start = datetime.strptime(base_start, '%Y-%m-%dT%H:%M:%S.%fZ')
	base_end = base_start + timedelta(days=1)
	#base_end = datetime.strptime(base_end, '%Y-%m-%dT%H:%M:%S.%fZ')
	td = base_now - base_end
	#メインループ。指定した日付から終わりの日付までループする。
	for i in range(td.days):
		start = base_start + timedelta(days=+i)
		end = base_end + timedelta(days=+i)
		weekday = start.weekday()
		start = start.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
		end = end.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
		#週末は価格情報がないためスキップする。
		if weekday > 5:
			continue
		r = InstrumentsCandles(instrument=chart_ins,
							params={
								"granularity": chart_sec, 
								"from"       : start,
								"to"         : end
								})
		#価格情報を取得する
		api.request(r)
		data = r.response["candles"]
		print(start)
		#取得した情報を整形する
		for j in range(len(data)):
			price.append({ "close_time" : pd.to_datetime(data[j]["time"]),
						"open_price" : float(data[j]["mid"]["o"]),
						"close_price" : float(data[j]["mid"]["c"]),
						"low_price" : float(data[j]["mid"]["l"]),
						"high_price" : float(data[j]["mid"]["h"])})
					
	
	return price

#現在から過去5000足に関して価格情報を取得する
def getPrice(chart_ins,chart_sec,chart_count):
	price = []
	r = InstrumentsCandles(instrument=chart_ins,
							params={
								"granularity":chart_sec, 
								"count" : chart_count 
								})
	#価格情報を取得する
	api.request(r)
	data = r.response["candles"]
	
	#取得した情報を整形する
	for j in range(len(data)):
		price.append({ "close_time" : pd.to_datetime(data[j]["time"]),
					"open_price" : float(data[j]["mid"]["o"]),
					"close_price" : float(data[j]["mid"]["c"]),
					"low_price" : float(data[j]["mid"]["l"]),
					"high_price" : float(data[j]["mid"]["h"])})
	return price

#指定した時間足の最新の価格情報を取得する
def getPriceNow(chart_ins,chart_sec):
	price = []
	r = InstrumentsCandles(instrument=chart_ins,
							params={
								"granularity":chart_sec, 
								"count" : 1
								})
	#価格情報を取得する
	api.request(r)
	data = r.response["candles"]
	
	#取得した情報を整形する
	price.append({ "close_time" : pd.to_datetime(data[0]["time"]),
				"open_price" : float(data[0]["mid"]["o"]),
				"close_price" : float(data[0]["mid"]["c"]),
				"low_price" : float(data[0]["mid"]["l"]),
				"high_price" : float(data[0]["mid"]["h"])})
					
	
	return price
 
 
#json形式のファイルから価格データを読み込む関数
def getPriceFromFile(path):
	file = open(path,'r',encoding='utf-8')
	price = json.load(file)
	return price		
 
 
#時間と始値と終値をprintする
def printPrice( data ):
	print( "時間: " + str(data["close_time"])
				+ " 始値: " + str(data["open_price"])
				+ " 終値: " + str(data["close_price"]) )

def printLog(flag):
	print(flag["records"]["log"][-1])
				
 
#時間と始値と終値をログ情報として記載する
def logPrice( data,flag ):
	log =  "時間： " + str(data["close_time"]) + " 始値： " + str(data["open_price"]) + " 終値： " + str(data["close_price"]) + "予想： "+ str(data["pred_price"]) + "\n"
	flag["records"]["log"].append(log)
	return flag
 
 
######################################################################################
#ブレイクを判定する。ドンチャン法
#過去指定した期間における最新の高値、安値を取得して現在の価格と比較し
#高値を上まわっていれば「買い」
#安値を下回っていれば「売り」注文をだす。
######################################################################################
def breakout( data,last_data,flag):
	highest = max(i["high_price"] for i in last_data[-1 * term:-2])
	lowest = min(i["low_price"] for i in last_data[-1 * term:-2])
	if flag["position"]["exist"] == True:
		if data["low_price"] < lowest:
			return {"side":"SELL","price":lowest}
	else:
		if data["pred_price"] > highest:
			return {"side":"BUY","price":highest}
	
	if flag["position"]["exist"] == True:
		if data["high_price"] > highest:
			return {"side":"BUY","price":highest}
	else:
		if data["pred_price"] < lowest:
			return {"side":"SELL","price":lowest}
	
	return {"side" : None , "price":0}
 
######################################################################################
#ブレイクを判定する。価格予想を用いたドンチャン法
#基本的にはbreakout関数と同様だが、学習による予想価格を用いている
######################################################################################
def breakout2( data,last_data,flag):
	highest = max(i["high_price"] for i in last_data[-1 * term:-2])
	lowest = min(i["low_price"] for i in last_data[-1 * term:-2])
	if break_num * data["pred_price"] + (1 - break_num) * data["high_price"] > highest:
		return {"side":"BUY","price":highest}
	
	if break_num * data["pred_price"] + (1 - break_num) * data["low_price"] < lowest:
		return {"side":"SELL","price":lowest}
	
	return {"side" : None , "price":0}

######################################################################################
#ブレイクを判定する。ポリンジャーバンドを利用。
#エントリー
#エントリーは値動きが激しくない(ポリンジャーバンドの幅が0.6以下)の時
#ポリンジャーバンドの外に高値、安値があるときにエントリーする。
#ポリンジャーバンドは移動平均±標準偏差*2で計算している。
#
#クローズ
#クローズは価格がエントリーから一定の動きを見せたとき行う。
#エントリー価格の0.0015倍、上下したときに決済する。
######################################################################################
def breakout3( data,last_data,flag):
	sma23 = getMovingAverage([s.get("close_price") for s in last_data[-100:]], 23)
	sigma = getStandardValue([s.get("close_price") for s in last_data[-100:]], 23)
	upper2_sigma = sma23 + sigma * 2
	lower2_sigma = sma23 - sigma * 2

	high = upper2_sigma.iloc[-1]
	low = lower2_sigma.iloc[-1]

	if flag["position"]["exist"] == True:
		if flag["position"]["side"] == "BUY":
			if(abs(flag["position"]["price"]/lot - data["close_price"]) > 0.0015*flag["position"]["price"]/lot):
				return {"side":"SELL","price":high}
		if flag["position"]["side"] == "SELL":
			if(abs(flag["position"]["price"]/lot - data["close_price"]) > 0.0015*flag["position"]["price"]/lot):
				return {"side":"BUY","price":low}
	else:
		if high - low > 0.6:
			return {"side" : None , "price":0}
		if break_num * data["pred_price"] + (1 - break_num) * data["high_price"] > high:
			return {"side":"SELL","price":high}
	
		if break_num * data["pred_price"] + (1 - break_num) * data["low_price"] < low:
			return {"side":"BUY","price":low}
	
	return {"side" : None , "price":0}
 
#ブレイクを判定してエントリー注文を出す関数
def entrySignal( data,last_data,flag,chart_ins ):
	#breakout判定。ここを変更することで、別判定方法を利用できる。
	signal = breakout3( data,last_data,flag )
	if signal["side"] == "BUY":
		#ログに売買情報を記載する
		flag["records"]["log"].append("過去{0}足の最高値{1}円を、直近の高値が{2}円でブレイクしました\n".format(term,signal["price"],data["high_price"]))
		flag["records"]["log"].append(str(data["close_price"]) + "円で買いの指値注文を出します\n")
 
		#デバッグがオフの場合注文を入れる
		if(debug == 0):
			ordersDecide(lot,chart_ins)
		
		#フラグ情報を更新する
		flag["order"]["exist"] = True
		flag["order"]["side"] = "BUY"
		flag["order"]["price"] = data["close_price"] * lot
 
	if signal["side"] == "SELL":
		#ログに売買情報を記載する
		flag["records"]["log"].append("過去{0}足の最安値{1}円を、直近の安値が{2}円でブレイクしました\n".format(term,signal["price"],data["low_price"]))
		flag["records"]["log"].append(str(data["close_price"]) + "円で売りの指値注文を出します\n")
 
		#デバッグがオフの場合注文を入れる
		if(debug == 0):
			ordersDecide(-lot,chart_ins)
		
		#フラグ情報を更新する
		flag["order"]["exist"] = True
		flag["order"]["side"] = "SELL"
		flag["order"]["price"] = data["close_price"] * lot
 
	return flag
		
#ポジションを決済する
def closePosition( data,last_data,flag ,chart_ins):
	
	flag["position"]["count"] += 1
	#breakout判定。ここを変更することで、別判定方法を利用できる。
	signal = breakout3( data,last_data,flag )
	if flag["position"]["side"] == "BUY":
		#ポジションが「買い」でbreakout判定が「売り」の場合
		if signal["side"] == "SELL":
			#ログに売買情報を記載する
			flag["records"]["log"].append("過去{0}足の最安値{1}円を、直近の安値が{2}円でブレイクしました\n".format(term,signal["price"],data["low_price"]))
			flag["records"]["log"].append(str(data["close_price"]) + "円あたりで成行注文を出してポジションを決済します\n")
			
			#デバッグがオフの場合、決済の注文を入れる
			if(debug == 0):
				tradeCloseDecide(chart_ins)
			
			#フラグ情報を更新する
			records( flag,data )
			flag["position"]["exist"] = False
			flag["position"]["count"] = 0
			flag["position"]["side"] = ""
			flag["position"]["price"] = 0
			
	if flag["position"]["side"] == "SELL":
		#ポジションが「売り」でbreakout判定が「買い」の場合
		if signal["side"] == "BUY":
			#ログに売買情報を記載する
			flag["records"]["log"].append("過去{0}足の最高値{1}円を、直近の高値が{2}円でブレイクしました\n".format(term,signal["price"],data["high_price"]))
			flag["records"]["log"].append(str(data["close_price"]) + "円あたりで成行注文を出してポジションを決済します\n")
			
			#デバッグがオフの場合、決済の注文を入れる
			if(debug == 0):
				tradeCloseDecide(chart_ins)
			
			#フラグ情報を更新する
			records( flag,data )
			flag["position"]["exist"] = False
			flag["position"]["count"] = 0
			flag["position"]["side"] = ""
			flag["position"]["price"] = 0
			
	return flag
	
 
#サーバーに出した注文が約定したかどうかチェックする関数
def checkOrder( flag ,chart_ins):
	

	p = positions.PositionDetails(accountID=accountID, instrument=chart_ins)
	rv = api.request( p )

	#print(rv)
	#注文状況を確認して通っていたら以下を実行
	if rv["position"]["long"]["units"] != "0":
		flag["order"]["exist"] = False
		flag["order"]["count"] = 0
		flag["position"]["exist"] = True
		flag["position"]["side"] = "BUY"
		flag["position"]["price"] = float(rv["position"]["long"]["averagePrice"]) * float(rv["position"]["long"]["units"])
		print(rv["position"]["long"]["averagePrice"])
	elif rv["position"]["short"]["units"] != "0":
		flag["order"]["exist"] = False
		flag["order"]["count"] = 0
		flag["position"]["exist"] = True
		flag["position"]["side"] = "SELL"
		flag["position"]["price"] = -1 * float(rv["position"]["short"]["averagePrice"]) * float(rv["position"]["short"]["units"])
		print(rv["position"]["short"]["averagePrice"])
	#注文が通っていなければキャンセルする
	else:
		flag["order"]["exist"] = False
		flag["order"]["count"] = 0
		flag["position"]["exist"] = False
		flag["position"]["count"] = 0
		flag["position"]["side"] = ""
		flag["position"]["price"] = 0

	
	
	return flag
	
 
#各トレードのパフォーマンスを記録する関数
def records(flag,data):
	
	#取引手数料等の計算
	entry_price = flag["position"]["price"]
	exit_price = data["close_price"] * lot
	trade_cost = round( lot * slippage )
	
	log = "スリッページ・手数料として " + str(trade_cost) + "円を考慮します\n"
	flag["records"]["log"].append(log)
	flag["records"]["slippage"].append(trade_cost)
	
	#値幅の計算
	buy_profit = exit_price - entry_price - trade_cost
	sell_profit = entry_price - exit_price - trade_cost
	
	#利益が出てるかの計算
	if flag["position"]["side"] == "BUY":
		flag["records"]["buy-count"] += 1
		flag["records"]["buy-profit"].append( buy_profit )
		flag["records"]["buy-return"].append( round( buy_profit / entry_price * 100, 4 ))
		flag["records"]["buy-holding-periods"].append( flag["position"]["count"] )
		if buy_profit  > 0:
			flag["records"]["buy-winning"] += 1
			log = str(buy_profit) + "円の利益です\n"
			flag["records"]["log"].append(log)
		else:
			log = str(buy_profit) + "円の損失です\n"
			flag["records"]["log"].append(log)
	
	if flag["position"]["side"] == "SELL":
		flag["records"]["sell-count"] += 1
		flag["records"]["sell-profit"].append( sell_profit )
		flag["records"]["sell-return"].append( round( sell_profit / entry_price * 100, 4 ))
		flag["records"]["sell-holding-periods"].append( flag["position"]["count"] )
		if sell_profit > 0:
			flag["records"]["sell-winning"] += 1
			log = str(sell_profit) + "円の利益です\n"
			flag["records"]["log"].append(log)
		else:
			log = str(sell_profit) + "円の損失です\n"
			flag["records"]["log"].append(log)
	
	return flag
	
 
#バックテストの結果をprintする
def backtest(flag):
	print("バックテストの結果")
	print("--------------------------")
	print("買いエントリの成績")
	print("--------------------------")
	print("トレード回数  :  {}回".format(flag["records"]["buy-count"] ))
	if(flag["records"]["buy-count"] == 0):
		print("勝率          :  0％")
		print("平均リターン  :   0％")
		print("総損益        :   0円")
		print("平均保有期間  :   0足分")
	else:
		print("勝率          :  {}％".format(round(flag["records"]["buy-winning"] / flag["records"]["buy-count"] * 100,1)))
		print("平均リターン  :  {}％".format(round(np.average(flag["records"]["buy-return"]),4)))
		print("総損益        :  {}円".format( np.sum(flag["records"]["buy-profit"]) ))
		print("平均保有期間  :  {}足分".format( round(np.average(flag["records"]["buy-holding-periods"]),1) ))
	
	print("--------------------------")
	print("売りエントリの成績")
	print("--------------------------")
	print("トレード回数  :  {}回".format(flag["records"]["sell-count"] ))
	if(flag["records"]["sell-count"] == 0):
		print("勝率          :  0％")
		print("平均リターン  :   0％")
		print("総損益        :   0円")
		print("平均保有期間  :   0足分")
	else:
		print("平均リターン  :  {}％".format(round(np.average(flag["records"]["sell-return"]),4)))
		print("勝率          :  {}％".format(round(flag["records"]["sell-winning"] / flag["records"]["sell-count"] * 100,1)))
		print("総損益        :  {}円".format( np.sum(flag["records"]["sell-profit"]) ))
		print("平均保有期間  :  {}足分".format( round(np.average(flag["records"]["sell-holding-periods"]),1) ))
	
	print("--------------------------")
	print("総合の成績")
	print("--------------------------")
	print("総損益        :  {}円".format( np.sum(flag["records"]["sell-profit"]) + np.sum(flag["records"]["buy-profit"]) ))
	print("手数料合計    :  {}円".format( np.sum(flag["records"]["slippage"]) ))
	
	# ログファイルの出力
	file =  open("./log/{0}-log.txt".format(datetime.now().strftime("%Y-%m-%d-%H-%M")),'wt',encoding='utf-8')
	file.writelines(flag["records"]["log"])
 
#リッジ回帰で学習を行う。過去5足の終値を用いる
def learnRidge(ridge,price):
	end_count = len(price)
	X = [[0 for j in range(5)] for i in range(end_count)]
	X_next = [0 for i in range(end_count)]
	for i,x in enumerate(price):
		#print(x["close_price"],i)
		if i >= 5:
			X[i-5][0] = x["close_price"]
		if i >= 4 & i < end_count:
			X[i-4][1] = x["close_price"]
		if i >= 3 & i < end_count:
			X[i-3][2] = x["close_price"]
		if i >= 2 & i < end_count:
			X[i-2][3] = x["close_price"]
		if i >= 1 & i < end_count:
			X[i-1][4] = x["close_price"]

		X_next[i] = x["close_price"]
	ridge.fit(X,X_next)

#リッジ回帰で学習を行う。過去5足の始値と終値を用いる
def learnRidge2(ridge,price):
	end_count = len(price)
	X = [[0 for j in range(10)] for i in range(end_count)]
	X_next = [0 for i in range(end_count)]
	for i,x in enumerate(price):
		#print(x["close_price"],i)
		if i >= 5:
			X[i-5][0] = x["open_price"]
			X[i-5][5] = x["close_price"]
		if i >= 4 & i < end_count:
			X[i-4][1] = x["open_price"]
			X[i-4][6] = x["close_price"]
		if i >= 3 & i < end_count:
			X[i-3][2] = x["open_price"]
			X[i-3][7] = x["close_price"]
		if i >= 2 & i < end_count:
			X[i-2][3] = x["open_price"]
			X[i-2][8] = x["close_price"]
		if i >= 1 & i < end_count:
			X[i-1][4] = x["open_price"]
			X[i-1][9] = x["close_price"]

		X_next[i] = x["close_price"]
	ridge.fit(X,X_next)

#リッジ回帰で学習した内容から価格の予想を行う
def predictRidge(ridge,price,start_num,end_num):
	end_count = len(price)
	X = [[0 for j in range(10)] for i in range(end_num-start_num)]
	for i,x in enumerate(price[end_count-(end_num-start_num)::]):
		if i >= 4:
			X[i-4][0] = x["open_price"]
			X[i-4][5] = x["close_price"]
		if i >= 3 & i < end_num-start_num:
			X[i-3][1] = x["open_price"]
			X[i-3][6] = x["close_price"]
		if i >= 2 & i < end_num-start_num:
			X[i-2][2] = x["open_price"]
			X[i-2][7] = x["close_price"]
		if i >= 1 & i < end_num-start_num:
			X[i-1][3] = x["open_price"]
			X[i-1][8] = x["close_price"]
		X[i][4] = x["open_price"]
		X[i][9] = x["close_price"]
	y_pred = ridge.predict(X)
	return y_pred

#注文を決済する
def ordersDecide(unit,chart_ins):
	data = {
	 "order": {
	   "instrument": chart_ins,
	   "units": unit,
	   "type": "MARKET",
	 }
	} #注文実行
	r = orders.OrderCreate(accountID, data=data)
	rv = api.request(r)
	print(rv)

#ポジションをクローズする
def tradeCloseDecide(chart_ins):
	p = positions.PositionDetails(accountID=accountID, instrument=chart_ins)
	rv = api.request( p )
	r = trades.TradeClose( accountID ,tradeID=rv["lastTransactionID"] )
	rv = api.request( r )
	print(rv)

#移動平均を計算する
def getMovingAverage(s,window):
	s = pd.Series([x for x in s])
	return s.rolling(window).mean()

#標準偏差を取得する
def getStandardValue(s,window):
	s = pd.Series([x for x in s])
	return s.rolling(window).std(ddof=0)
