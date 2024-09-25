import os
import csv
import pandas as pd
import numpy as np
import json
import requests
import akshare as ak
from Ashare_news import get_all_prompts_new
from glob import glob
from tqdm import tqdm
import datetime
import time
from datetime import date, datetime, timedelta
from openai import OpenAI


client = OpenAI(
    # 下面两个参数的默认值来自环境变量，可以不加
    api_key="sk-kk1BywoOENWbUoWcF3AeE226177141E8A28b098f397f8674",
    base_url="https://xiaoai.plus/v1",
)

SYSTEM_PROMPT = "你是一名经验丰富的股票市场分析师。你的任务是根据公司在过去几周内的相关新闻和季度财务状况，列出公司的积极发展和潜在担忧，然后结合你对整体金融经济市场的判断，对公司未来一周的股价变化提供预测和分析。" \
    "你的回答语言应为中文。你的回答格式应该如下：\n\n[积极发展]：\n1. ...\n\n[潜在担忧]：\n1. ...\n\n[预测和分析]：\n...\n"

def append_to_csv(filename, input_data, output_data):
    with open(filename, mode='a', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([input_data, output_data])


def initialize_csv(filename):
    with open(filename, "w", newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["prompt", "answer"])

def query_gpt4(symbol_list, min_past_weeks=1, max_past_weeks=2, with_basics=True):
    for symbol in symbol_list:

        csv_file = f'../data/csv/{symbol}_nobasics_gpt-4.csv'

        if not os.path.exists(csv_file):
            initialize_csv(csv_file)
            pre_done = 0
        else:
            df = pd.read_csv(csv_file, encoding="utf-8")
            pre_done = len(df)

        prompts = get_all_prompts_new(symbol, min_past_weeks, max_past_weeks, with_basics)

        for i, prompt in enumerate(prompts):

            if i < pre_done:
                continue

            print(f"{symbol} - {i}")

            cnt = 0
            while cnt < 5:
                try:
                    completion = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    print("==Generate answer successfully==")
                    break
                except Exception:
                    cnt += 1
                    print(f'retry cnt {cnt}')

            answer = completion.choices[0].message.content if cnt < 5 else ""
            append_to_csv(csv_file, prompt, answer)

def stock_news_em(symbol: str = "300059", page=1) -> pd.DataFrame:
    """
    按照股票代码获取新闻信息
    """
    url = "https://search-api-web.eastmoney.com/search/jsonp"
    params = {
        "cb": "jQuery3510875346244069884_1668256937995",
        "param": '{"uid":"",'
                 + f'"keyword":"{symbol}"'
                 + ',"type":["cmsArticleWebOld"],"client":"web","clientType":"web","clientVersion":"curr","param":{"cmsArticleWebOld":{"searchScope":"default","sort":"default",' + f'"pageIndex":{page}' + ',"pageSize":100,"preTag":"<em>","postTag":"</em>"}}}',
        "_": "1668256937996",
    }
    get = 200
    while get:
        try:
            r = requests.get(url, params=params)
            data_text = r.text
            get = 0
        except:
            time.sleep(1)

    data_text = r.text
    data_json = json.loads(
        data_text.strip("jQuery3510875346244069884_1668256937995(")[:-1]
    )
    temp_df = pd.DataFrame(data_json["result"]["cmsArticleWebOld"])
    temp_df.rename(
        columns={
            "date": "发布时间",
            "mediaName": "文章来源",
            "code": "-",
            "title": "新闻标题",
            "content": "新闻内容",
            "url": "新闻链接",
            "image": "-",
        },
        inplace=True,
    )
    temp_df["关键词"] = symbol
    temp_df = temp_df[
        [
            "关键词",
            "新闻标题",
            "新闻内容",
            "发布时间",
            "文章来源",
            "新闻链接",
        ]
    ]
    temp_df["新闻标题"] = (
        temp_df["新闻标题"]
        .str.replace(r"\(<em>", "", regex=True)
        .str.replace(r"</em>\)", "", regex=True)
    )
    temp_df["新闻标题"] = (
        temp_df["新闻标题"]
        .str.replace(r"<em>", "", regex=True)
        .str.replace(r"</em>", "", regex=True)
    )
    temp_df["新闻内容"] = (
        temp_df["新闻内容"]
        .str.replace(r"\(<em>", "", regex=True)
        .str.replace(r"</em>\)", "", regex=True)
    )
    temp_df["新闻内容"] = (
        temp_df["新闻内容"]
        .str.replace(r"<em>", "", regex=True)
        .str.replace(r"</em>", "", regex=True)
    )
    temp_df["新闻内容"] = temp_df["新闻内容"].str.replace(r"\u3000", "", regex=True)
    temp_df["新闻内容"] = temp_df["新闻内容"].str.replace(r"\r\n", " ", regex=True)
    return temp_df

def get_news(symbol, max_page=100):
    ##获取新闻信息集合

    df_list = []
    page = 0
    while page < max_page:
        try:
            df_list.append(stock_news_em(symbol, page))
            page += 1
        except KeyError:
            print(str(symbol) + "pages obtained for symbol: " + str(page))
            break

    news_df = pd.concat(df_list, ignore_index=True)
    return news_df

def get_company_prompt_new(symbol):
    #获取公司介绍

    index = 1
    while index:
        try:
            company_profile = dict(ak.stock_individual_info_em(symbol).values)
            index = 0
        except:
            print("Company Info Request Time Out! Please wait and retry.")
    company_profile["上市时间"] = pd.to_datetime(str(company_profile["上市时间"])).strftime("%Y年%m月%d日")

    template = "[公司介绍]:\n\n{股票简称}是一家在{行业}行业的领先实体，自{上市时间}成立并公开交易。截止今天，{股票简称}的总市值为{总市值}人民币，总股本数为{总股本}，流通市值为{流通市值}人民币，流通股数为{流通股}。" \
               "\n\n{股票简称}主要在中国运营，以股票代码{股票代码}在交易所进行交易。"

    formatted_profile = template.format(**company_profile)
    stockname = company_profile['股票简称']
    return formatted_profile, stockname

def get_stock(ts_code, start_date):
    end_date = n_weeks_after(start_date, 1)
    start_date = start_date.split(" ")[0].replace("-", "")
    end_date = end_date.split(" ")[0].replace("-", "")
    get = 200
    stock_hist_df = []
    while get or len(stock_hist_df)==0:
        try:
            stock_hist_df = ak.stock_zh_a_hist(symbol=ts_code, start_date=start_date, end_date=end_date)
            get = 0
        except:
            time.sleep(3)
            continue

    return stock_hist_df

def get_stock_all(ts_code, start_date):
    stock_hist_df = []
    start_date = n_weeks_before(start_date, 1)
    start_date = start_date.split(" ")[0].replace("-", "")
    get = 1
    while get or len(stock_hist_df) == 0:
        try:
            stock_hist_df = ak.stock_zh_a_hist(symbol=ts_code, start_date=start_date)
            get = 0
        except:
            time.sleep(3)
            continue

    return stock_hist_df

def transform_rate_data_online(df):
    rates = []
    rate_des = []
    price = df["收盘"]
    price = [float(p) for p in price]
    # maxp, minp = max(price), min(price)
    rate = (price[-1]-price[0])/price[0]

    if rate > 0:
        pre = "涨幅"
        if rate < 0.01:
            pre += "小于1%"
        elif rate >= 0.01 and rate < 0.03:
            pre += "在1%-3%之间"
        elif rate >= 0.03 and rate < 0.05:
            pre += "在3%-5%之间"
        else:
            pre += "大于5%"

    elif rate < 0:
        pre = "下跌"
        if rate < 0.01:
            pre += "小于1%"
        elif rate >= 0.01 and rate < 0.03:
            pre += "在1%-3%之间"
        elif rate >= 0.03 and rate < 0.05:
            pre += "在3%-5%之间"
        else:
            pre += "大于5%"

    else:
        pre = "股价可能持平"
    rates.append(rate)
    rate_des.append(pre)

    return rates, rate_des

def respones_gpt(system_prompt, user_input):
    index = 1
    time.sleep(1)
    request_time = 0
    while index:
        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ]
            )
            index = 0
        except:
            time.sleep(1)
        request_time += 1

        if request_time > 10:
            print("request time is {}".format(request_time))
            time.sleep(30)


    return completion.choices[0].message.content

def get_curday():
    return date.today().strftime("%Y-%m-%d %H:%M:%S")

def n_weeks_before(date_string, n, format='%Y-%m-%d %H:%M:%S'):
    date = datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S') - timedelta(days=7 * n)

    return date.strftime(format=format)

def n_weeks_after(date_string, n, format='%Y-%m-%d %H:%M:%S'):
    date = datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S') + timedelta(days=7 * n)

    return date.strftime(format=format)

def stock_concat2h5(folder):
    df_list = []
    csvs = glob(folder)
    for file in tqdm(csvs):
        df = pd.read_csv(file, encoding="utf-8", delimiter="\t")
        df_list.append(df)

    stock_daliy = pd.concat(df_list)
    stock_daliy.to_hdf("./data/raw/stock_daily.h5", key="data")

def transform_rate_data(df):
    rates = []
    rate_des = []
    for i, d in df.iterrows():
        price = [d["open1"], d["open2"], d["open3"], d["open4"], d["open5"]]
        price = [float(p) for p in price]
        maxp, minp = max(price), min(price)
        rate = (maxp-minp)/float(d["original_price"])

        if rate > 0:
            pre = "上涨幅度"
            if rate < 0.01:
                pre += "小于1%"
            elif rate >= 0.01 and  rate < 0.03:
                pre += "在1%-3%之间"
            elif rate >=0.03 and rate < 0.05:
                pre += "在3%-5%之间"
            else:
                pre += "大于5%"

        elif rate < 0:
            pre = "下跌幅度"
            if rate < 0.01:
                pre += "小于1%"
            elif rate >= 0.01 and rate < 0.03:
                pre += "在1%-3%之间"
            elif rate >= 0.03 and rate < 0.05:
                pre += "在3%-5%之间"
            else:
                pre += "大于5%"

        else:
            pre = "股价可能持平"
        rates.append(rate)
        rate_des.append(pre)

    return rates, rate_des

def get_basic_and_rate(symbol, data):
    """
    Get and match basic data to news dataframe.

    Args:
        symbol: str
            A-share market stock symbol
        data: DataFrame
            dated news data

    Return:
        financial news dataframe with matched basic_financial info and rates rates description
    """
    key_financials = ['报告期', '净利润同比增长率', '营业总收入同比增长率', '流动比率', '速动比率', '资产负债率']

    get_financials =1
    while get_financials:
    # load quarterly basic data
        try:
            basic_quarter_financials = ak.stock_financial_abstract_ths(symbol=symbol, indicator="按单季度")
            get_financials = 0
        except:
            time.sleep(1)
            continue
    basic_fin_dict = basic_quarter_financials.to_dict("index")
    basic_fin_list = [dict([(key, val) for key, val in basic_fin_dict[i].items() if (key in key_financials) and val])
                      for i in range(len(basic_fin_dict))]

    # match basic financial data to news dataframe
    matched_basic_fin = []
    for i, row in data.iterrows():

        newsweek_enddate = row['DATE']

        matched_basic = {}
        for basic in basic_fin_list:
            # match the most current financial report
            if basic["报告期"] < newsweek_enddate:
                matched_basic = basic
                break
        matched_basic_fin.append(json.dumps(matched_basic, ensure_ascii=False))
    rates, rates_des = transform_rate_data(data)
    data['基本面'] = matched_basic_fin
    data["比率"] = rates
    data["涨跌幅"] = rates_des

    return data

def get_basic_financials(symbol):
    key_financials = ['报告期', '净利润同比增长率', '营业总收入同比增长率', '流动比率', '速动比率', '资产负债率']

    get_financials = 1
    while get_financials:
        # load quarterly basic data
        try:
            basic_quarter_financials = ak.stock_financial_abstract_ths(symbol=symbol, indicator="按报告期")
            get_financials = 0
        except:
            time.sleep(1)
            continue
    basic_fin_dict = basic_quarter_financials.to_dict("index")
    basic_fin_list = [dict([(key, val) for key, val in basic_fin_dict[i].items() if (key in key_financials) and val])
                      for i in range(len(basic_fin_dict))]

    return basic_fin_list

def get_basic_new(basic_fin_list, data):
    """
    Get and match basic data to news dataframe.
    适用于最新数据
    Args:
        symbol: str
            A-share market stock symbol
        data: DataFrame
            dated news data

    Return:
        financial news dataframe with matched basic_financial info and rates rates description
    """
    if len(basic_fin_list) != 0:
        # match basic financial data to news dataframe
        # matched_basic_fin = []
        for i, row in data.iterrows():

            newsweek_enddate = row['发布时间']

            matched_basic = {}
            for basic in basic_fin_list:
                # match the most current financial report
                if basic["报告期"] < newsweek_enddate:
                    matched_basic = basic
                    break
            # matched_basic_fin.append(json.dumps(matched_basic, ensure_ascii=False))
    return matched_basic



    # match basic financial data to news dataframe
    # matched_basic_fin = []
    for i, row in data.iterrows():

        newsweek_enddate = row['发布时间']

        matched_basic = {}
        for basic in basic_fin_list:
            # match the most current financial report
            if basic["报告期"] < newsweek_enddate:
                matched_basic = basic
                break
        # matched_basic_fin.append(json.dumps(matched_basic, ensure_ascii=False))

    return basic_fin_list, matched_basic

def print_dict(dict):
    # dict = eval(dict)
    str = ""
    for key, value in dict.items():
        str += "{} {}\n".format(key, value)
    return str

def read_news_txt(file):
    f = open(file, "r", encoding="utf-8")
    news = f.read()
    news = [n.split("\t") for n in news]
    return news

def save_stock_news(file_path, data_path):
    codes = ['002607', '600857', '600519', '603286', '600867', '000797', '002908', '000566', '600319', '000411', '300326', '002341', '300644']

    df = pd.read_csv(file_path, delimiter="\t")
    df["CODE"] = df['CODE'].apply(lambda x: "%06d" % x)
    codes = np.unique(df["CODE"])
    codes = list(codes)

    for i in tqdm(range(len(codes))):
        df = get_news(codes[i])

        df.sort_values(by=["发布时间"], inplace=True)
        df = df.drop_duplicates(subset='发布时间')
        df = df.drop_duplicates(subset='新闻内容')
        df.reset_index(drop=True, inplace=True)

        filename = "{}/{}.csv".format(data_path, codes[i])
        if not os.path.isfile(filename):
            df.to_csv(filename, index=False)
        else:
            old_df = pd.read_csv(filename)

            start_date = old_df["发布时间"][len(old_df) - 1]
            df = df[df["发布时间"] > start_date]

            news_df = pd.concat([old_df, df], ignore_index=True)
            news_df.to_csv(filename, index=False)

if __name__ == "__main__":
    tickers =  ['002607', '600857', '600519', '603286', '600867', '000797', '002908', '000566', '600319', '000411', '300326', '002341', '300644']
    # query_gpt4(tickers)
    train_csv = "../data/train.csv"
    save_stock_news(train_csv, "../data/stock_news")