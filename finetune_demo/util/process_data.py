from utils import *
from datetime import datetime
import random
import numpy as np

random.seed(34557)
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

ORI_SYSTEM_PROMPT = "你是一名经验丰富的股票市场分析师。你的任务是根据公司在{}至{}的相关新闻和最近的季度财务状况，列出公司的积极发展和潜在担忧，然后结合你对整体金融经济市场的判断，对公司未来一周的股价变化提供预测和分析。" \
    "你的回答语言应为中文。你的回答格式应该如下：\n\n[积极发展]：\n1. ...\n\n[潜在担忧]：\n1. ...\n\n[预测和分析]：\n...\n[预测涨跌幅]：预测(上涨|下跌)二选一(小于1%|1%-3%|3%-5%|大于5%)四选一\n"

require = "\n让我们假设你对于下一周的预测是{}。根据上述的财务数据信息提供一个总结分析来支持你的预测。预测结果需要从你最后的分析中推断出来，因此不作为你分析的基础因素。" \
    "你的回答语言应为中文。你的回答格式应该如下：\n\n[积极发展]：\n1. ...\n\n[潜在担忧]：\n1. ...\n\n[预测和分析]：\n（要包含对涨跌幅度的预测）...\n[预测涨跌幅]：预测(上涨|下跌)二选一(小于1%|1%-3%|3%-5%|大于5%)四选一\n"

def read_news_and_prompt(file, symbol):
    df = pd.read_csv(file, delimiter="\t")
    df["CODE"] = df['CODE'].apply(lambda x: "%06d" % x)
    df.sort_values(by=["CREATED_DATE"], inplace=True)

    js = open("./data/my_data_qa.jsonl", "wt", encoding="utf-8")

    for i  in tqdm(range(len(df[:1000]))):
        tmp = {}
        tmp["conversation"] = []
        d = df.loc[i, :]
        symbol = d["CODE"]
        company_prompt, stock = get_company_prompt_new(symbol)
        start_date = n_weeks_before(d["CREATED_DATE"], 1)
        end_date = d["CREATED_DATE"]
        dfnews = df[df['CODE'] == symbol]
        dfnews = dfnews[(dfnews['CREATED_DATE'] >= start_date) & (dfnews['CREATED_DATE'] <= end_date)]
        d = get_basic_and_rate(symbol, dfnews)
        system_prompt = require.format(d["涨跌幅"].values[0])

        prompt = company_prompt+"\n"
        news_content = ""
        for j,d1 in dfnews.iterrows():
            news_content += "[新闻内容]\n"+d1["text_a"]+"\n"
        basic = "\n[金融基本面]\n"+print_dict(d["基本面"].values[0])
        prompt = ORI_SYSTEM_PROMPT+prompt+news_content+basic

        res = respones_gpt(system_prompt, prompt)

        sys = {"role":"system", "content":prompt}
        user = {"role":"user", "content":res}
        # assistant = {"role":"assistant", "content":res}
        tmp["conversation"].append(sys)
        tmp["conversation"].append(user)

        js.write(json.dumps(tmp, ensure_ascii=False) + "\n")

    return prompt


def get_new_and_format_prompt(symbol, database, mode = "train"):
    try:
        csv_path = "{}/{}.csv".format(database, symbol)
        df = pd.read_csv(csv_path)
    except:
        df = get_news(symbol)
    # df["CODE"] = df['CODE'].apply(lambda x: "%06d" % x)
    df.sort_values(by=["发布时间"], inplace=True)
    target_time = n_weeks_before(get_curday(), 2)

    df = df.drop_duplicates(subset='发布时间')
    df = df.drop_duplicates(subset='新闻内容')
    df.reset_index(drop=True, inplace=True)

    if mode != "train":
        start_date = n_weeks_before(get_curday(), 2)
        target_time = n_weeks_before(get_curday(), 1)
        df = df[(df["发布时间"]>start_date) & (df["发布时间"]<target_time)]
        df.reset_index(drop=True, inplace=True)
    else:
        df = df[df["发布时间"] < target_time]
        df.reset_index(drop=True, inplace=True)

    try:
        f_record = open("record/{}.txt".format(symbol), "r")
        records = f_record.readlines()
        df = df[df["发布时间"]>=records[-1]]
        df.reset_index(drop=True, inplace=True)
    except:
        pass

    if len(df)==0:
        return None

    f_record = open("record/{}.txt".format(symbol), "a")
    stock_data = get_stock_all(symbol, df["发布时间"][0])
    if stock_data is None:
        return None
    stock_data["日期"] = stock_data["日期"].apply(lambda x: datetime.strftime(x, '%Y-%m-%d %H:%M:%S'))

    matched_basic_all = get_basic_financials(symbol)
    js = open("../data/fin_news_stock/{}.jsonl".format(mode), "at", encoding="utf-8")

    company_prompt = ""
    for i in tqdm(range(len(df))):
        tmp = {}
        tmp["conversation"] = []
        d = df.loc[i, :]
        # symbol = d["CODE"]

        f_record.write(d["发布时间"]+"\n")

        start_date = n_weeks_before(d["发布时间"], 1)
        end_date = d["发布时间"]
        # dfnews = df[df['CODE'] == symbol]
        dfnews = df[(df['发布时间'] >= start_date) & (df['发布时间'] <= end_date)]
        future_date = n_weeks_after(d["发布时间"], 1)

        if len(dfnews)<2:
            continue
        if i == 0 or company_prompt=="":
            company_prompt, stock = get_company_prompt_new(symbol)
        stock_data_interval_past = stock_data[(stock_data['日期'] >= start_date) & (stock_data['日期'] <= end_date)]
        matched_basic = get_basic_new(matched_basic_all, dfnews)

        stock_data_interval_future = stock_data[(stock_data['日期'] >= end_date) & (stock_data['日期'] <= future_date)]
        future_rate = transform_rate_data_online(stock_data_interval_future)
        system_prompt = require.format(future_rate[1][0])

        prompt = company_prompt+"\n"
        past_rate = transform_rate_data_online(stock_data_interval_past)[1][0]
        volatility = "\n 自{} 至 {} {}股票{} \n".format(start_date.split(" ")[0], end_date.split(" ")[0], stock, past_rate)
        prompt += prompt+volatility
        news_content = ""

        for j,d1 in dfnews.iterrows():
            news_content += "[新闻标题]"+"\n"+d1["新闻标题"]+"\n"+"发布时间："+d1["发布时间"]+"\n"+"[新闻内容]"+"\n"+d1["新闻内容"]+"\n"
        basic = "\n如下所列为600519近期的一些金融基本面信息，记录时间为{}:\n[金融基本面]\n".format(matched_basic["报告期"])+print_dict(matched_basic)
        gptprompt = prompt+news_content+basic+system_prompt
        content = prompt+news_content+basic

        res = respones_gpt(system_prompt, gptprompt)

        system_prompt_glm = ORI_SYSTEM_PROMPT.format(start_date.split(" ")[0],end_date.split(" ")[0])

        inp = system_prompt_glm+"\n"+content
        sys = {"role":"system", "content":system_prompt_glm}
        user = {"role":"user", "content":content}
        assistant = {"role":"assistant", "content":res}
        # assistant = {"role":"assistant", "content":res}
        tmp["conversation"].append(sys)
        tmp["conversation"].append(user)
        tmp["conversation"].append(assistant)

        js.write(json.dumps(tmp, ensure_ascii=False) + "\n")


def sample_stock_new_predict(file, target_path):
    df = pd.read_csv(file, delimiter="\t")
    df["CODE"] = df['CODE'].apply(lambda x: "%06d" % x)
    codes = np.unique(df["CODE"])
    random.shuffle(codes)

    codes[0] = "600519"

    total = 10
    num = 0
    for code in codes:
        if num > total:
            break
        csv_path = "{}/{}.csv".format(target_path, code)
        df = pd.read_csv(csv_path)
        if len(df) < 700:
            df.close()
            continue
        # get_new_and_format_prompt(code, target_path)
        get_new_and_format_prompt(code, target_path,"dev")


if __name__ == "__main__":
    # comp = "./data/raw/company announcement.txt"
    # ss = read_news_txt(comp)
    target_path = "../data/stock_news/"
    file_path = "../data/train.csv"
    sample_stock_new_predict(file_path, target_path)
    symbol = "000967"
    # news = read_news_and_prompt(file_path, symbol)
    # prompt = get_new_and_format_prompt(symbol)
    # get_stock_news()
    # path = "./data/raw/stock_daily/*.csv"
    # df = stock_concat2h5(path)
