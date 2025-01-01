#实现了一个基于Streamlit框架的Web应用程序，用于分析和预测开源项目的活动模式。
# 主要功能包括从指定的GitHub仓库获取数据、数据预处理、使用ARIMA模型进行预测，并通过热力图和折线图进行数据可视化。 使用streamlit run main.py在py控制台运行
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st

# 获取数据
def fetch_data(repo_name):
    base_url = 'https://oss.x-lab.info/open_digger/github/'
    url = f"{base_url}{repo_name}/active_dates_and_times.json"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # 检查请求是否成功
        data = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"获取数据失败: {e}")
        return None
    return data

# 数据预处理
def preprocess_data(data):
    hours = list(range(24))  # 每小时数据
    days = ['周日', '周六', '周五', '周四', '周三', '周二', '周一']  # 一周的天数
    values = None

    # 对每个小时的数据进行累加
    for day_data in data.values():
        if values is None:
            values = np.array(day_data)
        else:
            values += np.array(day_data)

    # 对数据进行对数变换，平滑数据
    values = np.log(values + 1)

    # 标准化数据
    max_value = np.max(values)
    values = np.ceil(values * 10 / max_value)

    # 按小时和天数生成数据
    input_data = []
    for d in range(7):  # 7 天
        for h in range(24):  # 24 小时
            value = values[d * 24 + h]
            input_data.append([h, 6 - d, value if not np.isnan(value) else 0])

    # 打印调试信息
    print(f"输入数据长度: {len(input_data)}")
    print(f"输入数据样本: {input_data[:24]}")

    return input_data, hours, days

# ARIMA 模型预测
def arima_forecast(series, steps=30):
    # 使用 ARIMA 模型进行预测
    model = ARIMA(series, order=(1, 1, 0))  # ARIMA 参数 (p, d, q)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

# 数据分析与可视化
def visualize_data(input_data, hours, days):
    # 确保 input_data 是一个 7x24 的数组
    reshaped_data = np.array([row[2] for row in input_data]).reshape(7, 24)

    # 使用 Plotly 生成热力图
    fig = px.imshow(
        reshaped_data,  # 7天, 24小时
        labels={'x': '小时', 'y': '星期'},
        x=hours,
        y=days,
        title="活跃日期和时间"
    )
    st.plotly_chart(fig)

# 主函数
def main():
    st.title('开源项目活动分析')

    # 用户输入仓库名称
    repo_name = st.text_input('请输入仓库名称', 'X-lab2017/open-digger')

    if st.button('获取并分析数据'):
        raw_data = fetch_data(repo_name)
        if raw_data is None:
            st.error("无法获取数据，请检查仓库名称或稍后再试。")
            return

        # 数据预处理
        input_data, hours, days = preprocess_data(raw_data)

        # 可视化展示历史活动数据
        visualize_data(input_data, hours, days)

        # 基于历史数据进行 ARIMA 模型预测
        try:
            daily_activity = [sum(input_data[i][2] for i in range(j * 24, (j + 1) * 24)) for j in range(7)]
            forecast = arima_forecast(daily_activity, steps=30)
        except Exception as e:
            st.error(f"ARIMA 预测错误: {e}")
            forecast = np.zeros(30)  # 出现问题时返回零数组

        # 显示预测结果
        forecast_dates = pd.date_range(
            start=pd.Timestamp.now(),
            periods=31,  # 包括今天的预测
            freq='D'
        )

        forecast_fig = px.line(
            x=forecast_dates[1:],  # 从第二天开始
            y=forecast,
            title="每日活动预测",
            labels={'x': '日期', 'y': '预测活动水平'}
        )
        st.plotly_chart(forecast_fig)


if __name__ == '__main__':
    main()

