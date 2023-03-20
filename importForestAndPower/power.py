import pandas as pd
import psycopg2
from io import StringIO

path='data/power/household_power_consumption.txt'
fr=open(path,'r')
all_lines=fr.readlines()
dataset=[]
for line in all_lines:
    line=line.strip().split(';')
    dataset.append(line[2:])
dataset = dataset[1:]

result = []
for d in dataset:
    if '?' not in d:
        result.append(d)
result = pd.DataFrame(result)


f = StringIO()
# DataFrame 类型数据转换为IO缓冲区中的str类型
result.to_csv(f, sep='\t', index=False, header=False)
# 把f的游标移到第一位，write方法后，游标会变成最尾，使用StringIO(**)则不会
f.seek(0)
# 连接数据库
conn = psycopg2.connect(host='127.0.0.1', user="postgres", password="123456", database="aidb")
# 创建游标
cur = conn.cursor()
# 将内存对象f中的数据写入数据库，参数columns为所有列的元组
cur.copy_from(f, 'power', columns=('Global_active_power1','Global_reactive_power1','Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3'))
# 提交
conn.commit()
cur.close()
conn.close()
print('成功写入数据库')





