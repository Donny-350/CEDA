import psycopg2
import csv

# 连接postgresql
conn = psycopg2.connect(host="127.0.0.1", port=5432,user="postgres",password="123456", database='aidb')
cur = conn.cursor()

path = 'result/job-light.csv'
header = ['sql', 'estimate', 'actual']
sqlTxt = []

sql_prefix = 'explain '
with open('data/job-light.sql', 'r') as f:
    sqlContext = f.readlines()
    i = 1
    for sql in sqlContext:
        cur.execute(sql)
        cardinality_actual = cur.fetchall()[0][0]
        sql = sql.replace('COUNT(', '')
        sql = sql.replace(')', '')
        cur.execute(sql_prefix + sql)
        result = cur.fetchall()

        result_list = result[0][0].split('rows=')
        cardinality_estimate = result_list[1].split(' ')[0]

        content = []

        content.append(sql)
        content.append(cardinality_estimate)
        content.append(cardinality_actual)
        sqlTxt.append(content)
        print(i, sql, cardinality_estimate, cardinality_actual)
        i += 1

    with open(path, 'w', encoding='utf-8', newline='') as f1:
        writer = csv.writer(f1)
        writer.writerow(header)
        writer.writerows(sqlTxt)

