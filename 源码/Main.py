import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression, VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
pd.set_option('display.float_format',lambda x : '%.2f' % x)


if __name__ == '__main__':
    # 数据文件
    train_path = './kc_house_data.csv'
    # 读取数据，非空
    data = pd.read_csv(train_path).dropna()

    # 显示前5列
    # print(data.head(5))

    # 时间均在14.05-15.05之间，忽略时间对房价的影响
    data = data.drop(labels="date", axis=1)
    #ID只作为标识，删去
    data = data.drop(labels="id", axis=1)

    #选择X,Y
    X = data.drop(labels="price", axis=1)
    y = data["price"]
    # 显示前5列
    # print(data.head(5))


    #使用Pipeline
    clf = Pipeline([
        ('feature_selection', VarianceThreshold(threshold=0.0)),
        ('Regression', LinearRegression())
    ])

    # print(data.shape)

    #划分train,val,train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf.fit(X_train,y_train)
    # print(clf.score(X_train, y_train))
    y_pred=clf.predict(X_test)
    print(f"R方值(R2)：{r2_score(y_test,y_pred)}")

    d={
        'test':y_test,
        'predict':y_pred
    }
    df=pd.DataFrame(d)
    df=df.sort_values(axis=0,ascending=True,by='test')
    df = df.reset_index(drop=True)
    print(f"部分预测结果：\n{df[500:600]}")

    # 画图
    x=range(0,len(df))
    plt.plot(x,df['predict'],color='orange',label='predict')
    plt.plot(x,df['test'],color='red',label='fact')
    plt.legend(loc='upper right')
    plt.xlabel('index')
    plt.ylabel('price' )
    plt.title('2014-2015 House Sales in King County, USA')
    plt.show()

    # 保存结果
    df.to_csv('Result.csv', index=0)