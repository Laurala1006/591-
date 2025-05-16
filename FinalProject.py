from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium import webdriver
import pandas as pd
import time
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

#啟動 Chrome 瀏覽器
driver = webdriver.Chrome()
driver.maximize_window()

#載入網頁
url = "https://rent.591.com.tw/"
driver.get(url)

# 點擊網頁上所需的按鈕
button_condition = driver.find_element(By.XPATH, "/html/body/div[2]/div/div[2]/section[2]/ul/li[3]")
button_condition.click()
button_places = driver.find_element(By.XPATH, "/html/body/div[2]/div/div[2]/section[1]/ul[2]")

# 創建空的 DataFrame 來存儲所有頁面的資料
all_house_data = pd.DataFrame()

for i in range(1,6):
    button_place = button_places.find_elements(By.TAG_NAME, "li")[i]
    button_place.click()
    time.sleep(3) 
   
k=1
while k<33:
    # 滾動網頁到最下方
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)
    # 找到網頁要爬蟲的欄位，並創立housrDataList作為存資料的List
    information = driver.find_element(By.XPATH, "/html/body/div[2]/div/div[3]/div[1]")
    # 將房屋排序改為最新
    Newest_list = information.find_element(By.XPATH, "/html/body/div[2]/div/div[3]/div[1]/section[2]/div/ul/li[2]") 
    Newest_list.click()
    time.sleep(5)

    houseDataList = []
    houseNames = information.find_elements(By.CLASS_NAME, "item-title")
    rentPrices = information.find_elements(By.CLASS_NAME, "item-price-text") #每月租金
    houseSpaces = information.find_elements(By.CLASS_NAME, "item-style") #房間大小
    subwayDists = information.find_elements(By.CSS_SELECTOR, ".item-tip.subway") #離捷運站的距離

    for rentPrice, houseSpace, subwayDist, houseName in zip(rentPrices, houseSpaces, subwayDists, houseNames):

        area = houseSpace.find_elements(By.TAG_NAME, "li")[1]
        floor = houseSpace.find_elements(By.TAG_NAME, "li")[2]
        if area.text.strip() == "樓中樓":
            area = houseSpace.find_elements(By.TAG_NAME, "li")[2]
            floor = houseSpace.find_elements(By.TAG_NAME, "li")[3]

        #subwayDist = re.findall(r'\d',subwayDist.text) #[2,1,2]
        subwayDist = ''.join(re.findall(r'\d', subwayDist.text)) #212
        rentPrice = ''.join(re.findall(r'\d', rentPrice.text))
        area = ''.join(re.findall(r'\d+\.\d+|\d+', area.text))
         # 提取樓層的第一個數字
        first_floor_number = re.search(r'\d+', floor.text.strip()).group() if re.search(r'\d+', floor.text.strip()) else None

        houseData = {
            "Name": button_place.text,
            #"House":houseName.text,
            "Rent Price": rentPrice,
            "House Space": area,
            "Floor":first_floor_number,
            "Subway Distances":subwayDist
        }
        houseDataList.append(houseData)

    df_page = pd.DataFrame(houseDataList)
    all_house_data = pd.concat([all_house_data, df_page], ignore_index=True)

    pages_num = information.find_element(By.CLASS_NAME, "page-limit")
    for num in range(0,20):
        page = pages_num.find_elements(By. TAG_NAME, "a")[num]
        if page.text == "下一頁":
            next_page = page
            next_page.click()
            break
    time.sleep(5)
    k=k+1

print(all_house_data)    
   
driver.close()

#分析房租的價格
# 將租金轉換為數字
all_house_data['Rent Price'] = pd.to_numeric(all_house_data['Rent Price'], errors='coerce')
print("缺失值檢查：")
print(all_house_data.isnull().sum())

# 移除缺失值
all_house_data.dropna(inplace=True)

# 將類別變數轉換為獨熱編碼
all_house_data = pd.get_dummies(all_house_data, columns=['Name'], drop_first=True)

# 檢查數據類型確保所有特徵都是數值型
print(all_house_data.dtypes)

# 選擇特徵和目標變數
features = all_house_data.drop(columns=['Rent Price'])
target = all_house_data['Rent Price']

# 拆分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# 標準化處理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 線性回歸模型
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 預測
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# 評估模型性能
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Train RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")
print(f"Train R2: {train_r2}")
print(f"Test R2: {test_r2}")

# 可視化實際值和預測值
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Rent Price')
plt.ylabel('Predicted Rent Price')
plt.title('Actual vs Predicted Rent Price')
plt.show()

# 計算相關矩陣
correlation_matrix = all_house_data.corr()

# 繪製熱圖
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

'''
# 檢查數據中的缺失值
print("缺失值檢查：")
print(all_house_data.isnull().sum())

# 計算四分位數
Q1 = all_house_data['Rent Price'].quantile(0.25)
Q3 = all_house_data['Rent Price'].quantile(0.75)

# 計算四分位距
IQR = Q3 - Q1

# 定義異常值的範圍
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 標記異常值
outliers = all_house_data[(all_house_data['Rent Price'] < lower_bound) | (all_house_data['Rent Price'] > upper_bound)]
print("\n異常值範圍：")
print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")

# 處理異常值，替換為上下界值
all_house_data.loc[all_house_data['Rent Price'] < lower_bound, 'Rent Price'] = lower_bound
all_house_data.loc[all_house_data['Rent Price'] > upper_bound, 'Rent Price'] = upper_bound

# 輸出最終的房屋數據
print("\n最終房屋數據：")
print(all_house_data)

# 查看是否還有缺失值
print("\n檢查是否還有缺失值：")
print(all_house_data.isnull().sum())

# 繪製特徵相關性熱圖
plt.figure(figsize=(10, 8))
sns.heatmap(all_house_data[['Rent Price', 'House Space', 'Subway Distances']].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap: Rent Price, House Space, Subway Distances')
plt.show()

# 分割訓練集和測試集
X = all_house_data[['House Space', 'Subway Distances']]
y = all_house_data['Rent Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立決策樹回歸模型
tree_reg = DecisionTreeRegressor(max_depth=5, random_state=42)  # 設置最大深度為5，可以根據需求調整

# 在訓練集上訓練模型
tree_reg.fit(X_train, y_train)

# 在測試集上進行預測
y_pred = tree_reg.predict(X_test)

# 計算均方誤差（MSE）和決定係數（R^2）
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n決策樹模型評估結果：")
print(f"均方誤差（MSE）: {mse}")
print(f"決定係數（R^2）: {r2}")

# 可視化決策樹
from sklearn.tree import export_graphviz
import graphviz

dot_data = export_graphviz(tree_reg, out_file=None, 
                           feature_names=X.columns.tolist(),  
                           filled=True, rounded=True,  
                           special_characters=True)

graph = graphviz.Source(dot_data)
graph.render("rent_decision_tree", format='png', cleanup=True)
'''
'''
# 分割訓練集和測試集
X = all_house_data[['House Space', 'Subway Distances']]
y = all_house_data['Rent Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特徵標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 建立和訓練KNN模型
knn = KNeighborsRegressor(n_neighbors=20)
knn.fit(X_train_scaled, y_train)

# 在測試集上進行預測
y_pred = knn.predict(X_test_scaled)

# 評估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nKNN模型評估結果：")
print(f"均方誤差（MSE）: {mse}")
print(f"決定係數（R^2）: {r2}")

# 繪製實際值與預測值的散點圖
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.xlabel('Actual Rent Price')
plt.ylabel('Predicted Rent Price')
plt.title('Actual vs Predicted Rent Price (KNN)')
plt.grid(True)
plt.show()

# 繪製預測誤差的分佈圖
error = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.hist(error, bins=30, edgecolor='black')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors (KNN)')
plt.grid(True)
plt.show()
'''
'''
# 特徵標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 設置不同的K值進行測試
k_values = range(1, 21)
mse_values = []
r2_values = []

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mse_values.append(mse)
    r2_values.append(r2)
    print(f"K = {k}: MSE = {mse}, R^2 = {r2}")

# 繪製不同K值下的MSE和R^2曲線
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(k_values, mse_values, marker='o')
plt.title('KNN: Mean Squared Error vs K')
plt.xlabel('K')
plt.ylabel('Mean Squared Error')

plt.subplot(1, 2, 2)
plt.plot(k_values, r2_values, marker='o')
plt.title('KNN: R^2 Score vs K')
plt.xlabel('K')
plt.ylabel('R^2 Score')

plt.tight_layout()
plt.show()
'''
'''
#每月租金
rentPrices = driver.find_elements(By.CLASS_NAME, "item-price-text")
for rentPrice in rentPrices:
    houseData = {"Rent Price": rentPrice.text}  # 每次都新增一個字典才不會蓋掉舊資料
    houseDataList.append(houseData)
    print(rentPrice.text)

#房間大小
houseSpaces = driver.find_elements(By.CLASS_NAME, "item-style")
for houseSpace in houseSpaces:
    area = houseSpace.find_elements(By.TAG_NAME, "li")[1]
    if area.text.strip() == "樓中樓":
       area = houseSpace.find_elements(By.TAG_NAME, "li")[2] 
    houseData = {"House Space": area.text.strip()}
    houseDataList.append(houseData)
    print(houseSpace.text)
'''    


