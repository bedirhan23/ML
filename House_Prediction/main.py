import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.cluster import KMeans

class AirbnbPricePrediction:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)

    def data_preparation(self):
        self.data = self.data.drop(["Business", "Private Room", "Shared Room", "Attraction Index", "Restraunt Index"], axis=1)

        sns.set(font_scale=0.9)
        plt.figure(figsize=(16, 10))
        plt.bar(self.data.index, self.data['Price'])
        sns.heatmap(
            self.data.corr(),
            cmap='RdBu_r',
            annot=True,
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            linecolor='black',
            square=False,
        )
        plt.show()

        return self.data

    def encode_categorical_features(self):
        categorical_cols = ["Day", "Room Type"]
        encoded_features = pd.get_dummies(self.data[categorical_cols])
        self.data = pd.concat([self.data, encoded_features], axis=1)
        self.data.drop(categorical_cols, axis=1, inplace=True)

        int_columns = ["Superhost", "Room Type_Entire home/apt", "Room Type_Private room",
                       "Room Type_Shared room", "Day_Weekday", "Day_Weekend"]

        for col in int_columns:
            self.data[col] = self.data[col].astype(int)

        return self.data

    def input_variables(self):
        target_city = input("Kiralama yapmak istediğiniz şehri giriniz: ")

        self.data = self.data[self.data['City'] == target_city]
        self.data = self.data.drop("City", axis=1)

        self.data = self.encode_categorical_features()
        self.data = self.data_preparation()

        ev_ozellikleri = self.get_input_features(self.data.columns)
        best_predict = float(self.models_training(ev_ozellikleri))

        self.add_predicted_data(ev_ozellikleri, best_predict)
        self.elbow(10)
        self.kmeans(4, best_predict)

    def get_input_features(self, columns):
        ev_ozellikleri = {}
        for feature in columns:
            if feature == "Price":
                pass
            elif feature == "Cleanliness Rating":
                ev_ozellikleri[feature] = self.data["Cleanliness Rating"].mean()
            elif feature == "Guest Satisfaction":
                ev_ozellikleri[feature] = self.data["Guest Satisfaction"].mean()
            else:
                ev_ozellikleri[feature] = float(input(f"{feature} değerini giriniz: "))
        return ev_ozellikleri

    def models_training(self, features):
        models = [
            LinearRegression(),
            RandomForestRegressor(n_estimators=100, random_state=101),
            XGBRegressor(n_estimators=100, random_state=101)
        ]

        model_names = [
            "Linear Regression",
            "Random Forest",
            "XGBoost"
        ]

        X = self.data.drop("Price", axis=1)
        y = self.data["Price"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

        best_mae = float("inf")
        best_predict = 0
        best_model = None
        best_model_index = 0
        mae_values = []

        for model in models:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            mae_values.append(mae)

            if mae < best_mae:
                best_mae = mae
                best_predict = model.predict(pd.DataFrame(features, index=[0]))
                best_model = best_model_index

            best_model_index += 1

        print("En iyi tahmin ",model_names[best_model]," ile ",best_predict[0]," olarak yapılmıştır.")
        self.plot_mae(mae_values)

        return best_predict

    def plot_mae(self, mae_values):
        model_names = ["Linear Regression", "Random Forest", "XGBoost"]

        plt.figure(figsize=(8, 6))
        plt.bar(model_names, mae_values)
        plt.xlabel("Model")
        plt.ylabel("Mean Absolute Error (MAE)")
        plt.title("MAE Values for Different Models")
        plt.show()

    def add_predicted_data(self, features, prediction):
        features["Price"] = prediction

        features_df = pd.DataFrame([features])

        self.data = pd.concat([self.data, features_df], ignore_index=True)

    def elbow(self, max_clusters):
        distortions = []
        for i in range(1, max_clusters+1):
            kmeans = KMeans(n_clusters=i, random_state=101, n_init=10)
            kmeans.fit(self.data)
            distortions.append(kmeans.inertia_)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_clusters+1), distortions, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Distortion')
        plt.title('Elbow Method')
        plt.show()

    def kmeans(self, num_clusters, prediction):
        kmeans = KMeans(n_clusters=num_clusters, random_state=101, n_init=10)
        kmeans.fit(self.data)

        self.data['Cluster'] = kmeans.labels_

        # Belirli bir değeri arayın ve bulunan satırı getirin
        filtered_row = self.data.loc[self.data["Price"] == prediction]

        # Cluster değerini döndürün
        cluster_value = filtered_row["Cluster"].iloc[0]

        # Evimizin bulunduğu kümedeki evleri döndür
        self.data = self.data[self.data['Cluster'] == cluster_value]

        # Fiyatlarına göre sırala
        self.data = self.data.sort_values(by="Price")
        self.data = self.data.reset_index(drop=True)

        # Index sıfırlandığı için güncelleme
        filtered_row = self.data.loc[self.data["Price"] == prediction]

        # Bulduğunuz satırın indeksini alın
        index = filtered_row.index[0]

        # Önceki 3 ve sonraki 3 satırı istediğimiz için, eğer öncesinde 3 satır yok ise eksiğini sonrasındaki satırlardan eklemeli
        if index < 3:
            # İndeksten önceki satırları alın
            before_rows = self.data.iloc[max(0, index - 3):index]

            # İndeksten sonraki satırları alın
            after_rows = self.data.iloc[index + 1:index + 4 + (3 - index)]

        else:
            # İndeksten önceki 3 satırı alın
            before_rows = self.data.iloc[index - 3:index]

            # İndeksten sonraki 3 satırı alın
            after_rows = self.data.iloc[index + 1:index + 4]

        # Önceki ve sonraki satırları birleştirin
        selected_rows = pd.concat([before_rows, after_rows])

        # Sütunları güzelleştirme
        selected_rows["Day"] = ""
        selected_rows.loc[selected_rows['Day_Weekday'] == 1, 'Day'] = 'Weekday'
        selected_rows.loc[selected_rows['Day_Weekend'] == 1, 'Day'] = 'Weekend'

        selected_rows["Room Type"] = ""
        selected_rows.loc[selected_rows['Room Type_Entire home/apt'] == 1, 'Room Type'] = 'Entire home/apt'
        selected_rows.loc[selected_rows['Room Type_Private room'] == 1, 'Room Type'] = 'Private room'
        selected_rows.loc[selected_rows['Room Type_Shared room'] == 1, 'Room Type'] = 'Shared room'

        # Eski sütunları sil
        selected_rows = selected_rows.drop(
            ['Day_Weekend', 'Day_Weekday', 'Room Type_Entire home/apt', 'Room Type_Private room',
             'Room Type_Shared room', 'Cluster'], axis=1)

        # Sonucu yazdırın
        print(selected_rows.to_string(index=False))



# Örnek kullanım
airbnb = AirbnbPricePrediction("Aemf1.csv")
airbnb.input_variables()
