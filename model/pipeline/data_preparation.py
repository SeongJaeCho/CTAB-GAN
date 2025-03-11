import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection
import os

class DataPrep(object):
    
    """
    Data preparation class for pre-processing input data and post-processing generated data

    Variables:
    1) raw_df -> dataframe containing input data
    2) categorical -> list of categorical columns
    3) log -> list of skewed exponential numerical columns
    4) mixed -> dictionary of "mixed" column names with corresponding categorical modes 
    5) integer -> list of numeric columns without floating numbers
    6) type -> dictionary of problem type (i.e classification/regression) and target column
    7) test_ratio -> ratio of size of test to train dataset

    Methods:
    1) __init__() -> instantiates DataPrep object and handles the pre-processing steps for feeding it to the training algorithm
    2) inverse_prep() -> deals with post-processing of the generated data to have the same format as the original dataset
    """    

    def __init__(self, raw_df: pd.DataFrame, categorical: list, log: list, mixed: dict, integer: list, type: dict, test_ratio: float):
        
        self.categorical_columns = categorical
        self.log_columns = log
        self.mixed_columns = mixed
        self.integer_columns = integer
        self.column_types = dict()
        self.column_types["categorical"] = []
        self.column_types["mixed"] = {}
        self.lower_bounds = {}
        self.label_encoder_list = []

        # ✅ 기존 train/test 데이터가 존재하면 불러오기
        if os.path.exists("train_real_preprocessed.csv") and os.path.exists("test_real_preprocessed.csv"):
            print("✅ 기존 전처리된 train/test 데이터를 불러옵니다.")
            self.train_df = pd.read_csv("train_real_preprocessed.csv")
            self.test_df = pd.read_csv("test_real_preprocessed.csv")
            return  # 기존 데이터를 그대로 사용

        print("🚀 새로운 train/test 데이터를 생성합니다.")
        
        # ✅ Target Column 설정
        target_col = list(type.values())[0]
        y_real = raw_df[target_col]
        X_real = raw_df.drop(columns=[target_col])

        # ✅ Train / Test 데이터 분리
        X_train_real, X_test_real, y_train_real, y_test_real = model_selection.train_test_split(
            X_real, y_real, test_size=test_ratio, stratify=y_real, random_state=42
        )

        # Train 데이터에 타겟 컬럼 추가
        X_train_real[target_col] = y_train_real
        X_test_real[target_col] = y_test_real

        # ✅ Train/Test 데이터 저장 (원본 데이터)
        X_train_real.to_csv("train_real.csv", index=False)
        X_test_real.to_csv("test_real.csv", index=False)
        print("✅ 원본 Train/Test 데이터 저장 완료: train_real.csv, test_real.csv")

        # ✅ Train 데이터 전처리 수행
        self.df = X_train_real.copy()
        self.df = self.df.replace(r' ', np.nan)
        self.df = self.df.fillna('empty')

        # ✅ 결측값 처리 (-9999999로 변환)
        all_columns = set(self.df.columns)
        irrelevant_missing_columns = set(self.categorical_columns)
        relevant_missing_columns = list(all_columns - irrelevant_missing_columns)

        for i in relevant_missing_columns:
            if i in list(self.mixed_columns.keys()):
                if "empty" in list(self.df[i].values):
                    self.df[i] = self.df[i].apply(lambda x: -9999999 if x == "empty" else x)
                    self.mixed_columns[i].append(-9999999)
            else:
                if "empty" in list(self.df[i].values):   
                    self.df[i] = self.df[i].apply(lambda x: -9999999 if x == "empty" else x)
                    self.mixed_columns[i] = [-9999999]

        # ✅ 로그 변환 적용
        if self.log_columns:
            for log_column in self.log_columns:
                eps = 1
                lower = np.min(self.df.loc[self.df[log_column] != -9999999][log_column].values)
                self.lower_bounds[log_column] = lower
                if lower > 0: 
                    self.df[log_column] = self.df[log_column].apply(lambda x: np.log(x) if x != -9999999 else -9999999)
                elif lower == 0:
                    self.df[log_column] = self.df[log_column].apply(lambda x: np.log(x + eps) if x != -9999999 else -9999999) 
                else:
                    self.df[log_column] = self.df[log_column].apply(lambda x: np.log(x - lower + eps) if x != -9999999 else -9999999)

        # ✅ Categorical 데이터 Label Encoding
        for column_index, column in enumerate(self.df.columns):
            if column in self.categorical_columns:        
                label_encoder = preprocessing.LabelEncoder()
                self.df[column] = self.df[column].astype(str)
                label_encoder.fit(self.df[column])
                transformed_column = label_encoder.transform(self.df[column])
                self.df[column] = transformed_column
                self.label_encoder_list.append({"column": column, "label_encoder": label_encoder})
                self.column_types["categorical"].append(column_index)
            elif column in self.mixed_columns:
                self.column_types["mixed"][column_index] = self.mixed_columns[column]

        # ✅ 전처리된 Train 데이터 저장
        self.df.to_csv("train_real_preprocessed.csv", index=False)
        print("✅ 전처리된 Train 데이터 저장 완료: train_real_preprocessed.csv")

        # ✅ Test 데이터도 동일한 전처리 수행
        self.test_df = X_test_real.copy()
        self.test_df = self.test_df.replace(r' ', np.nan)
        self.test_df = self.test_df.fillna('empty')

        for i in relevant_missing_columns:
            if i in list(self.mixed_columns.keys()):
                if "empty" in list(self.test_df[i].values):
                    self.test_df[i] = self.test_df[i].apply(lambda x: -9999999 if x == "empty" else x)
            else:
                if "empty" in list(self.test_df[i].values):   
                    self.test_df[i] = self.test_df[i].apply(lambda x: -9999999 if x == "empty" else x)

        # ✅ 로그 변환 및 Label Encoding 적용 (같은 방식)
        for column in self.test_df.columns:
            if column in self.log_columns:
                lower_bound = self.lower_bounds[column]
                self.test_df[column] = self.test_df[column].apply(lambda x: np.log(x - lower_bound + 1) if x != -9999999 else -9999999)

        for encoder in self.label_encoder_list:
            le = encoder["label_encoder"]
            self.test_df[encoder["column"]] = le.transform(self.test_df[encoder["column"]].astype(str))

        # ✅ 전처리된 Test 데이터 저장
        self.test_df.to_csv("test_real_preprocessed.csv", index=False)
        print("✅ 전처리된 Test 데이터 저장 완료: test_real_preprocessed.csv")

        super().__init__()
