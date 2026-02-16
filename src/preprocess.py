
import pandas as pd
from sklearn.model_selection import train_test_split
import great_expectations as ge
from pandera import Column, DataFrameSchema, Check, String, Int, Float

class DataPreprocessor:
    def __init__(self, df):
        self.df = df

    # Базовая очистка данных
    def clean_data(self):
        # Убираем строки с пропущенными значениями
        self.df.dropna(inplace=True)
        
        # Преобразование типов данных
        for col in ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']:
            self.df[col] = self.df[col].astype('category')  # Преобразуем численные задержки в категории
            
        return self.df

    # Агрегированные признаки
    def feature_engineering(self):
        # Сумма задержек по платежам за последние полгода
        self.df['total_delay'] = self.df[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].sum(axis=1)
        
        # Биннинг возрастов (разделение на группы)
        bins = [20, 30, 40, 50, 60, 70]
        labels = ["Молодые", "Средний возраст", "Старше среднего", "Предпенсионеры", "Пожилые"]
        self.df["age_group"] = pd.cut(self.df["AGE"], bins=bins, labels=labels)
        
        return self.df

    # Проверка данных с использованием Great Expectations
    def validate_data(self):
        context = ge.get_context()
        suite = context.create_expectation_suite(expectation_suite_name="credit_card_default_suite")
        
        # Применяем ожидания
        batch_request = {
            "datasource_name": "pandas_datasource",
            "data_asset_name": "credit_card_dataset",
            "batch_spec": {"pandas_df": self.df}
        }
        validator = context.get_validator(batch_request=batch_request, expectation_suite=suite)
        
        # Устанавливаем ожидания
        validator.expect_column_values_to_be_between(column="LIMIT_BAL", min_value=0, max_value=None)
        validator.expect_column_values_to_not_be_null(column="SEX")
        validator.expect_column_values_to_be_in_set(column="EDUCATION", value_set={1, 2, 3})
        validator.expect_column_values_to_be_in_set(column="MARRIAGE", value_set={1, 2})
        validator.expect_column_values_to_be_between(column="AGE", min_value=20, max_value=80)
        
        # Запускаем проверку
        results = validator.save_expectation_suite(discards_failed_expectations=False)
        run_results = context.run_validation_operator("action_list_operator", assets_to_validate=[validator])
        
        if not run_results.success:
            raise ValueError("Data validation failed with Great Expectations.")
        
        return True

    # Тестирование схемы данных с помощью Pandera
    @staticmethod
    def schema():
        return DataFrameSchema(
            columns={
                "ID": Column(Int, nullable=False),
                "LIMIT_BAL": Column(Float, nullable=False),
                "SEX": Column(Int, checks=Check.isin([1, 2])),
                "EDUCATION": Column(Int, checks=Check.isin([1, 2, 3])),
                "MARRIAGE": Column(Int, checks=Check.isin([1, 2])),
                "AGE": Column(Int, checks=Check.greater_than_or_equal_to(min_value=20)),
                "default.payment.next.month": Column(Int, checks=Check.isin([0, 1])),
                "PAY_0": Column("category"),       # Добавляем столбцы с категориями
                "PAY_2": Column("category"),
                "PAY_3": Column("category"),
                "PAY_4": Column("category"),
                "PAY_5": Column("category"),
                "PAY_6": Column("category"),
            },
            strict=True,
            coerce=True
        )

if __name__ == "__main__":
    df = pd.read_csv('/content/UCI_Credit_Card.csv')
    processor = DataPreprocessor(df)
    cleaned_df = processor.clean_data()
    engineered_df = processor.feature_engineering()
    
    try:
        processor.schema()(engineered_df)
        print("Pandera Validation Passed!")
    except Exception as e:
        print(f"Pandera Validation Failed: {e}")
    
    try:
        processor.validate_data()
        print("Great Expectations Validation Passed!")
    except Exception as e:
        print(f"Great Expectations Validation Failed: {e}")