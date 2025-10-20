"""
模型训练器
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model_type: str = "lightgbm"):
        """
        初始化模型训练器
        
        Args:
            model_type: 模型类型 (lightgbm/xgboost/linear)
        """
        self.model_type = model_type
        self.model = None
        self.feature_importance = None
    
    def prepare_dataset(
        self,
        factors: pd.DataFrame,
        labels: pd.Series,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str
    ) -> tuple:
        """
        准备训练数据集
        
        Returns:
            (X_train, y_train, X_test, y_test)
        """
        # 训练集
        train_mask = (
            (factors.index.get_level_values(0) >= train_start) &
            (factors.index.get_level_values(0) <= train_end)
        )
        X_train = factors[train_mask]
        y_train = labels[train_mask]
        
        # 测试集
        test_mask = (
            (factors.index.get_level_values(0) >= test_start) &
            (factors.index.get_level_values(0) <= test_end)
        )
        X_test = factors[test_mask]
        y_test = labels[test_mask]
        
        # 去除NaN
        train_valid = ~(X_train.isna().any(axis=1) | y_train.isna())
        X_train = X_train[train_valid]
        y_train = y_train[train_valid]
        
        test_valid = ~(X_test.isna().any(axis=1) | y_test.isna())
        X_test = X_test[test_valid]
        y_test = y_test[test_valid]
        
        return X_train, y_train, X_test, y_test
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征（可选）
            y_val: 验证标签（可选）
            params: 模型参数
        """
        if params is None:
            params = self._get_default_params()
        
        if self.model_type == "lightgbm":
            import lightgbm as lgb
            
            train_data = lgb.Dataset(X_train, label=y_train)
            
            if X_val is not None and y_val is not None:
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                self.model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[train_data, val_data],
                    num_boost_round=1000,
                    callbacks=[lgb.early_stopping(50)]
                )
            else:
                self.model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=100
                )
            
            self.feature_importance = pd.Series(
                self.model.feature_importance(),
                index=X_train.columns
            ).sort_values(ascending=False)
        
        elif self.model_type == "xgboost":
            import xgboost as xgb
            
            dtrain = xgb.DMatrix(X_train, label=y_train)
            
            if X_val is not None and y_val is not None:
                dval = xgb.DMatrix(X_val, label=y_val)
                evals = [(dtrain, 'train'), (dval, 'val')]
                self.model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=1000,
                    evals=evals,
                    early_stopping_rounds=50,
                    verbose_eval=False
                )
            else:
                self.model = xgb.train(params, dtrain, num_boost_round=100)
            
            self.feature_importance = pd.Series(
                self.model.get_score(importance_type='weight'),
                index=X_train.columns
            ).sort_values(ascending=False)
        
        elif self.model_type == "linear":
            from sklearn.linear_model import Ridge
            
            self.model = Ridge(**params)
            self.model.fit(X_train, y_train)
            
            self.feature_importance = pd.Series(
                np.abs(self.model.coef_),
                index=X_train.columns
            ).sort_values(ascending=False)
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """预测"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if self.model_type == "lightgbm":
            predictions = self.model.predict(X)
        elif self.model_type == "xgboost":
            import xgboost as xgb
            dtest = xgb.DMatrix(X)
            predictions = self.model.predict(dtest)
        elif self.model_type == "linear":
            predictions = self.model.predict(X)
        
        return pd.Series(predictions, index=X.index)
    
    def _get_default_params(self) -> Dict:
        """获取默认参数"""
        if self.model_type == "lightgbm":
            return {
                "objective": "regression",
                "metric": "mse",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "verbose": -1
            }
        elif self.model_type == "xgboost":
            return {
                "objective": "reg:squarederror",
                "learning_rate": 0.05,
                "max_depth": 6
            }
        elif self.model_type == "linear":
            return {"alpha": 1.0}
        
        return {}