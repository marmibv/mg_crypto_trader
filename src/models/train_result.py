from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class TrainResult(Base):
    __tablename__ = 'train_results'

    id = Column(Integer, primary_key=True)
    # Single arguments
    symbol = Column(String)
    estimator = Column(String)
    model_name = Column(String)
    date_training = Column(DateTime)
    train_size = Column(Integer)
    start_train_date = Column(String)
    start_test_date = Column(String)
    regression_times = Column(Integer)
    times_regression_profit_and_loss = Column(Integer)
    stop_loss = Column(Integer)
    fold = Column(Integer)
    start_value = Column(Float)
    profit_and_loss = Column(Float)
    # List arguments
    numeric_features = Column(String)
    regression_features = Column(String)
    arguments = Column(String)
    # Boolean arguments
    use_all_data_to_train = Column(Boolean)
    no_tune = Column(Boolean)
