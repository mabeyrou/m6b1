from tensorflow.keras import datasets
from os.path import join, dirname, abspath
from datetime import datetime

from modules.models import create_cnn_model, train

(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

model = create_cnn_model()

trained_model, history = train(
    model=model,
    X=X_train,
    y=y_train,
    X_val=X_test,
    y_val=y_test,
    epochs=50,
)

current_dir = dirname(abspath(__file__))

models_dir = abspath(join("models"))

latest_model_name = "cnn_latest.keras"
current_date_time = datetime.now().strftime("%Y%m%d%H%M%S")
current_model_name = f"cnn_{current_date_time}.keras"

trained_model.save(join(models_dir, latest_model_name))
trained_model.save(join(models_dir, current_model_name))
