import os
import subprocess
import sys
import numpy as np
import pathlib
import tarfile


if __name__ == "__main__":
    model_path = os.getenv('sm_model')
    with tarfile.open(f"{model_path}/model.tar.gz", 'r:gz') as tar:
        tar.extractall('./model')
        
    import tensorflow as tf
    model = tf.keras.models.load_model('./model/1')
    test_path = os.getenv("sm_input")
    x_test = np.load(os.path.join(test_path, 'x_test.npy'))
    y_test = np.load(os.path.join(test_path, 'y_test.npy'))
    scores = model.evaluate(x_test, y_test, verbose=2)
    print("\nTest MSE :", scores)
    
    output_dir = os.getenv("sm_output")
    
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    evaluation_path = f"{output_dir}/score-report.txt"
    with open(evaluation_path, 'w') as writer:
         writer.write(f"Test MSE : {scores}")