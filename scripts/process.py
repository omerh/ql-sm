import os
import pandas as pd
import numpy as np
import argparse
import pyarrow.parquet as pq
import pyarrow as pa


# Input files from S3 will downloaded to /opt/ml/processing/input/ env var for it is sm_input
# Output files from S3 should be saved at /opt/ml/processing/output/ env var for it is sm_output

if __name__ == "__main__":
    print("This is a processing job")
    
    print(os.environ)
    # df = pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})
    
    input_files_local_path = os.getenv('sm_input', '/opt/ml/input')
    output_files_local_path = os.getenv('sm_output', '/opt/ml/output')
    
    df_x = pd.DataFrame(np.load(f"{input_files_local_path}/x_train.npy"))
    pdf_x = pa.Table.from_pandas(df_x)
    pq.write_table(pdf_x, f"{output_files_local_path}/x_train.parquet.gzip",
                   compression='gzip')
    
    df_y = pd.DataFrame(np.load(f"{input_files_local_path}/y_train.npy"))
    pdf_y = pa.Table.from_pandas(df_y)
    pq.write_table(pdf_y,f"{output_files_local_path}y_train.parquet.gzip",
                   compression='gzip')

    
