import pandas as pd
from imgnet.collections.store import IndexedDatasets

store = IndexedDatasets()

df = store.index("Totalsegmentator")
print(df.columns)