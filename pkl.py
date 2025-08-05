
import pandas as pd

# Load your pickle file
obj = pd.read_pickle('models/important_features.pkl')

# Or if you used joblib:
# import joblib
# obj = joblib.load('path/to/your_file.pkl')

print(type(obj))   # See what type it actually is
print(obj)         # Preview contents according to its type
