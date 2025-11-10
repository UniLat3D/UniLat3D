import numpy as np 

if __name__ == "__main__":
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 3]])
    b = np.array([[9, 7, 8], [6, 5, 4], [3, 2, 1]])
    
    print(b @ a[0])